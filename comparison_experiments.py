"""Focused experiments using the personal factor MLP and existing H samplers.

The population oracle is evaluation-only. All reported arrays are raw returns.
No author backend, pretrained weights, or unmerged PR is required.
"""
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import importlib.metadata
import json
from pathlib import Path
import platform
import re
from time import perf_counter
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from scipy.special import log_ndtr, ndtri_exp
from scipy.stats import norm
import torch
from threadpoolctl import threadpool_limits

from DFM import train_dfm
from guided_evaluation import conditional_diagnostics
from guided_sampling import (generate_conditional_samples, sample_conditional_initial,
                             sample_reference_conditional)
from h_setup import train_h_from_checkpoint
from return_preprocessing import NumpyStandardizer
from sampling import generate_samples

FRENCH_URL = ('https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
              '10_Industry_Portfolios_daily_CSV.zip')


@dataclass(frozen=True)
class ExperimentConfig:
    k: int = 4
    dfm_epochs: int = 1000
    h_epochs: int = 40
    train_paths: int = 4096
    validation_paths: int = 1024
    steps: int = 128
    samples: int = 512
    reference_samples: int = 2048
    repeats: int = 3
    batch_size: int = 128
    max_proposals: int = 200000
    T: float = 5.0
    t0: float = 0.01
    seed: int = 20261023

    @classmethod
    def smoke(cls):
        # Execution check only; not a scientific accuracy preset.
        return cls(dfm_epochs=10, h_epochs=2, train_paths=1024,
                   validation_paths=256, steps=32, samples=64,
                   reference_samples=256, repeats=1)


def paper_population(d=32, k=8, seed=617):
    """Gaussian recipe of DFM Appendix D; reduced dimensions by default."""
    if not 1 <= k < d:
        raise ValueError('Require 1 <= k < d')
    rng = np.random.default_rng(seed)
    factor_mean = .1 * (1 - rng.random(k))
    loadings = rng.normal(size=(d, k))
    noise_std = .4 * (1 - rng.random(d))
    mean = loadings @ factor_mean
    cov = (loadings * (1.5 * factor_mean)**2) @ loadings.T + np.diag(noise_std**2)
    return mean, cov


def gaussian_tail(mean, cov, weights, threshold):
    """Exact mean and covariance of R | w'R <= c for Gaussian R."""
    mean, cov, w = (np.asarray(x, dtype=float) for x in (mean, cov, weights))
    if (mean.ndim != 1 or cov.shape != (len(mean), len(mean)) or w.shape != mean.shape
            or not all(np.isfinite(x).all() for x in (mean, cov, w))
            or not np.isfinite(threshold) or not np.allclose(cov, cov.T)):
        raise ValueError('Require finite, matching moments and event')
    np.linalg.cholesky(cov)
    variance = float(w @ cov @ w)
    if variance <= 0:
        raise ValueError('Portfolio variance must be positive')
    sd = np.sqrt(variance)
    a = (threshold - w @ mean) / sd
    mills = np.exp(norm.logpdf(a) - log_ndtr(a))
    v = cov @ w / sd
    target_cov = cov - np.outer(v, v) * mills * (a + mills)
    return mean - mills * v, (target_cov + target_cov.T) / 2


def gaussian_tail_draws(mean, cov, w, threshold, n, seed):
    """Exact independent oracle draws via a Gaussian residual decomposition."""
    gaussian_tail(mean, cov, w, threshold)
    rng = np.random.default_rng(seed)
    x = rng.multivariate_normal(mean, cov, size=n)
    variance = w @ cov @ w
    residual = x - mean - np.outer((x - mean) @ w, cov @ w / variance)
    a = (threshold - w @ mean) / np.sqrt(variance)
    z = ndtri_exp(log_ndtr(a) + np.log1p(-rng.random(n)))
    return mean + np.outer(z, cov @ w / np.sqrt(variance)) + residual


def read_french_zip(path):
    """Read only the value-weighted daily table; percentages become decimals."""
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith('.csv')]
        if len(names) != 1:
            raise ValueError('Expected one CSV in the French archive')
        lines = archive.read(names[0]).decode('utf-8-sig').splitlines()
    marker = next(i for i, line in enumerate(lines)
                  if 'average value weighted returns -- daily' in line.lower())
    first = next(i for i in range(marker + 1, len(lines)) if re.match(r'^\s*\d{8},', lines[i]))
    last = first
    while last < len(lines) and re.match(r'^\s*\d{8},', lines[last]):
        last += 1
    frame = pd.read_csv(io.StringIO('\n'.join([lines[first-1], *lines[first:last]])), index_col=0)
    frame.columns = frame.columns.str.strip()
    frame.index = pd.to_datetime(frame.index.astype(str), format='%Y%m%d')
    frame = frame.replace([-99.99, -999.0], np.nan) / 100.0
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError('Expected unique chronological daily observations')
    return frame


def load_market_returns(cache, csv_path=None):
    """Public ten-industry daily returns or user's dated decimal-return CSV.

    No imputation or silent synthetic fallback. Keep/hash the input snapshot.
    """
    path = Path(csv_path) if csv_path else Path(cache)
    if csv_path:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        source = 'user-supplied decimal simple-return CSV'
    else:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(FRENCH_URL, timeout=60) as response:
                content = response.read()
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                archive.testzip()
            path.write_bytes(content)
        frame = read_french_zip(path)
        source = FRENCH_URL
    if (not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans
            or not frame.index.is_unique or not frame.index.is_monotonic_increasing
            or not frame.columns.is_unique or frame.shape[1] < 2):
        raise ValueError('Need unique ordered dates and at least two return columns')
    frame = frame.apply(pd.to_numeric, errors='raise')
    if np.isinf(frame.to_numpy()).any():
        raise ValueError('Infinite return found')
    # Fixed split is explicit, and missing rows are counted rather than filled.
    selected = frame.loc['2010-01-01':'2024-12-31']
    clean = selected.dropna()
    train, test = clean.loc[:'2019-12-31'], clean.loc['2020-01-01':]
    if min(len(train), len(test)) < 20:
        raise ValueError('Need >=20 complete rows in both 2010–2019 and 2020–2024')
    metadata = dict(source=source, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    units='decimal daily simple returns', dropped_rows=len(selected)-len(clean),
                    train_rows=len(train), test_rows=len(test), assets=list(clean.columns),
                    train_dates=[str(train.index[0].date()), str(train.index[-1].date())],
                    test_dates=[str(test.index[0].date()), str(test.index[-1].date())])
    return train, test, metadata


def filtered_guidance(h, n, *, steps, batch_size, max_proposals, generator):
    """H tilt + guidance + terminal filtering, with two global proposal budgets.

    Filtering enforces the event, not the correctness of the conditional law.
    Initial rejected proposals and terminal overshoot are both accounted for.
    """
    accepted, proposals, initial_proposals, count = [], 0, 0, 0
    while count < n and proposals < max_proposals:
        size = min(batch_size, max_proposals - proposals)
        remaining = max_proposals - initial_proposals
        if remaining < size:
            raise RuntimeError('Filtered H exhausted its total initial-proposal budget')
        starts, initial = sample_conditional_initial(h, size, batch_size=batch_size,
            max_proposals=remaining, generator=generator)
        initial_proposals += initial['proposals']
        result = generate_conditional_samples(h, size, num_steps=steps,
            initial_state=starts, batch_size=batch_size, max_proposals=max_proposals,
            generator=generator)
        keep = result.samples @ h.w <= h.C
        accepted.append(result.samples[keep])
        count += int(keep.sum())
        proposals += size
    if count < n:
        raise RuntimeError(f'Filtered H exhausted {proposals} terminal proposals: {count}/{n}')
    return torch.cat(accepted)[:n], dict(terminal_proposals=proposals,
        terminal_accepted=count, returned=n, terminal_discarded=proposals-n,
        initial_proposals=initial_proposals, initial_discarded=initial_proposals-proposals,
        acceptance_rate=count/proposals, terminal_filtering=True)


def moment_errors(x, mean, cov):
    estimated = np.cov(x, rowvar=False, ddof=1)
    def corr(c):
        sd = np.sqrt(np.diag(c))
        return c / np.outer(sd, sd)
    return dict(covariance_relative_error=float(np.linalg.norm(estimated-cov)/np.linalg.norm(cov)),
                mean_rmse=float(np.sqrt(np.mean((x.mean(0)-mean)**2))),
                correlation_rmse=float(np.sqrt(np.mean((corr(estimated)-corr(cov))**2))))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run_comparison(training, weights, threshold, output, config=ExperimentConfig(), *,
                   assets=None, population=None, heldout=None, provenance=None):
    """Train once, freeze, fit H, then compare three equal-count sampling methods.

    population=(mean,cov) is used ONLY after fitting, for evaluation.
    A fresh larger rejection sample measures guidance error; the analytical
    population target measures total error, including DFM fitting error.
    """
    training, weights = np.asarray(training, dtype=float), np.asarray(weights, dtype=float)
    cfg = config
    if training.ndim != 2 or not 1 <= cfg.k < training.shape[1]:
        raise ValueError('Require a return matrix and 1 <= k < d')
    for name in ('dfm_epochs', 'h_epochs', 'train_paths', 'validation_paths', 'steps',
                 'samples', 'reference_samples', 'repeats', 'batch_size', 'max_proposals'):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if min(cfg.samples, cfg.reference_samples) < 2 or not 0 < cfg.t0 < cfg.T:
        raise ValueError('Need >=2 samples and 0 < t0 < T')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    assets = list(assets) if assets is not None else [f'asset_{i}' for i in range(training.shape[1])]
    report = dict(status='running', config=asdict(cfg), provenance=provenance,
        implementation='personal factor MLP, OU/Euler, float32 CPU; no author backend',
        started_at=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
        versions={'python': platform.python_version(), **{name: importlib.metadata.version(name)
                  for name in ('torch', 'numpy', 'scipy', 'pandas', 'matplotlib', 'threadpoolctl',
                               'nbformat', 'nbclient', 'ipykernel')}},
        assets=assets, threshold=float(threshold),
        weights=weights.tolist(), rows=[], failures=[], costs={})
    # Store exact source snapshots, since git HEAD alone omits uncommitted edits.
    sources = output / 'sources'
    sources.mkdir()
    report['source_hashes'] = {}
    root = Path(__file__).resolve().parent
    for source in [*root.glob('*.py'), root / 'focused_comparison.ipynb',
                   root / 'requirements-comparison.txt', root / 'scripts/run_comparison_notebook.py']:
        if source.exists():
            content = source.read_bytes()
            if source.suffix == '.ipynb':
                notebook = json.loads(content)
                for cell in notebook['cells']:
                    if cell['cell_type'] == 'code':
                        cell['outputs'] = []
                        cell['execution_count'] = None
                    cell.get('metadata', {}).pop('execution', None)
                content = json.dumps(notebook, indent=1).encode()
            (sources / source.name).write_bytes(content)
            report['source_hashes'][source.name] = hashlib.sha256(content).hexdigest()
    arrays = dict(training_raw=training)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1), torch.random.fork_rng(devices=[]):
            scaler = NumpyStandardizer.fit(training, assets)
            scaler.event(weights, threshold)  # Validate before expensive training.
            started = perf_counter()
            torch.manual_seed(cfg.seed)
            with (output / 'dfm_training.log').open('w') as log, redirect_stdout(log):
                model, history = train_dfm(scaler.transform(training), cfg.k,
                    epochs=cfg.dfm_epochs, batch_size=cfg.batch_size, T=cfg.T, t0=cfg.t0,
                    hidden_dims=[64, 64], device='cpu')
            report['costs']['dfm_training_seconds'] = perf_counter() - started
            report['dfm_loss'] = history
            checkpoint = output / 'dfm.pt'
            torch.save(dict(state_dict=model.state_dict(), assets=assets,
                model_config=dict(d=training.shape[1], k=cfg.k, sigma_max=1., T=cfg.T,
                                  t0=cfg.t0, hidden_dims=[64, 64]),
                preprocessing=dict(mean=scaler.mean.tolist(), std=scaler.std.tolist())), checkpoint)
            print(f'{output.name}: DFM fitted; generating H labels and fitting H', flush=True)
            started = perf_counter()
            h, metadata = train_h_from_checkpoint(checkpoint,
                weights=dict(zip(assets, weights)), threshold=float(threshold),
                num_steps=cfg.steps, train_trajectories=cfg.train_paths,
                held_out_trajectories=cfg.validation_paths, epochs=cfg.h_epochs,
                batch_size=cfg.batch_size, rollout_batch_size=cfg.batch_size,
                seed=cfg.seed+100)
            report['costs']['h_setup_including_validation_seconds'] = perf_counter()-started
            report['h'] = metadata
            h.save_checkpoint(output / 'h.pt', metadata=metadata)
            report['preprocessing'] = dict(mean=scaler.mean.tolist(), std=scaler.std.tolist())
            print(f'{output.name}: H fitted; benchmarking equal-count draws', flush=True)
            def draw(method, n, seed):
                g = torch.Generator().manual_seed(seed)
                settings = dict(num_steps=cfg.steps, batch_size=cfg.batch_size,
                                max_proposals=cfg.max_proposals, generator=g)
                if method == 'h_filtered':
                    return filtered_guidance(h, n, steps=cfg.steps, batch_size=cfg.batch_size,
                                             max_proposals=cfg.max_proposals, generator=g)
                result = (sample_reference_conditional(h, n, **settings) if method == 'dfm_rejection'
                          else generate_conditional_samples(h, n, **settings))
                return result.samples, result.diagnostics
            started = perf_counter()
            reference, info = draw('dfm_rejection', cfg.reference_samples, cfg.seed+1000)
            arrays['fitted_reference_raw'] = scaler.inverse(reference.numpy())
            arrays['fitted_reference_covariance'] = np.cov(arrays['fitted_reference_raw'], rowvar=False)
            report['costs']['independent_reference_seconds'] = perf_counter()-started
            report['reference_diagnostics'] = info
            # Check unconditional DFM fit independently of H/conditional accuracy.
            unconditional = generate_samples(h.reference, cfg.reference_samples, num_steps=cfg.steps,
                generator=torch.Generator().manual_seed(cfg.seed+1001)).numpy()
            arrays['unconditional_dfm_raw'] = scaler.inverse(unconditional)
            targets = {'fitted_dfm_conditional': (arrays['fitted_reference_raw'].mean(0),
                np.cov(arrays['fitted_reference_raw'], rowvar=False), arrays['fitted_reference_raw'])}
            if population is not None:
                mean, cov = population
                arrays['population_mean'], arrays['population_covariance'] = mean, cov
                cm, cc = gaussian_tail(mean, cov, weights, threshold)
                arrays['true_conditional_mean'], arrays['true_conditional_covariance'] = cm, cc
                oracle = gaussian_tail_draws(mean, cov, weights, threshold,
                                            cfg.reference_samples, cfg.seed+1002)
                arrays['oracle_raw'] = oracle
                targets['true_population_conditional'] = (cm, cc, oracle)
                report['unconditional_fit'] = moment_errors(arrays['unconditional_dfm_raw'], mean, cov)
                report['empirical_training_fit'] = moment_errors(training, mean, cov)
                report['oracle_sampling_error'] = moment_errors(oracle[:cfg.samples], cm, cc)
            if heldout is not None:
                heldout = np.asarray(heldout, dtype=float)
                arrays['heldout_raw'] = heldout
                tail = heldout[heldout @ weights <= threshold]
                arrays['heldout_tail_raw'] = tail
                report['heldout_event_count'] = len(tail)
                report['heldout_event_rate'] = len(tail)/len(heldout)
                report['unconditional_fit'] = moment_errors(arrays['unconditional_dfm_raw'],
                    heldout.mean(0), np.cov(heldout, rowvar=False))
                if len(tail) >= 2:
                    targets['historical_heldout_tail_descriptive'] = (tail.mean(0),
                        np.cov(tail, rowvar=False), tail)
            methods = ('dfm_rejection', 'h_raw', 'h_filtered')
            for i, method in enumerate(methods):
                draw(method, min(16, cfg.samples), cfg.seed+2000+i)
            for repeat in range(cfg.repeats):
                order = methods[repeat % 3:] + methods[:repeat % 3]
                for position, method in enumerate(order):
                    seed = cfg.seed+3000+10*repeat+methods.index(method)
                    started = perf_counter()
                    try:
                        values, diagnostics = draw(method, cfg.samples, seed)
                        seconds = perf_counter()-started
                        raw = scaler.inverse(values.numpy())
                        arrays[f'{method}_{repeat}_raw'] = raw
                        arrays[f'{method}_{repeat}_covariance'] = np.cov(raw, rowvar=False)
                        for target, (mean, cov, reference) in targets.items():
                            quality = conditional_diagnostics(raw, reference, weights, threshold,
                                                              seed=cfg.seed+9000)
                            quality.update(moment_errors(raw, mean, cov))
                            quality['terminal_filtering'] = method == 'h_filtered'
                            report['rows'].append(dict(method=method, repeat=repeat, position=position,
                                seed=seed, target=target, seconds=seconds,
                                diagnostics=diagnostics, **quality))
                    except RuntimeError as error:
                        report['failures'].append(dict(method=method, repeat=repeat, seed=seed,
                            seconds=perf_counter()-started, error=str(error)))
            report['status'] = 'complete' if not report['failures'] else 'completed_with_failures'
    except Exception as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        torch.set_num_threads(previous_threads)
        np.savez_compressed(output / 'samples.npz', **arrays)
        _write_json(output / 'report.json', report)
        pd.DataFrame(report['rows']).drop(columns=['diagnostics'], errors='ignore').to_csv(
            output / 'metrics.csv', index=False)
    return report, arrays


def comparison_figures(report, arrays):
    """Covariance matrices, portfolio CDFs, marginal quantiles, accuracy and cost."""
    import matplotlib.pyplot as plt
    w = np.asarray(report['weights'])
    reference = arrays.get('oracle_raw', arrays['fitted_reference_raw'])
    cov = arrays.get('true_conditional_covariance', np.cov(reference, rowvar=False))
    methods = [m for m in ('dfm_rejection', 'h_raw', 'h_filtered') if f'{m}_0_raw' in arrays]
    target_label = 'True population' if 'oracle_raw' in arrays else 'Fitted DFM reference'
    colors = {'dfm_rejection': 'tab:blue', 'h_raw': 'tab:orange', 'h_filtered': 'tab:green',
              target_label: 'black', 'Historical held-out tail': 'tab:purple'}
    fig, axes = plt.subplots(1, len(methods)+1, figsize=(4*(len(methods)+1), 3.4))
    limit = np.abs(cov).max()
    for ax, title, matrix in zip(np.atleast_1d(axes), [target_label]+methods,
            [cov]+[np.cov(arrays[f'{m}_0_raw'], rowvar=False) for m in methods]):
        im = ax.imshow(matrix, cmap='RdBu_r', vmin=-limit, vmax=limit)
        ax.set_title(title); ax.set_xlabel('Asset'); ax.set_ylabel('Asset')
    fig.colorbar(im, ax=list(np.atleast_1d(axes)), label='Covariance (return units squared)', shrink=.8)
    fig.suptitle('Conditional covariance — first sampling repeat', y=1.04)
    fig2, axes = plt.subplots(1, 3, figsize=(14, 3.7))
    distributions = [(target_label, reference)]+[(m, arrays[f'{m}_0_raw']) for m in methods]
    if len(arrays.get('heldout_tail_raw', [])) >= 2:
        distributions.append(('Historical held-out tail', arrays['heldout_tail_raw']))
    for label, x in distributions:
        pnl = np.sort(x @ w)
        axes[0].plot(pnl, np.arange(1, len(pnl)+1)/len(pnl), label=label, color=colors[label])
        axes[1].plot(np.quantile(reference[:, 0], np.linspace(.01,.99,99)),
                     np.quantile(x[:, 0], np.linspace(.01,.99,99)), label=label, color=colors[label])
    axes[0].axvline(report['threshold'], color='black', ls='--', lw=1)
    axes[0].set(xlabel='Portfolio return', ylabel='Empirical CDF', title='All returned draws')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Target asset-0 quantiles', ylabel='Generated quantiles', title='Marginal distribution')
    rows = pd.DataFrame(report['rows'])
    for m in methods:
        selected = rows[(rows.method == m) & (rows.target == 'fitted_dfm_conditional')]
        axes[2].scatter(selected.seconds, selected.covariance_relative_error, label=m, color=colors[m])
    axes[2].set(xlabel='Generation seconds', ylabel='Covariance relative error', title='Fitted-DFM target / all repeats')
    axes[2].legend(fontsize=8)
    fig2.tight_layout()
    return fig, fig2
