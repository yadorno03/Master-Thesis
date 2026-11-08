"""DFM upgrade campaign: I0-I4 on P1 and P2 at d=500.

Populations
  P1  Gaussian population of Chen et al. (2025), d=500, k=16, raw return units.
  P2  Gaussian population calibrated on 500 standardized real equities (training 2010-2019),
      k=8: PCA loadings, factor variances and residual variances of the panel.
  P3  P2 with multivariate Student-t factors (5 degrees of freedom, same covariance): heavy tails.
Arms
  base            current DFM (Eq. 18, width 64)
  precond_pc1     I2, only the first principal direction shrunk (rho=20)
  precond_rho20   I2, every leading direction with lambda > 20 c_bar shrunk
  gaussres        I3, Gaussian factor posterior + zero-initialized MLP
  stdin           I4, inputs of g standardized per coordinate and per time
  gauss_ref       I0, fitted Gaussian reference (no training, no checkpoint)
Every trained arm keeps two checkpoints: best validation DSM and best I1 criterion. Selection
uses training/validation rows only; evaluation uses reserved population draws, in common
(original) coordinates.

  modal run --detach scripts/dfm_upgrades.py::screen --campaign screen1
  modal run --detach scripts/dfm_upgrades.py::confirm --campaign confirm1 --arms base,gaussres --seeds 314,2718
  python scripts/dfm_upgrades.py summarize artifacts/dfm_upgrades/screen1/results.json
"""
from copy import deepcopy
from pathlib import Path
import json
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PANEL = REPO / 'artifacts/market_data_cleaning/20261029T104817954744Z/train_complete.npz'
ARMS = ('base', 'precond_pc1', 'precond_rho20', 'gaussres', 'stdin', 'gauss_ref')
T, T0, STEPS, WIDTH, LR, BATCH = 8.0, 0.005, 400, 64, 1e-3, 256


# ----------------------------------------------------------------------------- populations
def panel_population_spec(d=500, k=8, cohort_seed=500):
    """Gaussian population calibrated on a seeded 500-asset cohort of the training panel."""
    returns = np.load(PANEL, allow_pickle=False)['returns_decimal_float64']
    columns = np.sort(np.random.default_rng(cohort_seed).permutation(returns.shape[1])[:d])
    x = returns[:, columns]
    z = (x - x.mean(0)) / x.std(0, ddof=1)
    covariance = np.cov(z, rowvar=False)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    eigenvalues, vectors = eigenvalues[::-1], vectors[:, ::-1]
    B = vectors[:, :k]
    residual = z - (z @ B) @ B.T
    D = residual.var(0, ddof=1)
    S = np.diag(eigenvalues[:k] - np.einsum('dk,d,dk->k', B, D, B))
    info = dict(n_days=len(z), columns=columns.tolist(), eigenvalues=eigenvalues[:12].tolist(),
                c_bar=float(D.mean()), lambda1_over_c=float(eigenvalues[0] / D.mean()))
    return dict(name='P2', B=B.tolist(), D=D.tolist(), S=S.tolist(), k=k, info=info)


class TFactorPopulation:
    """P3: panel-calibrated loadings, multivariate Student-t factors with covariance S, Gaussian residuals.

    Exact moments are known; portfolio distributions use a large fixed Monte Carlo reference
    (only portfolio values are stored). Conditional draws use exact rejection.
    """

    def __init__(self, B, D, S, nu, reference_size=400_000, seed=8080):
        self.B, self.D, self.nu = np.asarray(B), np.asarray(D), float(nu)
        self.S = np.asarray(S)[None]
        self.m, self.pi = np.zeros((1, self.B.shape[1])), np.ones(1)
        self._L = np.linalg.cholesky(self.S[0])
        rng = np.random.default_rng(seed)
        weights = portfolio_weights(self)
        values = {name: [] for name in weights}
        for _ in range(reference_size // 20_000):
            x = self._draw(20_000, rng)
            for name, w in weights.items():
                values[name].append(x @ w)
        self.reference_portfolios = {name: np.sort(np.concatenate(v)) for name, v in values.items()}

    d = property(lambda self: len(self.D))
    k = property(lambda self: self.B.shape[1])
    w = property(lambda self: np.ones(len(self.D)) / len(self.D))
    mu = property(lambda self: np.zeros((1, len(self.D))))
    covs = property(lambda self: (self.B @ self.S[0] @ self.B.T + np.diag(self.D))[None])

    def _draw(self, n, rng):
        scale = np.sqrt((self.nu - 2) / rng.chisquare(self.nu, n))[:, None]
        factors = scale * (rng.standard_normal((n, self.k)) @ self._L.T)
        return factors @ self.B.T + rng.standard_normal((n, self.d)) * np.sqrt(self.D)

    def moments(self, c=None):
        if c is not None:
            raise NotImplementedError('conditional moments are not available for P3')
        return self.mu[0], self.covs[0]

    def portfolio_cdf(self, name='equal'):
        ref = self.reference_portfolios[name]
        return lambda v: np.searchsorted(ref, np.asarray(v), side='right') / len(ref)

    def cdf(self, x, c=None):
        return self.portfolio_cdf()(x)

    def threshold(self, probability):
        return float(np.quantile(self.reference_portfolios['equal'], probability))

    def evaluate_generated_samples(self, x):
        return float('nan'), float('nan')

    def sample(self, n, rng, c=None):
        if c is None:
            return self._draw(n, rng)
        accepted, total = [], 0
        while total < n:
            x = self._draw(20_000, rng)
            x = x[x @ self.w <= c]
            accepted.append(x)
            total += len(x)
        return np.concatenate(accepted)[:n]


def build_population(spec):
    from synthetic_conditional_truth import FactorPopulation, make_population
    if spec['name'] == 'P1':
        return make_population(spec['d'], 'gaussian', k=spec['k'])
    if spec['name'] == 'P3':
        return TFactorPopulation(np.array(spec['B']), np.array(spec['D']), np.array(spec['S']), spec['nu'])
    B, S = np.array(spec['B']), np.array(spec['S'])
    return FactorPopulation(B, np.array(spec['D']), np.ones(1), np.zeros((1, B.shape[1])), S[None],
                            dict(population_seed=-1, design='panel-calibrated-gaussian'))


# ----------------------------------------------------------------------------- evaluation
def factor_metrics(x, pop):
    ev, U = np.linalg.eigh(pop.S[0])
    order = ev.argsort()[::-1]
    ev, U = ev[order], U[:, order]
    E = pop.B @ U
    true_var = ev + np.einsum('dk,d,dk->k', E, pop.D, E)
    identifiable = ev >= 5 * pop.D.mean()
    cov = np.cov(x, rowvar=False)
    ratios = np.einsum('dk,de,ek->k', E, cov, E)[identifiable] / true_var[identifiable]
    resid_gen = np.trace(cov) - np.einsum('dk,de,ek->', pop.B, cov, pop.B)
    resid_true = pop.D.sum() - (pop.D * (pop.B ** 2).sum(1)).sum()
    true_cov = pop.covs[0]
    top = np.linalg.eigvalsh(cov)[::-1][:3] / np.linalg.eigvalsh(true_cov)[::-1][:3]
    return dict(factor_ratios=ratios.tolist(), factor_ratio_min=float(ratios.min()),
                factor_ratio_max=float(ratios.max()), factor_ratio_median=float(np.median(ratios)),
                factor_mean_abs_log=float(np.abs(np.log(ratios)).mean()),
                residual_ratio=float(resid_gen / resid_true), top3_eigen_ratio=top.tolist())


def portfolio_weights(pop):
    cov = pop.covs[0]
    _, vectors = np.linalg.eigh(cov)
    pc1, pc2 = vectors[:, -1], vectors[:, -2]
    ls = np.random.default_rng(77).normal(size=pop.d)
    return dict(equal=pop.w, pc1=pc1 * np.sign(pc1.sum()), pc2=pc2, long_short=ls - ls.mean())


def portfolio_metrics(x, pop):
    """KS, 5 %/1 % tail frequencies, sd ratio and excess kurtosis of predefined portfolios."""
    from scipy.stats import kstest, kurtosis, norm
    out = {}
    mean, cov = pop.mu[0], pop.covs[0]
    for name, w in portfolio_weights(pop).items():
        p = x @ w
        if hasattr(pop, 'reference_portfolios'):
            ref = pop.reference_portfolios[name]
            cdf, (q5, q1), sd, true_kurt = pop.portfolio_cdf(name), np.quantile(ref, [0.05, 0.01]), ref.std(), kurtosis(ref)
        else:
            mu, sd = w @ mean, np.sqrt(w @ cov @ w)
            cdf, q5, q1, true_kurt = (lambda v, mu=mu, sd=sd: norm.cdf(v, mu, sd)), mu + sd * norm.ppf(0.05), mu + sd * norm.ppf(0.01), 0.0
        out[name] = dict(ks=float(kstest(p, cdf).statistic), tail5=float(np.mean(p <= q5)), tail1=float(np.mean(p <= q1)),
                         sd_ratio=float(p.std(ddof=1) / sd), kurtosis=float(kurtosis(p)), true_kurtosis=float(true_kurt))
    return out


def evaluate_unconditional(x, pop, truth):
    from synthetic_benchmark.reporting import compare
    keep = ('mean_error', 'covariance_error', 'correlation_error', 'portfolio_ks',
            'portfolio_population_ks', 'sliced_w1', 'diversity_trace_ratio', 'event_0.05', 'event_0.01')
    base = compare(x, truth, pop)
    return dict({k: float(base[k]) for k in keep}, **factor_metrics(x, pop),
                portfolios=portfolio_metrics(x, pop))


def sliced_w1(a, b, seed=781, projections=64):
    from scipy.stats import wasserstein_distance
    directions = np.random.default_rng(seed).normal(size=(projections, a.shape[1]))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return float(np.mean([wasserstein_distance(a @ u, b @ u) for u in directions]))


def evaluate_conditional(accepted, proposals, hits, pop, threshold, cond_truth, cond_truth2):
    from scipy.stats import kstest, norm
    w = pop.w
    if hasattr(pop, 'reference_portfolios'):
        base_cdf = pop.portfolio_cdf('equal')
        p_event = float(base_cdf(threshold))
    else:
        mu, sd = w @ pop.mu[0], np.sqrt(w @ pop.covs[0] @ w)
        base_cdf = lambda v: norm.cdf(v, mu, sd)  # noqa: E731
        p_event = norm.cdf(threshold, mu, sd)

    def cdf(v):
        return np.clip(base_cdf(v) / p_event, 0, 1)
    n = min(len(accepted), len(cond_truth))
    if n < 2:  # a degenerate model can miss the event entirely
        return dict(accepted=int(len(accepted)), proposals=int(proposals), acceptance=float(hits / max(proposals, 1)),
                    nominal=float(p_event), sliced_w1_vs_truth=float('nan'), sliced_w1_reference=float('nan'),
                    portfolio_ks_exact=float('nan'), portfolio_ks_reference=float('nan'))
    return dict(accepted=int(len(accepted)), proposals=int(proposals),
                acceptance=float(hits / max(proposals, 1)), nominal=float(p_event),
                sliced_w1_vs_truth=sliced_w1(accepted[:n], cond_truth[:n]),
                sliced_w1_reference=sliced_w1(cond_truth2[:n], cond_truth[:n]),
                portfolio_ks_exact=float(kstest(accepted @ w, cdf).statistic),
                portfolio_ks_reference=float(kstest(cond_truth2 @ w, cdf).statistic))


# ----------------------------------------------------------------------------- one run
def run_one(spec):
    """Train/evaluate one arm on one population, N and seed. Returns a JSON-ready dict."""
    import torch
    from DFM import UnconditionalDFM
    from dfm_variants import (
        AnalyticGaussianDFM, FactorPreconditioner, GaussianResidualScoreNetwork,
        StandardizedInputScoreNetwork, factor_selection_score, fit_gaussian_factor_model,
        sample_gaussian_factor_model, selection_reference,
    )
    from sampling import generate_samples
    from synthetic_benchmark.configuration import Streams
    from types import SimpleNamespace
    from synthetic_benchmark.unconditional import (
        dfm_training_loss, dfm_training_times, initialize_dfm_from_observations,
    )
    torch.set_num_threads(spec.get('threads', 4))
    started = time.perf_counter()
    pop = build_population(spec['population'])
    d, k = pop.d, pop.k
    arm, N, seed = spec['arm'], spec['N'], spec['seed']
    epochs, n_eval, n_cond = spec.get('epochs', 1000), spec.get('n_eval', 3000), spec.get('n_cond', 1000)
    select_every, steps = spec.get('select_every', 50), spec.get('steps', STEPS)
    n_val = 3000 if N >= 12000 else 750
    streams = Streams(seed)
    x_train = pop.sample(N, streams.np('train'))
    x_val = pop.sample(n_val, streams.np('validation'))
    # Reserved evaluation draws: identical for every arm and seed, never used for selection.
    truth = pop.sample(n_eval, np.random.default_rng(9001))
    threshold = pop.threshold(0.05)
    cond_truth = pop.sample(n_cond, np.random.default_rng(9003), c=threshold)
    cond_truth2 = pop.sample(n_cond, np.random.default_rng(9004), c=threshold)
    result = dict(spec={k_: v for k_, v in spec.items() if k_ != 'population'},
                  population=spec['population']['name'], d=d, k=k, n_val=n_val,
                  reference=evaluate_unconditional(pop.sample(n_eval, np.random.default_rng(9002)), pop, truth))

    def rejection(sample_fn, max_proposals):
        """Accepted rows (first n_cond), proposals drawn, and all event hits before truncation."""
        accepted, total = [], 0
        while sum(map(len, accepted)) < n_cond and total < max_proposals:
            x = sample_fn()
            total += len(x)
            accepted.append(x[x @ pop.w <= threshold])
        hits = sum(map(len, accepted))
        return np.concatenate(accepted)[:n_cond], total, hits

    if arm == 'gauss_ref':
        params = fit_gaussian_factor_model(x_train, k)
        direct = sample_gaussian_factor_model(params, n_eval, np.random.default_rng(9100))
        model = AnalyticGaussianDFM(params, T, T0)
        sampled = generate_samples(model, n_eval, num_steps=steps,
                                   generator=torch.Generator().manual_seed(99)).double().numpy()
        rng = np.random.default_rng(9101)
        acc, total, hits = rejection(lambda: sample_gaussian_factor_model(params, 4096, rng), 2_000_000)
        result.update(direct=evaluate_unconditional(direct, pop, truth),
                      sampled=evaluate_unconditional(sampled, pop, truth),
                      conditional_direct=evaluate_conditional(acc, total, hits, pop, threshold, cond_truth, cond_truth2),
                      seconds=time.perf_counter() - started)
        return result

    pre = None
    if arm.startswith('precond'):
        pre = FactorPreconditioner.fit(x_train, k, rho=20.0, mode='pc1' if arm == 'precond_pc1' else 'rho')
        result['preconditioner'] = pre.info

    def to_model(x):
        return pre.forward(x) if pre else x

    def to_common(x):
        return pre.inverse(x) if pre else x

    xt = torch.tensor(to_model(x_train), dtype=torch.float32)
    xv = torch.tensor(to_model(x_val), dtype=torch.float32)
    torch.manual_seed(streams.seeds['model_init'])
    model = UnconditionalDFM(d, k, sigma_max=2.0, T=T, t0=T0, hidden_dims=[WIDTH, WIDTH])
    if arm == 'gaussres':
        model.factor_score_network = GaussianResidualScoreNetwork(d, k, 2.0, (WIDTH, WIDTH))
    elif arm == 'stdin':
        model.factor_score_network = StandardizedInputScoreNetwork(d, k, 2.0, (WIDTH, WIDTH), T=T, t0=T0)
    net = model.factor_score_network
    initialize_dfm_from_observations(model, xt)
    if arm == 'gaussres':
        net.initialize_prior(xt)
    stats_rng = torch.Generator().manual_seed(seed + 17)
    if arm == 'stdin':
        net.refresh_input_statistics(xt, stats_rng)
    cfg = SimpleNamespace(t0=T0, T=T, dfm_time_sampling='log_balanced', device='cpu')
    rng, vrng = streams.torch('dfm_noise'), streams.torch('dfm_validation_noise')
    vt = dfm_training_times(len(xv), cfg, vrng)
    vz = torch.randn(xv.shape, generator=vrng)
    others = [p for name, p in net.named_parameters() if name != 'V']
    opt = torch.optim.Adam([{'params': [net.V], 'lr': LR * 0.1}, {'params': others, 'lr': LR}])
    reference = selection_reference(x_val, k)
    best = dict(dsm=(np.inf, None, 0), i1=(np.inf, None, 0))
    history = []
    train_started = time.perf_counter()

    def sample(n, generator_seed, num_steps):
        model.eval()
        with torch.no_grad():
            x = generate_samples(model, n, num_steps=num_steps,
                                 generator=torch.Generator().manual_seed(generator_seed))
        model.train()
        return to_common(x.double().numpy())

    for epoch in range(1, epochs + 1):
        net.V.requires_grad_(epoch > min(100, epochs - 1))
        total = 0.0
        for ix in torch.randperm(len(xt), generator=rng).split(BATCH):
            t = dfm_training_times(len(ix), cfg, rng)
            z = torch.randn(len(ix), d, generator=rng)
            opt.zero_grad()
            loss = dfm_training_loss(model, xt[ix], t, z, cfg)
            if not torch.isfinite(loss):
                raise ValueError(f'nonfinite loss at epoch {epoch}')
            loss.backward()
            opt.step()
            net.project_orthogonal()
            total += loss.item() * len(ix)
        if arm == 'stdin' and epoch % 100 == 0:
            net.refresh_input_statistics(xt, stats_rng)
        with torch.no_grad():
            model.eval()
            vl = dfm_training_loss(model, xv, vt, vz, cfg).item()
            model.train()
        row = dict(epoch=epoch, train=total / len(xt), validation=vl)
        if vl < best['dsm'][0]:
            best['dsm'] = (vl, deepcopy(model.state_dict()), epoch)
        if epoch % select_every == 0 or epoch == epochs:
            score, _ = factor_selection_score(sample(1000, 7, 200), reference)
            row['i1'] = score
            if score < best['i1'][0]:
                best['i1'] = (score, deepcopy(model.state_dict()), epoch)
        history.append(row)
    result['train_seconds'] = time.perf_counter() - train_started
    result['history'] = history
    batch_seed = [1234]

    def proposal():
        batch_seed[0] += 1
        return sample(2048, batch_seed[0], steps)

    for name in ('dsm', 'i1'):
        criterion, state, epoch = best[name]
        model.load_state_dict(state)
        batch_seed[0] = 1234
        acc, total, hits = rejection(proposal, int(60 * n_cond / 0.05))
        result[name] = dict(epoch=epoch, criterion=float(criterion),
                            unconditional=evaluate_unconditional(sample(n_eval, 99, steps), pop, truth),
                            conditional=evaluate_conditional(acc, total, hits, pop, threshold, cond_truth, cond_truth2))
        if spec.get('save_dir'):
            folder = Path(spec['save_dir'])
            folder.mkdir(parents=True, exist_ok=True)
            torch.save(state, folder / f'dfm_{name}.pt')
    result['seconds'] = time.perf_counter() - started
    return result


def run_id(spec):
    return f"{spec['population']['name']}_N{spec['N']}_{spec['arm']}_seed{spec['seed']}"


# ----------------------------------------------------------------------------- Modal
try:
    import modal
except ImportError:  # summarize/local smoke runs do not need Modal
    modal = None

if modal is not None:
    IGNORE = ['artifacts', 'data', 'report', '.git', '**/__pycache__', '**/.ipynb_checkpoints', '.venv', '*.ipynb']
    app = modal.App('thesis-dfm-upgrades')
    volume = modal.Volume.from_name('thesis-runs', create_if_missing=True)
    image = (modal.Image.debian_slim(python_version='3.12')
             .pip_install_from_requirements(str(REPO / 'requirements-synthetic.txt'),
                                            extra_index_url='https://download.pytorch.org/whl/cpu')
             .env({'MPLBACKEND': 'Agg', 'PYTHONDONTWRITEBYTECODE': '1'})
             .add_local_dir(str(REPO), '/root/project', ignore=IGNORE))

    @app.function(image=image, volumes={'/vol': volume}, cpu=4, memory=8192, timeout=4 * 3600)
    def run_remote(spec):
        import os
        import traceback
        sys.path.insert(0, '/root/project')
        os.chdir('/root/project')
        folder = Path('/vol/dfm_upgrades') / spec['campaign'] / run_id(spec)
        spec = dict(spec, save_dir=str(folder))
        try:
            result = run_one(spec)
        except Exception:  # noqa: BLE001 - reported to the launcher
            result = dict(spec={k: v for k, v in spec.items() if k != 'population'}, error=traceback.format_exc())
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'result.json').write_text(json.dumps(result))
        volume.commit()
        return result

    def _launch(campaign, specs):
        out = REPO / 'artifacts/dfm_upgrades' / campaign
        out.mkdir(parents=True, exist_ok=True)
        calls = [(run_id(s), run_remote.spawn(s)) for s in specs]
        results = []
        for name, call in calls:
            result = call.get()
            results.append(result)
            status = 'ERROR ' + result['error'][-1500:] if 'error' in result else f"{result['seconds'] / 60:.1f} min"
            print(f'{name}: {status}', flush=True)
            (out / 'results.json').write_text(json.dumps(results))
        print('Saved', out / 'results.json')

    def _specs(campaign, arms, seeds, populations, sizes, epochs):
        pops = []
        if 'P1' in populations:
            pops.append(dict(name='P1', d=500, k=16))
        if 'P2' in populations:
            pops.append(panel_population_spec())
        if 'P3' in populations:
            pops.append(dict(panel_population_spec(), name='P3', nu=5.0))
        return [dict(campaign=campaign, population=p, N=N, arm=a, seed=s, epochs=epochs)
                for s in seeds for p in pops for N in sizes for a in arms]

    @app.local_entrypoint()
    def screen(campaign: str = 'screen1', epochs: int = 1000, arms: str = ','.join(ARMS),
               populations: str = 'P1,P2', sizes: str = '2500,12000', seeds: str = '42'):
        specs = _specs(campaign, arms.split(','), [int(s) for s in seeds.split(',')], populations.split(','),
                       [int(n) for n in sizes.split(',')], epochs)
        p2 = next((s['population'] for s in specs if s['population']['name'] in ('P2', 'P3')), None)
        if p2:
            out = REPO / 'artifacts/dfm_upgrades' / campaign
            out.mkdir(parents=True, exist_ok=True)
            (out / 'p2_population_info.json').write_text(json.dumps(p2['info']))
            print('P2:', json.dumps({k: p2['info'][k] for k in ('c_bar', 'lambda1_over_c')}),
                  'eigenvalues', np.round(p2['info']['eigenvalues'][:8], 2), flush=True)
        print(f'{len(specs)} runs', flush=True)
        _launch(campaign, specs)

    @app.local_entrypoint()
    def confirm(campaign: str = 'confirm1', arms: str = 'base,gaussres', seeds: str = '314,2718',
                epochs: int = 1000, populations: str = 'P1,P2', sizes: str = '2500,12000'):
        specs = _specs(campaign, arms.split(','), [int(s) for s in seeds.split(',')], populations.split(','),
                       [int(n) for n in sizes.split(',')], epochs)
        print(f'{len(specs)} runs', flush=True)
        _launch(campaign, specs)


# ----------------------------------------------------------------------------- summary
def summarize(paths):
    rows = []
    for path in paths:
        for r in json.loads(Path(path).read_text()):
            if 'error' in r:
                print('ERROR in', r['spec']); continue
            key = (r['population'], r['spec']['N'], r['spec']['arm'], r['spec']['seed'])
            if r['spec']['arm'] == 'gauss_ref':
                variants = [('direct', r['direct'], r['conditional_direct']), ('sampled', r['sampled'], None)]
            else:
                variants = [(f"{c}@{r[c]['epoch']}", r[c]['unconditional'], r[c]['conditional']) for c in ('dsm', 'i1')]
            for label, u, cnd in variants:
                rows.append((key, label, u, cnd, r['reference'], r.get('train_seconds', r['seconds'])))
    print('pop  N      arm            seed  ckpt      fac[min-max] mal  resid  cov   swd   eqKS  eq5%  eq1%  kurt(true) '
          '| cond acc  cSW (ref)    cKS (ref)  | train s')
    for (pop, N, arm, seed), label, u, cnd, ref, secs in sorted(rows, key=lambda r: r[0]):
        c = (f"{cnd['acceptance']:.3f} {cnd['sliced_w1_vs_truth']:.4f} ({cnd['sliced_w1_reference']:.4f}) "
             f"{cnd['portfolio_ks_exact']:.3f} ({cnd['portfolio_ks_reference']:.3f})") if cnd else '-'
        e = u['portfolios']['equal']
        print(f"{pop:<4} {N:<6} {arm:<14} {seed:<5} {label:<9} {u['factor_ratio_min']:.2f}-{u['factor_ratio_max']:.2f} "
              f"{u['factor_mean_abs_log']:.2f} {u['residual_ratio']:.2f}  {u['covariance_error']:.3f} "
              f"{u['sliced_w1']:.4f} {e['ks']:.3f} {e['tail5']:.3f} {e['tail1']:.3f} {e.get('kurtosis', float('nan')):5.2f}({e.get('true_kurtosis', 0):.2f}) "
              f"| {c} | {secs:.0f}")
    seen = set()
    for (pop, N, *_), _, _, _, ref, _ in rows:
        if (pop, N) not in seen:
            seen.add((pop, N))
            e = ref['portfolios']['equal']
            print(f"reference truth-vs-truth {pop} N={N}: cov {ref['covariance_error']:.3f} swd {ref['sliced_w1']:.4f} "
                  f"fac {ref['factor_ratio_min']:.2f}-{ref['factor_ratio_max']:.2f} eqKS {e['ks']:.3f}")


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == 'summarize':
        summarize(sys.argv[2:])
