"""Guided conditional benchmark on the I3 DFM at d = 500, k = 8 (population P2).

The frozen DFM is the I3 checkpoint of the d = 500 campaign (P2, N = 2,500, seed 42,
DSM-selected). Both H models are trained on the same unguided trajectories with the
notebook recipe (40k trajectories, 50 epochs; Guo et al. (2026) for H and its Doob
guidance). Sources compared for the event w'R <= C:

* conditional truth: exact truncated Gaussian law of P2 (two independent draws);
* DFM rejection: the frozen I3 DFM conditioned by rejection, the target of guidance;
* Refined CDG-ML + exact final step;
* CDG-ML + exact final step (H without refinements);
* No guidance + exact final step: unguided DFM paths, only the last step is
  drawn from the exact conditioned transition (no H at all).

CDG-ML with a plain Euler last step is not sampled or reported.

  modal run --detach scripts/i3_conditional.py::launch --tails 0.05,0.01
  modal volume get thesis-runs /i3_conditional/<run> artifacts/i3_conditional/
  python scripts/i3_conditional.py report artifacts/i3_conditional/<run>
  python scripts/i3_conditional.py smoke          # tiny local end-to-end check
"""
from pathlib import Path
import json
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
T, T0, STEPS, WIDTH = 8.0, 0.005, 400, 64
CHECKPOINT = 'dfm_upgrades/screen1/P2_N2500_gaussres_seed42/dfm_dsm.pt'
MODELS = ('refined_exact', 'cdg_ml_exact', 'unguided_exact')
SOURCES = ('truth', 'rejection', *MODELS)
LABELS = {
    'truth': 'Conditional truth',
    'rejection': 'DFM rejection (target)',
    'refined_exact': 'Refined CDG-ML + exact step',
    'cdg_ml_exact': 'CDG-ML + exact step',
    'unguided_exact': 'No guidance + exact step',
}
SEEDS = dict(initialization=2027, guided=42001, truth=42002, rejection=42003, rejection_reference=42004,
             truth_reference=42005, projection=42006, matching=42007, unguided=42008)


def p2_population(spec):
    from synthetic_conditional_truth import FactorPopulation
    B, S = np.array(spec['B']), np.array(spec['S'])
    return FactorPopulation(B, np.array(spec['D']), np.ones(1), np.zeros((1, B.shape[1])), S[None],
                            dict(population_seed=-1, design='panel-calibrated-gaussian'))


def load_i3_reference(path, d, k):
    import torch
    from DFM import UnconditionalDFM
    from dfm_variants import GaussianResidualScoreNetwork
    from reference import FrozenDFMReference
    model = UnconditionalDFM(d, k, sigma_max=2.0, T=T, t0=T0, hidden_dims=[WIDTH, WIDTH])
    model.factor_score_network = GaussianResidualScoreNetwork(d, k, 2.0, (WIDTH, WIDTH))
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return FrozenDFMReference(model)


def summary_table(samples, comparisons, moments, threshold, weights):
    """One column per source; distances of truth/rejection to themselves are Monte Carlo levels."""
    import pandas as pd
    from synthetic_benchmark.reporting import portfolio_values
    comparisons = comparisons.set_index(['left', 'right'])
    moments = moments.set_index('source')
    model_n = max(len(samples[m]) for m in MODELS)

    def lookup(left, right, n=None):
        rows = comparisons.loc[[(left, right)]]
        if n is not None:
            rows = rows[rows.n_used == min(n, rows.n_used.max())]
        return rows.iloc[0]

    pairs = {('truth', 'rejection'): ('rejection', 'truth'), ('truth', 'truth'): ('truth', 'truth_reference', model_n),
             ('rejection', 'rejection'): ('rejection', 'rejection_reference', len(samples['rejection'])),
             ('rejection', 'truth'): ('rejection', 'truth')}
    table = {}
    for source in SOURCES:
        p = portfolio_values(samples[source], weights)
        column = {'Rows': len(p), 'Event rate': float(np.mean(p <= threshold))}
        for reference, label in (('rejection', 'DFM rejection'), ('truth', 'truth')):
            values = lookup(*pairs.get((source, reference), (source, reference)))
            for metric, name in (('portfolio_ks', 'Portfolio KS'), ('portfolio_w1', 'Portfolio W1'),
                                 ('sliced_w1', 'Sliced W1')):
                column[f'{name} vs {label}'] = float(values[metric])
            column[f'Rows compared vs {label}'] = int(values.n_used)
        column['Analytic portfolio KS'] = float(moments.loc[source, 'analytic_portfolio_ks'])
        column['Mean error'] = float(moments.loc[source, 'mean_error'])
        column['Covariance error'] = float(moments.loc[source, 'covariance_error'])
        column['Trace ratio'] = float(moments.loc[source, 'diversity_trace_ratio'])
        table[LABELS[source]] = column
    return pd.DataFrame(table)


def run_tail(spec, output, *, tail, checkpoint, n_train=40_000, n_validation=2_000, n_test=2_000, epochs=50,
             n_samples=3000, rejection_n=1000, steps=STEPS, validation_interval_steps=157):
    """Train both H models on the frozen I3 DFM, sample every source, and save samples and metrics."""
    import torch
    from guided_sampling import DFMHGuidance, exact_final_step
    from sampling import generate_samples
    from synthetic_benchmark.conditional import (
        CDG_ML_OPTIONS, DEFAULT_TRAIN_SEEDS, REFINED_OPTIONS, fit_and_evaluate_h, h_diagnostic_table, initialize_h,
        sample_dfm_rejection, sample_h_guided, sample_h_training_data,
    )
    from synthetic_benchmark.reporting import conditional_comparisons, conditional_moment_table
    from synthetic_conditional_truth import check_population_model
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    population = p2_population(spec)
    d, k = population.d, population.k
    reference = load_i3_reference(checkpoint, d, k)
    check_population_model(population, reference)
    variants = {'refined': ('Refined CDG-ML', REFINED_OPTIONS, True),
                'cdg_ml': ('CDG-ML', CDG_ML_OPTIONS, False)}
    models = {key: initialize_h(reference, population, tail, (64, 64), SEEDS['initialization'],
                                log_remaining_time=log_rt)[1] for key, (_, _, log_rt) in variants.items()}
    threshold = float(models['refined'].C)
    print(f'd={d} k={k} tail={tail} threshold={threshold:.6f}', flush=True)
    timings = {}
    t = time.perf_counter()
    data = sample_h_training_data(models['refined'], (('train', n_train), ('validation', n_validation),
                                                      ('test', n_test)),
                                  steps=steps, seeds=DEFAULT_TRAIN_SEEDS, path_batch_size=256, device='cpu')
    timings['trajectory_collection_seconds'] = time.perf_counter() - t
    frozen_before = {name: value.detach().clone() for name, value in reference.state_dict().items()}
    h_results = {}
    for key, (label, options, _) in variants.items():
        t = time.perf_counter()
        selection, history, probability, test_metrics = fit_and_evaluate_h(
            models[key], reference, data, frozen_before, epochs=epochs, batch_size=64, lr=1e-3,
            seeds=DEFAULT_TRAIN_SEEDS, device='cpu',
            training_options=dict(options, validation_interval_steps=validation_interval_steps))
        timings[f'{key}_training_seconds'] = time.perf_counter() - t
        h_results[key] = dict(label=label, selection=selection, test_metrics=test_metrics, history=history,
                              training_probability=probability)
        torch.save(dict(state=models[key].network.state_dict(),
                        normalization={n: getattr(models[key], n).detach().clone() for n in
                                       ('normalization_times', 'normalization_mean', 'normalization_scale')}),
                   output / f'h_{key}.pt')
    del data
    h_table = h_diagnostic_table(h_results)

    samples, diagnostics = {}, {}
    proposals = max(int(np.ceil(20 * n_samples / tail)), 256)
    for key, source in (('refined', 'refined_exact'), ('cdg_ml', 'cdg_ml_exact')):
        t = time.perf_counter()
        diagnostics[source] = {}
        samples[source] = sample_h_guided(models[key], DFMHGuidance(models[key]), n_samples=n_samples,
                                          batch_size=256, steps=steps, seed=SEEDS['guided'],
                                          initialization_max_proposals=proposals,
                                          diagnostics=diagnostics[source], final_step='exact')
        timings[f'{source}_sampling_seconds'] = time.perf_counter() - t
    t = time.perf_counter()
    with torch.no_grad():
        generator = torch.Generator().manual_seed(SEEDS['unguided'])
        samples['unguided_exact'] = np.concatenate([
            generate_samples(reference, len(chunk), num_steps=steps, generator=generator,
                             final_step=lambda x, g: exact_final_step(models['refined'], x, steps, g)).numpy()
            for chunk in np.array_split(np.arange(n_samples), max(1, n_samples // 256))])
    timings['unguided_exact_sampling_seconds'] = time.perf_counter() - t
    samples['truth'] = population.sample(n_samples, np.random.default_rng(SEEDS['truth']), c=threshold)
    samples['truth_reference'] = population.sample(n_samples, np.random.default_rng(SEEDS['truth_reference']),
                                                   c=threshold)
    cap = int(4 * rejection_n / tail)
    for name in ('rejection', 'rejection_reference'):
        samples[name], diagnostics[name] = sample_dfm_rejection(
            reference, population.w, threshold, rejection_n, steps=steps, batch_size=2048, seed=SEEDS[name],
            max_proposals=cap)
        info = diagnostics[name]
        print(f"{name}: {info['status']} {info['retained_count']}/{info['requested_count']} from "
              f"{info['total_proposals']} proposals in {info['sampling_seconds']:.0f}s", flush=True)
        timings[f'{name}_seconds'] = info['sampling_seconds']

    comparisons, directions = conditional_comparisons(samples, population.w, projection_seed=SEEDS['projection'],
                                                      matching_seed=SEEDS['matching'], projections=64, models=MODELS)
    moments = conditional_moment_table(samples, population, threshold)
    summary = summary_table(samples, comparisons, moments, threshold, population.w)
    np.savez_compressed(output / 'conditional_samples.npz', **samples, weights=population.w, threshold=threshold,
                        projection_directions=directions, B=population.B)
    comparisons.to_csv(output / 'comparisons.csv', index=False)
    moments.to_csv(output / 'moments.csv', index=False)
    summary.to_csv(output / 'summary.csv')
    h_table.to_csv(output / 'h_diagnostics.csv')
    timings['total_seconds'] = time.perf_counter() - started

    def clean(value):
        if isinstance(value, dict):
            return {str(a): clean(b) for a, b in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    run = dict(tail=tail, threshold=threshold, d=d, k=k, steps=steps, checkpoint=str(checkpoint),
               population='P2', h_recipe=dict(n_train=n_train, n_validation=n_validation, n_test=n_test,
                                              epochs=epochs, widths=[64, 64], batch_size=64, lr=1e-3),
               n_samples=n_samples, rejection_n=rejection_n, seeds=SEEDS, timings=timings,
               sampling_diagnostics=diagnostics,
               h_selection={key: dict(best_epoch=r['selection'].get('best_epoch'),
                                      best_step=r['selection'].get('best_step'),
                                      validation=r['selection'].get('history', r['selection'].get('validation')))
                            for key, r in h_results.items()})
    (output / 'run.json').write_text(json.dumps(clean(run), indent=1, default=str))
    print(summary.to_string(), flush=True)
    return dict(tail=tail, timings=timings, summary=json.loads(summary.to_json()))


# ----------------------------------------------------------------------------- Modal
try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    IGNORE = ['artifacts', 'data', 'report', '.git', '**/__pycache__', '**/.ipynb_checkpoints', '.venv', '*.ipynb']
    app = modal.App('thesis-i3-conditional')
    volume = modal.Volume.from_name('thesis-runs', create_if_missing=True)
    image = (modal.Image.debian_slim(python_version='3.12')
             .pip_install_from_requirements(str(REPO / 'requirements-synthetic.txt'),
                                            extra_index_url='https://download.pytorch.org/whl/cpu')
             .env({'MPLBACKEND': 'Agg', 'PYTHONDONTWRITEBYTECODE': '1'})
             .add_local_dir(str(REPO), '/root/project', ignore=IGNORE))

    @app.function(image=image, volumes={'/vol': volume}, cpu=8, memory=16384, timeout=8 * 3600)
    def run_remote(spec, run, tail):
        import os
        import traceback
        import torch
        sys.path.insert(0, '/root/project')
        os.chdir('/root/project')
        torch.set_num_threads(8)
        output = Path('/vol/i3_conditional') / run / f'tail{tail}'
        try:
            return run_tail(spec, output, tail=tail, checkpoint=Path('/vol') / CHECKPOINT)
        except Exception:  # noqa: BLE001 - reported to the launcher
            output.mkdir(parents=True, exist_ok=True)
            (output / 'error.txt').write_text(traceback.format_exc())
            return dict(tail=tail, error=traceback.format_exc())
        finally:
            volume.commit()

    @app.local_entrypoint()
    def launch(tails: str = '0.05,0.01', run: str = ''):
        from datetime import datetime, timezone
        sys.path.insert(0, str(REPO / 'scripts'))
        from dfm_upgrades import panel_population_spec
        run = run or 'p2_i3_d500_k8_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        print('Run:', run, flush=True)
        spec = panel_population_spec()
        calls = [run_remote.spawn(spec, run, float(t)) for t in tails.split(',')]
        out = REPO / 'artifacts/i3_conditional' / run
        out.mkdir(parents=True, exist_ok=True)
        results = []
        for call in calls:
            result = call.get()
            results.append(result)
            print(('ERROR ' + result['error'][-2000:]) if 'error' in result else
                  f"tail {result['tail']}: {result['timings']['total_seconds'] / 60:.1f} min", flush=True)
            (out / 'launcher_results.json').write_text(json.dumps(results, indent=1))
        print('Volume folder: /i3_conditional/' + run)


# ----------------------------------------------------------------------------- local commands
def smoke():
    """Tiny end-to-end run on the downloaded checkpoint to check the plumbing."""
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / 'scripts'))
    from dfm_upgrades import panel_population_spec
    checkpoint = REPO / 'artifacts/dfm_upgrades/checkpoints/P2_N2500_gaussres_seed42.pt'
    run_tail(panel_population_spec(), REPO / 'artifacts/i3_conditional/smoke/tail0.05', tail=0.05,
             checkpoint=checkpoint, n_train=400, n_validation=100, n_test=100, epochs=2, n_samples=64,
             rejection_n=16, steps=60, validation_interval_steps=5)


if __name__ == '__main__':
    sys.path.insert(0, str(REPO))
    command = sys.argv[1] if len(sys.argv) > 1 else ''
    if command == 'smoke':
        smoke()
    elif command == 'report':
        from plot_i3_conditional import report
        report(Path(sys.argv[2]))
    else:
        print(__doc__)
