"""Run the notebook's benchmark on Modal CPU containers at a chosen dimension.

One container trains and evaluates the unconditional DFM exactly as the
notebook's first part does; then one container per event probability runs
``run_conditional_benchmark`` on that frozen DFM. Everything is written to the
Modal volume ``thesis-runs`` under /benchmark/<run>/, so results survive a
disconnected client.

  modal run --detach scripts/modal_benchmark.py::high_dimension --d 2048 --k 16 --tails 0.05,0.01
  modal volume get thesis-runs /benchmark/<run> artifacts/benchmark/
"""
from pathlib import Path
import json

import modal

REPO = Path(__file__).resolve().parents[1]
ROOT = '/vol/benchmark'
IGNORE = ['artifacts', 'data', 'report', '.git', '**/__pycache__', '**/.ipynb_checkpoints', '.venv', '*.ipynb']
THREADS = 8

app = modal.App('thesis-benchmark')
volume = modal.Volume.from_name('thesis-runs', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install_from_requirements(str(REPO / 'requirements-synthetic.txt'),
                                        extra_index_url='https://download.pytorch.org/whl/cpu')
         .env({'MPLBACKEND': 'Agg', 'PYTHONDONTWRITEBYTECODE': '1'})
         .add_local_dir(str(REPO), '/root/project', ignore=IGNORE))


def _config(d, k):
    """The notebook's standard configuration and CONFIG_OVERRIDES, at dimension d x k."""
    import os
    import sys
    from dataclasses import replace
    import torch
    sys.path.insert(0, '/root/project')
    os.chdir('/root/project')
    torch.set_num_threads(THREADS)
    from synthetic_benchmark.configuration import Config
    return replace(Config.preset('standard'), dfm_initialization='pca', dfm_time_sampling='log_balanced',
                   dfm_loading_lr_scale=0.1, dfm_loading_warmup_epochs=100, lr=1e-3, d=int(d), k=int(k))


@app.function(image=image, volumes={'/vol': volume}, cpu=THREADS, memory=32768, timeout=8 * 3600)
def unconditional(run: str, d: int, k: int) -> dict:
    """Train, sample and evaluate the DFM as in the notebook's first part."""
    import time
    config = _config(d, k)
    from reference import FrozenDFMReference
    from sampling import generate_samples
    from synthetic_benchmark.unconditional import (
        evaluate_unconditional_samples, fit_and_save_dfm, initialize_unconditional_run, prepare_unconditional_case,
        save_unconditional_case, save_unconditional_summary,
    )
    output = Path(ROOT) / run
    try:
        initialize_unconditional_run(output, config)
        started = time.perf_counter()
        case, seed = 'gaussian', config.seeds[0]
        folder, streams, pop = prepare_unconditional_case(output, config, case, seed)
        train = pop.sample(config.n_train, streams.np('train'))
        val = pop.sample(config.n_val, streams.np('validation'))
        truth = pop.sample(config.n_eval, streams.np('truth'))
        truth2 = pop.sample(config.n_eval, streams.np('truth2'))
        model, history, selected, fit_seconds = fit_and_save_dfm(train, val, config, streams, pop, case, folder)
        reference = FrozenDFMReference(model)
        generation_started = time.perf_counter()
        generated = generate_samples(reference, config.n_eval, num_steps=config.steps,
                                     generator=streams.torch('generated', config.device)).cpu().numpy()
        generation_seconds = time.perf_counter() - generation_started
        result, baseline = evaluate_unconditional_samples(generated, truth, truth2, pop, reference, config, case,
                                                          seed, fit_seconds, generation_seconds)
        save_unconditional_case(folder, case, generated, truth, truth2, pop, result, baseline)
        metrics, _ = save_unconditional_summary([result, baseline], output, started)
        return dict(run=run, selected_epoch=int(selected), fit_seconds=fit_seconds,
                    wall_seconds=time.perf_counter() - started,
                    metrics=json.loads(metrics.to_json(orient='records')))
    finally:
        volume.commit()


@app.function(image=image, volumes={'/vol': volume}, cpu=THREADS, memory=32768, timeout=8 * 3600)
def conditional(run: str, d: int, k: int, tail: float, rejection_n: int, rejection_cap: int) -> dict:
    """The notebook's conditional benchmark on the frozen DFM of ``run``."""
    import time
    import traceback
    volume.reload()
    config = _config(d, k)
    from synthetic_benchmark.conditional import run_conditional_benchmark
    from synthetic_benchmark.configuration import ConditionalAssessmentConfig
    started = time.perf_counter()
    try:
        result = run_conditional_benchmark(
            config, Path(ROOT) / run, tail=float(tail), n_train=40_000, n_validation=2_000, n_test=2_000,
            epochs=50, validation_interval_steps=157, n_samples=config.n_eval, show=False,
            assessment_settings=ConditionalAssessmentConfig(
                rejection_n=int(rejection_n), rejection_batch_size=2048, rejection_max_proposals=int(rejection_cap),
                rejection_reference_max_proposals=int(rejection_cap), rejection_seed=42003,
                rejection_reference_seed=42004, truth_reference_seed=42005, projection_seed=42006,
                matching_seed=42007, projections=64))
        return dict(run=run, tail=tail, wall_seconds=time.perf_counter() - started, timings=result['timings'],
                    summary=json.loads(result['summary'].to_json()),
                    h_diagnostics=json.loads(result['h_diagnostics'].to_json()))
    except Exception:  # noqa: BLE001 - reported back to the launcher
        return dict(run=run, tail=tail, error=traceback.format_exc())
    finally:
        volume.commit()


@app.local_entrypoint()
def high_dimension(d: int = 2048, k: int = 16, tails: str = '0.05,0.01', run: str = '', rejection_n: int = 1000,
                   out_file: str = ''):
    """Train one DFM, then run the conditional benchmark for every event probability in parallel."""
    from datetime import datetime, timezone
    run = run or f'd{d}_k{k}_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    print('Run:', run, flush=True)
    dfm = unconditional.remote(run, d, k)
    print('Unconditional DFM:', json.dumps({key: dfm[key] for key in ('selected_epoch', 'fit_seconds',
                                                                       'wall_seconds')}), flush=True)
    _conditional_runs(run, d, k, tails, rejection_n, out_file, [dict(unconditional=dfm)])


@app.local_entrypoint()
def conditional_only(run: str, d: int = 2048, k: int = 16, tails: str = '0.05,0.01', rejection_n: int = 1000,
                     out_file: str = ''):
    """Rerun the conditional benchmark on the DFM already saved under /benchmark/<run>."""
    _conditional_runs(run, d, k, tails, rejection_n, out_file, [])


def _conditional_runs(run, d, k, tails, rejection_n, out_file, results):
    calls = []
    for tail in (float(t) for t in tails.split(',')):
        cap = int(4 * rejection_n / tail)  # About four times the expected number of proposals.
        calls.append(conditional.spawn(run, d, k, tail, rejection_n, cap))
    for call in calls:
        result = call.get()
        results.append(result)
        print(('ERROR ' + result['error'][-2000:]) if 'error' in result else
              f"done tail {result['tail']}: {result['wall_seconds'] / 60:.1f} min", flush=True)
        if out_file:
            Path(out_file).write_text(json.dumps(results, indent=1))
    print('Volume folder: /benchmark/' + run)
