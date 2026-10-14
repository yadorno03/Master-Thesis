"""Matched-count CPU generation timings and independent conditional accuracy."""

from functools import partial
import math
from time import perf_counter

import torch
from threadpoolctl import threadpool_limits

from guided_evaluation import conditional_diagnostics
from guided_sampling import generate_conditional_samples, sample_reference_conditional
from h_model import _positive_int


def benchmark_guidance(h_model, *, num_samples=512, num_steps=128, repeats=5,
                       validation_samples=2048, batch_size=128, max_proposals=100000,
                       initialization='h_rejection', scale=1., seed=74000,
                       setup_seconds=None):
    """Warm up, alternate method order, and score every draw against a third sample.

    Generation includes initialization and rejected proposals. H setup/training
    is caller-measured and separate; None means unknown, never zero. CPU only,
    with one PyTorch/BLAS thread. Returns a JSON report and all sample arrays.
    """
    counts = dict(num_samples=num_samples, num_steps=num_steps, repeats=repeats,
                  validation_samples=validation_samples, batch_size=batch_size,
                  max_proposals=max_proposals)
    for name, value in counts.items():
        _positive_int(value, name)
    if min(num_samples, validation_samples) < 2:
        raise ValueError('Accuracy comparison needs at least two draws')
    if max_proposals < max(num_samples, validation_samples):
        raise ValueError('Proposal budget must cover both requested sample counts')
    if setup_seconds is not None and (not math.isfinite(setup_seconds) or setup_seconds < 0):
        raise ValueError('setup_seconds must be finite/nonnegative or None')
    parameter = next(h_model.reference.parameters())
    if parameter.device.type != 'cpu':
        raise ValueError('This benchmark controls CPU timing only; load H on CPU')
    settings = dict(num_steps=num_steps, batch_size=batch_size, max_proposals=max_proposals)
    samplers = {
        'guided': partial(generate_conditional_samples, initialization=initialization, scale=scale),
        'rejection': sample_reference_conditional,
    }

    def draw(method, count, draw_seed):
        generator = torch.Generator().manual_seed(draw_seed)
        return samplers[method](h_model, count, generator=generator, **settings)

    rows, samples = [], {}
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            for offset, method in enumerate(samplers):
                draw(method, min(16, num_samples), seed + offset)
            validation = draw('rejection', validation_samples, seed + 2)
            samples['validation_model'] = validation.samples.numpy()
            for repeat in range(repeats):
                order = ('guided', 'rejection') if repeat % 2 == 0 else ('rejection', 'guided')
                for position, method in enumerate(order):
                    draw_seed = seed + 10 + 2 * repeat + (method == 'rejection')
                    started = perf_counter()
                    result = draw(method, num_samples, draw_seed)
                    seconds = perf_counter() - started
                    values = result.samples.numpy()
                    samples[f'{method}_{repeat}_model'] = values
                    quality = conditional_diagnostics(
                        values, samples['validation_model'], h_model.w.numpy(), h_model.C,
                        seed=seed + 3)
                    rows.append({
                        'repeat': repeat, 'position': position, 'method': method,
                        'seed': draw_seed, 'seconds': seconds,
                        'milliseconds_per_draw': 1000 * seconds / num_samples,
                        'generation_diagnostics': result.diagnostics, **quality,
                    })
    finally:
        torch.set_num_threads(previous_threads)
    report = {
        'config': {**counts, 'initialization': initialization, 'scale': scale, 'seed': seed},
        'dtype': str(parameter.dtype), 'device': 'cpu', 'torch_threads': 1, 'blas_thread_limit': 1,
        'setup_seconds': setup_seconds, 'rows': rows,
        'validation_reference_diagnostics': validation.diagnostics,
        'timing_scope': 'generation includes initialization and rejected proposals; '
                        'excludes H setup, loading, warm-up, validation, metrics and file I/O',
    }
    return report, samples
