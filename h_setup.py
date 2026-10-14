"""Fit H to a fixed raw-return event on a saved, frozen equity DFM."""

import hashlib
import math
from pathlib import Path
from time import perf_counter

import torch

from conditioning import portfolio_event_in_model_coordinates
from h_model import HNetwork, _positive_int, evaluate_h, train_h
from reference_checkpoint import load_reference_checkpoint


def train_h_from_checkpoint(path, *, weights, threshold, num_steps=128,
                            train_trajectories=4096, held_out_trajectories=1024,
                            epochs=100, batch_size=64, rollout_batch_size=32,
                            hidden_dims=(64, 64), lr=1e-3, seed=20261010):
    """Return (trained H, bundle metadata); weights map every asset to a raw weight.

    Reuse saved training scaling and horizon. Independent CPU streams supply
    initialization, training paths, shuffling and held-out paths. Setup costs
    exclude validation, checkpoint I/O and reference pretraining.
    """
    settings = dict(num_steps=num_steps, train_trajectories=train_trajectories,
                    held_out_trajectories=held_out_trajectories, epochs=epochs,
                    batch_size=batch_size, rollout_batch_size=rollout_batch_size)
    for name, value in settings.items():
        _positive_int(value, name)
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('lr must be finite and positive')
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    ref, assets, mean, std = load_reference_checkpoint(path)
    if not isinstance(weights, dict) or set(weights) != set(assets):
        raise ValueError('Weights must map exactly the checkpoint asset identifiers')
    raw_weights = [weights[asset] for asset in assets]
    event = portfolio_event_in_model_coordinates(raw_weights, threshold, mean, std)
    seeds = {name: seed + offset for offset, name in enumerate(
        ('h_init', 'training_paths', 'shuffle', 'validation_paths'))}

    def generator(name):
        return torch.Generator().manual_seed(seeds[name])

    started = perf_counter()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seeds['h_init'])
        h = HNetwork(ref, event, hidden_dims=hidden_dims)
    initialization_seconds = perf_counter() - started
    started = perf_counter()
    training = h.sample_trajectories(train_trajectories, num_steps,
        batch_size=rollout_batch_size, generator=generator('training_paths'))
    label_seconds = perf_counter() - started
    baseline = float(training.labels.double().mean())
    if baseline in (0., 1.):
        raise ValueError('H training needs both event classes; increase trajectories or change event')
    started = perf_counter()
    history = train_h(h, training, epochs=epochs, batch_size=batch_size, lr=lr,
                      generator=generator('shuffle'))
    optimization_seconds = perf_counter() - started
    del training
    h.eval()
    held_out = h.sample_trajectories(held_out_trajectories, num_steps,
        batch_size=rollout_batch_size, generator=generator('validation_paths'))
    validation = evaluate_h(h, held_out, constant_probability=baseline, batch_size=batch_size)
    validation['event_rate'] = float(held_out.labels.double().mean())
    metadata = {
        'assets': assets, 'preprocessing': {'mean': mean.tolist(), 'std': std.tolist()},
        'raw_event': {'weights': [float(w) for w in raw_weights], 'threshold': float(threshold)},
        'source_checkpoint_sha256': digest, 'seeds': seeds,
        'training_config': {**settings, 'hidden_dims': list(h.hidden_dims), 'lr': lr},
        'training_event_rate': baseline, 'training_loss': history, 'h_validation': validation,
        'h_setup_cost': {
            'initialization_seconds': initialization_seconds,
            'label_generation_seconds': label_seconds,
            'optimization_seconds': optimization_seconds,
            'total_seconds': initialization_seconds + label_seconds + optimization_seconds,
            'torch_threads': torch.get_num_threads(),
            'scope': 'H initialization + training labels + optimization; excludes '
                     'reference pretraining, loading, validation and file I/O'},
    }
    return h, metadata
