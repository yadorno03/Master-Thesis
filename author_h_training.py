"""Terminal-label regression on the exact frozen DDPM transition grid."""
from dataclasses import dataclass
import math
import torch
from author_sampling import sample_author_ddpm
from h_model import terminal_label_loss


@dataclass
class DDPMLabelData:
    states: torch.Tensor       # [N,J,d], terminal endpoint excluded
    labels: torch.Tensor       # [N,1]
    reference: object          # identity binds data to the exact in-memory model
    event: object


@torch.no_grad()
def collect_ddpm_labels(h, count, *, batch_size=32, generator=None):
    if any(type(n) is not int or n < 1 for n in (count, batch_size)):
        raise ValueError('count and batch_size must be positive integers')
    states, labels = [], []
    for start in range(0, count, batch_size):
        paths = sample_author_ddpm(h.reference, min(batch_size, count-start),
                                    generator=generator, return_trajectory=True)
        states.append(paths[:, :-1].clone())
        labels.append(h.terminal_event(paths[:, -1])[:, None].to(paths.dtype))
    return DDPMLabelData(torch.cat(states), torch.cat(labels), h.reference, h.event)


def _validate(h, data):
    if data.reference is not h.reference or data.event != h.event:
        raise ValueError('Labels must come from this exact reference and event')
    x, y = data.states, data.labels
    if (x.ndim != 3 or len(x) == 0 or x.shape[1:] != (len(h.reference.indexes), h.reference.d)
            or y.shape != (len(x), 1) or y.dtype != x.dtype or y.device != x.device
            or not ((y == 0) | (y == 1)).all() or x.requires_grad or y.requires_grad):
        raise ValueError('Invalid detached whole-trajectory terminal-label data')
    h.reference.check_state(x.flatten(0, 1))
    if any(p.requires_grad or p.grad is not None for p in h.reference.parameters()):
        raise ValueError('Reference must remain frozen')


def _loss(h, states, labels):
    n, steps, d = states.shape
    indexes = torch.tensor(h.reference.indexes, device=states.device).repeat(n)
    predictions = h(indexes, states.reshape(-1, d)).reshape(n, steps, 1)
    weights = states.new_full((steps,), 1 / steps)
    return terminal_label_loss(predictions, labels, weights)


def train_ddpm_h(h, data, *, epochs=100, batch_size=32, lr=1e-3, generator=None):
    """Uniform transition-weighted Brier loss; split data by whole trajectory."""
    _validate(h, data)
    if any(type(n) is not int or n < 1 for n in (epochs, batch_size)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError('Require positive epochs/batch_size and finite positive lr')
    optimizer = torch.optim.Adam(h.network.parameters(), lr=lr)
    modes = [(m, m.training) for m in h.modules()]
    losses = []
    try:
        h.train()
        for _ in range(epochs):
            order = torch.randperm(len(data.states), device=data.states.device, generator=generator)
            total = 0.
            for subset in order.split(batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = _loss(h, data.states[subset], data.labels[subset])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite H loss')
                loss.backward()
                optimizer.step()
                total += loss.item() * len(subset)
            losses.append(total / len(data.states))
        optimizer.zero_grad(set_to_none=True)
    finally:
        for module, training in modes:
            module.training = training
    return losses


@torch.no_grad()
def evaluate_ddpm_h(h, data):
    _validate(h, data)
    return {'brier_loss': float(_loss(h, data.states, data.labels)),
            'positive_labels': int(data.labels.sum()), 'trajectories': len(data.states)}
