"""Smooth terminal-event probability network and trajectory regression loss."""

from dataclasses import dataclass
import math
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

from conditioning import DFMConditioning
from sampling import generate_samples


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass
class TerminalLabelData:
    """Whole trajectories: features [N,J,k+1], binary labels [N,1].

    times and weights have shape [J]. Only preterminal times are stored;
    weights[j] = delta_t[j] / (T-t0). Split by trajectory, never by time row.
    """

    times: torch.Tensor
    features: torch.Tensor
    labels: torch.Tensor
    weights: torch.Tensor


class HNetwork(nn.Module):
    """H_phi(t,y) with a smooth tanh MLP and sigmoid scalar probability.

    y contains all k+1 coordinates (Z,U). Times are generative and normalized
    internally by T-t0. log_prob uses logsigmoid directly on logits, so it stays
    finite even when the floating-point sigmoid output underflows to zero.
    The attached reference and event specify what probability is being learned.
    """

    def __init__(self, conditioning, hidden_dims=(32, 32)):
        super().__init__()
        if not isinstance(conditioning, DFMConditioning):
            raise TypeError("conditioning must be a DFMConditioning")
        self.conditioning = conditioning
        self.hidden_dims = tuple(hidden_dims)
        if not self.hidden_dims:
            raise ValueError("At least one smooth hidden layer is required")
        layers = []
        dimension = conditioning.reference._score.k + 2  # y plus time
        for width in self.hidden_dims:
            _positive_int(width, "hidden width")
            layers.extend((nn.Linear(dimension, width), nn.Tanh()))
            dimension = width
        layers.append(nn.Linear(dimension, 1))
        parameter = next(conditioning.reference.parameters())
        self.network = nn.Sequential(*layers).to(device=parameter.device, dtype=parameter.dtype)

    @property
    def horizon(self):
        ref = self.conditioning.reference
        return ref.T - ref.t0

    def _logits(self, t, y):
        t = self.conditioning._batch(t, y, self.conditioning.reference._score.k + 1)
        return self.network(torch.cat((y, t / self.horizon), dim=-1))

    def forward(self, t, y):
        return torch.sigmoid(self._logits(t, y))

    def log_prob(self, t, y):
        return F.logsigmoid(self._logits(t, y))

    @torch.no_grad()
    def sample_trajectories(self, num_trajectories, num_steps=100, *, generator=None,
                            batch_size=None):
        """Collect unguided Gaussian-start Euler-Maruyama paths and terminal labels.

        Labels are evaluated at T-t0; that endpoint is excluded from regression
        inputs. This learns the discretized reference event probability.
        Optional batch_size bounds ambient path memory; only reduced features
        are retained across batches. Seeds reproduce a fixed batching setup.
        """
        _positive_int(num_trajectories, 'num_trajectories')
        _positive_int(num_steps, 'num_steps')
        if batch_size is not None:
            _positive_int(batch_size, 'batch_size')
            chunks = [self.sample_trajectories(min(batch_size, num_trajectories - start),
                      num_steps, generator=generator)
                      for start in range(0, num_trajectories, batch_size)]
            return TerminalLabelData(chunks[0].times,
                torch.cat([chunk.features for chunk in chunks]),
                torch.cat([chunk.labels for chunk in chunks]), chunks[0].weights)
        ref = self.conditioning.reference
        paths = generate_samples(ref, num_trajectories, num_steps=num_steps,
                                 generator=generator, return_trajectory=True)
        dt = self.horizon / num_steps
        times = torch.arange(num_steps, dtype=paths.dtype, device=paths.device) * dt
        features = torch.stack([self.conditioning.features(t, paths[:, j])
                                for j, t in enumerate(times)], dim=1)
        labels = (paths[:, -1] @ self.conditioning.w <= self.conditioning.C)
        weights = torch.full_like(times, dt / self.horizon)
        return TerminalLabelData(times, features, labels[:, None].to(paths.dtype), weights)


def terminal_label_loss(probabilities, labels, weights):
    """Sum time-weighted squared errors, then average over trajectories.

    probabilities: [N,J,1], labels: [N,1], weights: [J]. No class balancing,
    terminal-time term, or extra division by the number of time points is used.
    """
    if (probabilities.ndim != 3 or probabilities.shape[-1] != 1
            or labels.shape != (probabilities.shape[0], 1)
            or weights.shape != (probabilities.shape[1],)):
        raise ValueError("Expected probabilities [N,J,1], labels [N,1], weights [J]")
    return ((probabilities - labels[:, None]).square()
            * weights[None, :, None]).sum(dim=1).mean()


def _validate_data(model, data):
    dimension = model.conditioning.reference._score.k + 1
    if (data.features.ndim != 3 or min(data.features.shape) == 0
            or data.features.shape[2] != dimension):
        raise ValueError("features must have nonempty shape [N,J,k+1]")
    n, steps, _ = data.features.shape
    for tensor in (data.times, data.features, data.labels, data.weights):
        model.conditioning.reference._check_tensor(tensor, "trajectory data")
        if tensor.requires_grad:
            raise ValueError("Training trajectories must be detached from autograd")
    if data.times.shape != (steps,) or data.labels.shape != (n, 1) or data.weights.shape != (steps,):
        raise ValueError("Inconsistent trajectory/time/label/weight dimensions")
    if not ((data.labels == 0) | (data.labels == 1)).all():
        raise ValueError("Terminal labels must be binary")
    if data.times[0] != 0 or (data.times[1:] <= data.times[:-1]).any() or data.times[-1] >= model.horizon:
        raise ValueError("Times must increase from zero and exclude the terminal endpoint")
    edges = torch.cat((data.times, data.times.new_tensor([model.horizon])))
    expected_weights = edges.diff() / model.horizon
    if not torch.allclose(data.weights, expected_weights) or not (data.weights > 0).all():
        raise ValueError("Each time weight must be delta_t / (T-t0)")


def _predict_trajectories(model, times, features):
    n, steps, dimension = features.shape
    times = times[None, :].expand(n, -1).reshape(-1, 1)
    return model(times, features.reshape(-1, dimension)).reshape(n, steps, 1)


def train_h(model, data, *, epochs=100, batch_size=64, lr=1e-3, generator=None):
    """Train only H on complete trajectories; keep partial minibatches.

    data must come from this model's unguided reference and event. An optional
    shuffle generator must match the data device. Returns sample-weighted epoch
    losses; evaluation should use independent held-out trajectories.
    """
    _validate_data(model, data)
    _positive_int(epochs, "epochs")
    _positive_int(batch_size, "batch_size")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    ref = model.conditioning.reference
    if any(parameter.requires_grad or parameter.grad is not None for parameter in ref.parameters()):
        raise ValueError("The DFM reference must remain frozen with no parameter gradients")
    optimizer = torch.optim.Adam(model.network.parameters(), lr=lr)
    history = []
    modes = [(module, module.training) for module in model.modules()]
    n = data.features.shape[0]
    try:
        model.train()
        for _ in range(epochs):
            order = torch.randperm(n, device=data.features.device, generator=generator)
            total = 0.0
            for indices in order.split(batch_size):
                predictions = _predict_trajectories(model, data.times, data.features[indices])
                loss = terminal_label_loss(predictions, data.labels[indices], data.weights)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite H training loss")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(indices)
            history.append(total / n)
    finally:
        for module, training in modes:
            module.training = training
    return history


@torch.no_grad()
def evaluate_h(model, data, *, constant_probability, batch_size=256):
    """Held-out weighted squared loss and a supplied constant-probability baseline.

    Estimate the baseline probability from training labels only. These losses
    measure probability prediction, not accuracy of spatial/log-probability gradients.
    """
    _validate_data(model, data)
    _positive_int(batch_size, "batch_size")
    if not math.isfinite(constant_probability) or not 0 <= constant_probability <= 1:
        raise ValueError("constant_probability must lie in [0,1]")
    total = baseline = 0.0
    n = data.features.shape[0]
    for start in range(0, n, batch_size):
        features = data.features[start:start+batch_size]
        labels = data.labels[start:start+batch_size]
        predictions = _predict_trajectories(model, data.times, features)
        total += terminal_label_loss(predictions, labels, data.weights).item() * len(features)
        constant = torch.full_like(predictions, constant_probability)
        baseline += terminal_label_loss(constant, labels, data.weights).item() * len(features)
    return {"h_loss": total / n, "constant_loss": baseline / n}
