"""Smooth terminal-event probability network and trajectory regression loss."""

from dataclasses import dataclass
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

from conditioning import DFMConditioning


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
