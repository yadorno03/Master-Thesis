"""Trajectory data and time-weighted terminal-event probability loss."""

from dataclasses import dataclass

import torch


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
