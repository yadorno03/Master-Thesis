"""Fixed full-portfolio events and training-coordinate conversion."""

from dataclasses import dataclass
import math

import torch


def portfolio_event_in_model_coordinates(weights, threshold, mean, std):
    """Convert w_raw @ (mean + std*x) <= C_raw without losing the offset.

    Vectors must already have the checkpoint's asset order. This helper does
    not fit preprocessing or use evaluation data.
    """
    config = ConditioningConfig(weights, threshold)
    w = torch.tensor(config.w, dtype=torch.float64)
    for value in (mean, std):
        raw = torch.as_tensor(value)
        if raw.is_complex() or raw.dtype == torch.bool:
            raise ValueError('Preprocessing must be real-valued')
    mean = torch.as_tensor(mean, dtype=torch.float64).detach().cpu()
    std = torch.as_tensor(std, dtype=torch.float64).detach().cpu()
    if (mean.shape != w.shape or std.shape != w.shape
            or not torch.isfinite(mean).all() or not torch.isfinite(std).all()
            or not (std > 0).all()):
        raise ValueError('mean/std must match weights, be finite, and std must be positive')
    return ConditioningConfig(w * std, config.C - float(w @ mean))


@dataclass(frozen=True)
class ConditioningConfig:
    """Fixed event U_(T-t0) <= C, with w in the model's return coordinates.

    w is stored as an immutable tuple, even when supplied as a tensor or list.
    """

    w: tuple[float, ...]
    C: float

    def __post_init__(self):
        raw_w, raw_C = torch.as_tensor(self.w), torch.as_tensor(self.C)
        if (raw_w.is_complex() or raw_w.dtype == torch.bool
                or raw_C.is_complex() or raw_C.dtype == torch.bool):
            raise ValueError("w and C must be real-valued")
        w = torch.as_tensor(self.w, dtype=torch.float64).detach().cpu()
        if (w.ndim != 1 or w.numel() == 0 or not torch.isfinite(w).all()
                or not (w != 0).any()):
            raise ValueError("w must be a finite nonzero vector")
        if raw_C.ndim != 0 or not math.isfinite(float(self.C)):
            raise ValueError("C must be a finite scalar")
        object.__setattr__(self, "w", tuple(w.tolist()))
        object.__setattr__(self, "C", float(self.C))
