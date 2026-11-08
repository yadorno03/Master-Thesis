"""Fixed full-portfolio event in the model's return coordinates."""

from dataclasses import dataclass
import math

import torch


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
