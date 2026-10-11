"""Full-portfolio DFM coordinates from Rework.pdf, Section 2 and Appendix C."""

from dataclasses import dataclass
import math

import torch
from torch import nn

from reference import FrozenDFMReference


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


class DFMConditioning(nn.Module):
    """Coordinates and reduced coefficients for one frozen reference and event.

    All methods take generative time t in [0, reference.T-reference.t0].
    Scalar times produce unbatched ell/G/A; [batch] or [batch, 1] times produce
    batched coefficients. features and mu always take and return row batches.
    G is time dependent and may be rank deficient; no orthonormality is imposed.
    """

    def __init__(self, reference, config):
        super().__init__()
        if not isinstance(reference, FrozenDFMReference):
            raise TypeError("reference must be a FrozenDFMReference")
        if not isinstance(config, ConditioningConfig):
            raise TypeError("config must be a ConditioningConfig")
        if len(config.w) != reference.d:
            raise ValueError("w must contain one weight per asset")
        self.reference = reference
        self._config = config
        parameter = next(reference.parameters())
        self.register_buffer("_w", torch.tensor(
            config.w, device=parameter.device, dtype=parameter.dtype))
        if not torch.isfinite(self._w).all() or not (self._w != 0).any():
            raise ValueError("w must remain finite and nonzero in reference precision")
        if not torch.isfinite(torch.tensor(config.C, dtype=parameter.dtype)):
            raise ValueError("C must remain finite in reference precision")

    @property
    def config(self):
        return self._config

    @property
    def w(self):
        """Return a copy so callers cannot mutate the configured portfolio."""
        return self._w.clone()

    @property
    def C(self):
        return self.config.C

    def G(self, t):
        """G_t: [k+1, d] or [batch, k+1, d], retaining the complete U row."""
        return self.reference.G(t, self._w)

    def ell(self, t):
        """ell_t = (alpha_(T-t)/alpha_t0) D_(T-t) Lambda_t0 w."""
        return self.G(t)[..., -1, :]

    def _batch(self, t, state, dimension):
        self.reference._check_tensor(state, "state")
        if state.ndim != 2 or state.shape[0] == 0 or state.shape[1] != dimension:
            raise ValueError(f"state must have nonempty shape [batch, {dimension}]")
        t = self.reference._time(t, 0, self.reference.T - self.reference.t0)
        if t.shape[0] == 1:
            t = t.expand(state.shape[0], 1)
        if t.shape[0] != state.shape[0]:
            raise ValueError("Time batch must match state batch")
        return t

    def features(self, t, x):
        """Y=(Z,U)=G_t X, returned as [batch, k+1]."""
        t = self._batch(t, x, self.reference.d)
        return torch.bmm(self.G(t), x.unsqueeze(-1)).squeeze(-1)

    def mu(self, t, y):
        """Explicit closed reduced drift (Rework.pdf, equation 2.5).

        The U drift has no linear U term, but the U coordinate is retained.
        No ambient state reconstruction or separate drift network is used.
        """
        network = self.reference._score
        t = self._batch(t, y, network.k + 1)
        s, alpha, D, lambda0 = self.reference._schedule(t)
        z = y[:, :network.k]
        n = network.g_zeta(torch.cat((z, s), dim=-1))
        D2Bn = D.square() * (n @ network.V.T)
        mu_z = -0.5 * z + alpha * (D2Bn @ network.V)
        mu_u = (alpha.square() / math.exp(-self.reference.t0 / 2)
                * (D2Bn * lambda0 * self._w).sum(dim=-1, keepdim=True))
        return torch.cat((mu_z, mu_u), dim=-1)

    def A(self, t):
        """Reduced noise covariance G_t G_t^T; may be singular."""
        G = self.G(t)
        return G @ G.transpose(-1, -2)

    def terminal_event(self, y):
        """Evaluate U <= C on terminal reduced states [batch, k+1]."""
        self._batch(self.reference.T - self.reference.t0, y,
                    self.reference._score.k + 1)
        return y[:, -1] <= self.C
