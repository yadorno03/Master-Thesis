"""Frozen score and time-dependent coordinates for ambient DFM conditioning."""

from copy import deepcopy
import math

import torch
from torch import nn

from checkers import validate_projection


class FrozenDFMReference(nn.Module):
    """Snapshot a trained DFM and expose only its score and coordinate map.

    score(x, s) uses forward time s; G(t, w) uses generative time t.
    T and t0 are read-only copies of the model's time settings. Parameters stay
    frozen and in evaluation mode; ordinary autograd still tracks input gradients.
    State and portfolio tensors must match the reference's dtype and device.
    """

    def __init__(self, model):
        super().__init__()
        self._times = (float(model.T), float(model.t0))
        if not all(map(math.isfinite, self._times)) or not 0 < self.t0 < self.T:
            raise ValueError("Require finite times with 0 < t0 < T")
        self._score = deepcopy(model.factor_score_network)
        self._score.requires_grad_(False)
        for parameter in self.parameters():
            parameter.grad = None
            if (parameter.dtype != self._score.V.dtype
                    or parameter.device != self._score.V.device
                    or not torch.isfinite(parameter).all()):
                raise ValueError("Reference parameters must be finite with one dtype/device")
        c = self._score.c_raw.sigmoid() * self._score.sigma_max
        if not torch.isfinite(c).all() or not (c > 0).all():
            raise ValueError("Learned variances must be finite and strictly positive")
        validate_projection(self._score.V, self.d,
                            device=self._score.V.device, dtype=self._score.V.dtype)
        self.train(False)

    @property
    def T(self):
        return self._times[0]

    @property
    def t0(self):
        return self._times[1]

    @property
    def d(self):
        return self._score.d

    def train(self, mode=True):
        super().train(False)
        return self

    def _time(self, value, lower, upper):
        raw = torch.as_tensor(value)
        if raw.is_complex() or raw.dtype == torch.bool:
            raise ValueError("Time must be real-valued")
        time = torch.as_tensor(value, dtype=self._score.V.dtype,
                               device=self._score.V.device)
        if time.ndim <= 1:
            time = time.reshape(-1, 1)
        if (time.ndim != 2 or time.shape[1] != 1 or time.shape[0] == 0
                or not torch.isfinite(time).all()
                or (time < lower).any() or (time > upper).any()):
            raise ValueError(f"Time must be scalar, [batch], or [batch, 1] in [{lower}, {upper}]")
        return time

    def _check_tensor(self, value, name):
        if (not isinstance(value, torch.Tensor)
                or value.dtype != self._score.V.dtype
                or value.device != self._score.V.device
                or not torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite and match reference dtype/device")

    def score(self, x, s):
        """Ambient score at forward time s in [t0, T], for x shaped [batch, d]."""
        self._check_tensor(x, "x")
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != self.d:
            raise ValueError("x must have nonempty shape [batch, d]")
        s = self._time(s, self.t0, self.T)
        if s.shape[0] == 1:
            s = s.expand(x.shape[0], 1)
        if s.shape[0] != x.shape[0]:
            raise ValueError("Time batch must match state batch")
        return self._score(x, s)

    # generate_samples reads model.factor_score_network as the score.
    factor_score_network = score

    def _schedule(self, t):
        """Internal forward-time quantities for validated generative times."""
        t = self._time(t, 0, self.T - self.t0)
        s = self.T - t
        c = self._score.c_raw.sigmoid() * self._score.sigma_max
        alpha = torch.exp(-s / 2)
        D = (-torch.expm1(-s) + torch.exp(-s) * c).reciprocal()
        lambda0 = -math.expm1(-self.t0) + math.exp(-self.t0) * c
        return s, alpha, D, lambda0

    def G(self, t, w):
        """Return G_t = [B^T D_(T-t); ell_t^T] for a portfolio w of shape [d].

        A scalar t returns [k+1, d]; [batch] or [batch, 1] times return
        [batch, k+1, d]. G_t maps asset states to (Z,U), with U=w^T X at
        t=T-t0. w must be nonzero and expressed in the model's coordinates.
        """
        self._check_tensor(w, "w")
        if w.shape != (self.d,) or not (w != 0).any():
            raise ValueError("w must be a nonzero vector with shape [d]")
        scalar_time = torch.as_tensor(t).ndim == 0
        s, _, D, lambda0 = self._schedule(t)
        ell = torch.exp((self.t0 - s) / 2) * D * lambda0 * w
        factors = self._score.V.T.unsqueeze(0) * D.unsqueeze(1)
        matrix = torch.cat((factors, ell.unsqueeze(1)), dim=1)
        return matrix[0] if scalar_time else matrix
