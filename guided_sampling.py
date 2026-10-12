"""Time-dependent full-portfolio H guidance for a frozen DFM."""

from contextlib import contextmanager

import torch

from checkers import checked_output, guidance_scale
from conditioning import DFMConditioning


@contextmanager
def evaluating(module):
    """Preserve all module modes, including when sampling raises."""
    modes = [(m, m.training) for m in module.modules()]
    try:
        module.eval()
        yield
    finally:
        for m, training in modes:
            m.training = training


def _check_h(model):
    if (not isinstance(model, torch.nn.Module)
            or not isinstance(getattr(model, 'conditioning', None), DFMConditioning)
            or not callable(getattr(model, 'log_prob', None))):
        raise TypeError('H must be a module with conditioning and log_prob(t, y)')
    if any(p.requires_grad or p.grad is not None
           for p in model.conditioning.reference.parameters()):
        raise ValueError('The DFM reference must remain frozen with no gradients')


def _log_probability(model, t, y, *, allow_zero=False):
    value = model.log_prob(t, y)
    if (not isinstance(value, torch.Tensor) or value.shape != (len(y), 1)
            or value.dtype != y.dtype or value.device != y.device
            or torch.isnan(value).any() or (value > 0).any()
            or (not allow_zero and not torch.isfinite(value).all())):
        raise ValueError('log_prob must return [batch, 1] log probabilities <= 0 '
                         'with matching dtype/device (finite for guidance)')
    return value


class DFMHGuidance:
    """Lift grad_y log H through time-dependent G, without orthogonality assumptions.

    H must be pointwise across samples. Calling this object gives ambient
    guidance; reduced(t, y) gives covariance-weighted guidance. Only scale=1
    is the supplied H's Doob correction. Neither H nor reference gradients
    are accumulated. Terminal time is excluded because H has a hard boundary.
    """

    def __init__(self, h_model, *, scale=1.0):
        _check_h(h_model)
        self.h_model = h_model
        self.conditioning = h_model.conditioning
        self.scale = guidance_scale(scale)

    def _gradient(self, t, y):
        if torch.is_inference_mode_enabled():
            raise ValueError('Guidance needs autograd; use no_grad, not inference_mode')
        c = self.conditioning
        t = c._batch(t, y, c.reference._score.k + 1)
        if (t >= c.reference.T - c.reference.t0).any():
            raise ValueError('Guidance requires preterminal generative time')
        with evaluating(self.h_model), torch.enable_grad():
            y = y.detach().requires_grad_(True)
            log_h = _log_probability(self.h_model, t.detach(), y)
            if not log_h.requires_grad:
                raise ValueError('log_prob must be differentiably connected to y')
            gradient = torch.autograd.grad(log_h.sum(), y)[0]
        return checked_output(gradient.detach(), y, 'log-H gradient')

    def __call__(self, state, time):
        c = self.conditioning
        t = c._batch(time, state, c.reference.d)
        # Shared-time samplers need only one small coordinate matrix.
        G = c.G(t[0, 0] if torch.equal(t, t[:1].expand_as(t)) else t)
        y = state @ G.T if G.ndim == 2 else torch.bmm(G, state.unsqueeze(-1)).squeeze(-1)
        grad = self._gradient(t, y)
        lifted = grad @ G if G.ndim == 2 else torch.bmm(grad.unsqueeze(1), G).squeeze(1)
        return checked_output(self.scale * lifted, state, 'ambient guidance').detach()

    def reduced(self, time, state):
        c = self.conditioning
        t = c._batch(time, state, c.reference._score.k + 1)
        A = c.A(t[0, 0] if torch.equal(t, t[:1].expand_as(t)) else t)
        grad = self._gradient(t, state)
        correction = grad @ A if A.ndim == 2 else torch.bmm(grad.unsqueeze(1), A).squeeze(1)
        return checked_output(self.scale * correction, state, 'reduced guidance').detach()
