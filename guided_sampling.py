"""Time-dependent full-portfolio H guidance for a frozen DFM.

The guided drift is the Doob h-transform correction of Guo et al. (2026),
applied through reduced coordinates G_t.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import math
from numbers import Real
import time

import torch

from checkers import checked_output
from h_model import HNetwork, _positive_int
from sampling import generate_samples


def _synchronize(reference):
    device = next(reference.parameters()).device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


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
    if (not isinstance(model, HNetwork)
            or not callable(getattr(model, 'log_prob', None))):
        raise TypeError('H must be an HNetwork with log_prob(t, y)')
    if any(p.requires_grad or p.grad is not None
           for p in model.reference.parameters()):
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

    H must be pointwise across samples. Calling this object gives the ambient
    Doob correction from the supplied H. Neither H nor reference gradients
    are accumulated. Terminal time is excluded because H has a hard boundary.
    """

    def __init__(self, h_model):
        _check_h(h_model)
        self.h_model = h_model

    def _gradient(self, t, y):
        if torch.is_inference_mode_enabled():
            raise ValueError('Guidance needs autograd; use no_grad, not inference_mode')
        c = self.h_model
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
        c = self.h_model
        t = c._batch(time, state, c.reference.d)
        # Shared-time samplers need only one small coordinate matrix.
        G = c.G(t[0, 0] if torch.equal(t, t[:1].expand_as(t)) else t)
        y = state @ G.T if G.ndim == 2 else torch.bmm(G, state.unsqueeze(-1)).squeeze(-1)
        grad = self._gradient(t, y)
        lifted = grad @ G if G.ndim == 2 else torch.bmm(grad.unsqueeze(1), G).squeeze(1)
        return checked_output(lifted, state, 'ambient guidance').detach()

def _budget(num_samples, batch_size, max_proposals):
    _positive_int(num_samples, 'num_samples')
    _positive_int(batch_size, 'batch_size')
    if max_proposals is None:
        max_proposals = max(100 * num_samples, batch_size)
    _positive_int(max_proposals, 'max_proposals')
    if max_proposals < num_samples:
        raise ValueError('max_proposals must be at least num_samples')
    return max_proposals


def _normal(reference, n, generator):
    p = next(reference.parameters())
    return torch.randn(n, reference.d, dtype=p.dtype, device=p.device, generator=generator)


@torch.no_grad()
def sample_conditional_initial(h_model, num_samples, *, batch_size=1024,
                               max_proposals=None, generator=None):
    """Accept x~N(0,I) with probability H(0,G_0 x).

    Samples the normalized learned-H tilt exactly up to floating point; this
    equals the desired initial law only for the true H. Zero probabilities are
    allowed. A proposal budget failure raises instead of silently falling back.
    """
    _check_h(h_model)
    limit = _budget(num_samples, batch_size, max_proposals)
    c = h_model
    G = c.G(0.)
    accepted, count, proposed = [], 0, 0
    with evaluating(h_model):
        while count < num_samples and proposed < limit:
            x = _normal(c.reference, min(batch_size, limit - proposed), generator)
            log_h = _log_probability(h_model, 0., x @ G.T, allow_zero=True).squeeze(1)
            uniform = torch.rand(len(x), dtype=x.dtype, device=x.device, generator=generator)
            keep = uniform.log() < log_h
            accepted.append(x[keep])
            count += int(keep.sum())
            proposed += len(x)
    if count < num_samples:
        raise RuntimeError(f'Conditional initialization exhausted {proposed} proposals: '
                           f'accepted {count}/{num_samples}. Increase the budget or improve H.')
    return torch.cat(accepted)[:num_samples], {
        'method': 'h_rejection', 'proposals': proposed, 'accepted': count,
        'returned': num_samples, 'acceptance_rate': count / proposed,
        'target': 'normalized learned H(0,G0 x) times N(0,I)',
    }


@dataclass
class RejectionSamples:
    samples: torch.Tensor
    initial_states: torch.Tensor
    diagnostics: dict


@torch.no_grad()
def sample_reference_conditional(h_model, num_samples, *, num_steps=100,
                                 batch_size=1024, max_proposals=None, generator=None):
    """Independent rejection benchmark for the finite-step unguided reference.

    Returns terminal draws AND their initial states, conditioned on the full
    terminal event. Starts can validate H-based initialization. No learned H is
    used. This may be expensive for rare events; the budget is always finite.
    """
    if not isinstance(h_model, HNetwork):
        raise TypeError('h_model must be HNetwork')
    _positive_int(num_steps, 'num_steps')
    limit = _budget(num_samples, batch_size, max_proposals)
    ref = h_model.reference
    terminals, starts, count, proposed = [], [], 0, 0
    while count < num_samples and proposed < limit:
        x0 = _normal(ref, min(batch_size, limit - proposed), generator)
        x = generate_samples(ref, len(x0), num_steps=num_steps,
                             initial_state=x0, generator=generator)
        keep = x @ h_model.w <= h_model.C
        starts.append(x0[keep])
        terminals.append(x[keep])
        count += int(keep.sum())
        proposed += len(x)
    if count < num_samples:
        raise RuntimeError(f'Reference rejection exhausted {proposed} proposals: '
                           f'accepted {count}/{num_samples}. Increase max_proposals.')
    return RejectionSamples(torch.cat(terminals)[:num_samples],
                            torch.cat(starts)[:num_samples], {
        'method': 'reference_rejection', 'num_steps': num_steps,
        'proposals': proposed, 'accepted': count, 'returned': num_samples,
        'acceptance_rate': count / proposed,
        'target': 'finite-step unguided reference conditioned on terminal event',
    })


@dataclass
class ConditionalSamples:
    samples: torch.Tensor
    initial_states: torch.Tensor
    diagnostics: dict


def generate_conditional_samples(h_model, num_samples, *, num_steps=100,
                                 initialization='h_rejection',
                                 batch_size=1024, max_proposals=None,
                                 generator=None, initial_state=None,
                                 brownian_increments=None,
                                 return_trajectory=False, guidance_scale=1.0):
    """Run ambient guided sampling with the reference, event and horizon bound to H.

    initialization: h_rejection (default learned tilt), reference_rejection
    (finite-grid conditional starts), or gaussian (explicit biased ablation).
    Supplied starts override initialization and are marked as supplied/unverified.
    guidance_scale multiplies only the log-H drift correction: 1 gives the
    unscaled Doob-guided drift, 0 disables the correction, and values above 1 intensify
    it. Initialization is unchanged. Scaling can alter conditional-law accuracy
    and does not guarantee terminal event satisfaction.
    """
    if (isinstance(guidance_scale, bool) or not isinstance(guidance_scale, Real)
            or not math.isfinite(guidance_scale) or guidance_scale < 0):
        raise ValueError('guidance_scale must be a finite nonnegative real number')
    guidance_scale = float(guidance_scale)
    guidance = DFMHGuidance(h_model)
    _positive_int(num_samples, 'num_samples')
    _positive_int(num_steps, 'num_steps')
    _budget(num_samples, batch_size, max_proposals)
    if not isinstance(return_trajectory, bool):
        raise ValueError('return_trajectory must be boolean')
    if initialization not in ('h_rejection', 'reference_rejection', 'gaussian'):
        raise ValueError('Unknown initialization')
    c = h_model
    if brownian_increments is not None:
        c.reference._check_tensor(brownian_increments, 'brownian_increments')
        if brownian_increments.shape != (num_steps, num_samples, c.reference.d):
            raise ValueError('brownian_increments must have shape [steps,N,d]')
    _synchronize(c.reference)
    initialization_started = time.perf_counter()
    if initial_state is not None:
        c._batch(0., initial_state, c.reference.d)
        if len(initial_state) != num_samples:
            raise ValueError('initial_state batch must match num_samples')
        starts = initial_state.detach().clone()
        info = {'method': 'supplied', 'target': 'caller supplied; unverified'}
    elif initialization == 'h_rejection':
        starts, info = sample_conditional_initial(h_model, num_samples,
            batch_size=batch_size, max_proposals=max_proposals, generator=generator)
    elif initialization == 'reference_rejection':
        rejection = sample_reference_conditional(c, num_samples, num_steps=num_steps,
            batch_size=batch_size, max_proposals=max_proposals, generator=generator)
        starts, info = rejection.initial_states, rejection.diagnostics
    else:
        starts = _normal(c.reference, num_samples, generator)
        info = {'method': 'gaussian', 'target': 'unconditional N(0,I); ablation'}
    _synchronize(c.reference)
    initialization_seconds = time.perf_counter() - initialization_started
    integration_started = time.perf_counter()
    # Bound sampling memory without changing the event or truncating outputs.
    outputs = []
    with evaluating(h_model):
        for start in range(0, num_samples, batch_size):
            x0 = starts[start:start + batch_size]
            dw = None if brownian_increments is None else brownian_increments[:, start:start + len(x0)]
            kwargs = dict(num_steps=num_steps, generator=generator, initial_state=x0,
                          brownian_increments=dw, return_trajectory=return_trajectory,
                          guidance_scale=guidance_scale)
            output = generate_samples(c.reference, len(x0), guidance_fn=guidance, **kwargs)
            outputs.append(output)
    samples = torch.cat(outputs)
    terminal = samples[:, -1] if return_trajectory else samples
    portfolio = terminal @ c.w
    _synchronize(c.reference)
    integration_seconds = time.perf_counter() - integration_started
    return ConditionalSamples(samples, starts, {
        'space': 'ambient', 'num_steps': num_steps, 'guidance_scale': guidance_scale,
        'initialization': info, 'event_rate': float((portfolio <= c.C).double().mean()),
        'timing': {'initialization_seconds': initialization_seconds,
                   'integration_seconds': integration_seconds},
        'terminal_filtering': False,
        'target': ('approximate conditional frozen reference; H and discretization errors remain'
                   if guidance_scale == 1.0 else
                   'scaled-guidance approximation; scaling changes the sampling law'),
    })
