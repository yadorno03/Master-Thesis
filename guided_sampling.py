"""Time-dependent full-portfolio H guidance for a frozen DFM.

The guided drift is the Doob h-transform correction of Guo et al. (2026),
applied through reduced coordinates G_t. sample_guided_smc corrects the guided
sampler by sequential Monte Carlo, with H as twisting function (Wu et al., 2023).
"""

from contextlib import contextmanager
import math

import torch

from checkers import checked_output
from h_model import HNetwork, _positive_int


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

    def _gradient(self, t, y, *, with_value=False):
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
        gradient = checked_output(gradient.detach(), y, 'log-H gradient')
        return (log_h.detach().squeeze(1), gradient) if with_value else gradient

    def value_and_gradient(self, state, time):
        """log H(t, G_t x) as [batch] and its ambient gradient as [batch, d], from one evaluation of H."""
        c = self.h_model
        t = c._batch(time, state, c.reference.d)
        # Shared-time samplers need only one small coordinate matrix.
        G = c.G(t[0, 0] if torch.equal(t, t[:1].expand_as(t)) else t)
        y = state @ G.T if G.ndim == 2 else torch.bmm(G, state.unsqueeze(-1)).squeeze(-1)
        log_h, grad = self._gradient(t, y, with_value=True)
        lifted = grad @ G if G.ndim == 2 else torch.bmm(grad.unsqueeze(1), G).squeeze(1)
        return log_h, checked_output(lifted, state, 'ambient guidance').detach()

    def __call__(self, state, time):
        return self.value_and_gradient(state, time)[1]

@torch.no_grad()
def exact_final_step(h_model, x, num_steps, generator=None):
    """Exact finite-grid conditional last transition X_{J-1} -> X_J given w'X_J <= C.

    The unguided Euler step N(mu(x), dt I), mu(x) = x + dt (x/2 + s(x, T - t_{J-1})),
    restricted to the event: a standard normal truncated to (-inf, a(x)] along
    u = w/|w|, with a(x) = (C - w'mu(x)) / (sqrt(dt)|w|), and unconstrained noise
    orthogonal to u. Uses no H. States whose event mass underflows land on C.
    """
    reference = h_model.reference
    dt = h_model.horizon / num_steps
    t = h_model.horizon - dt
    mean = x + dt * (.5 * x + reference.score(x, reference.T - t))
    w = h_model._w.to(x.dtype)
    norm = w.norm()
    direction = w / norm
    bound = ((h_model.C - mean @ w) / (math.sqrt(dt) * norm)).double()
    uniform = torch.rand(len(x), generator=generator, dtype=torch.float64, device=x.device)
    mass = torch.special.ndtr(bound)
    along = torch.special.ndtri((uniform * mass).clamp_min(1e-300))
    along = torch.where(mass > 0, torch.minimum(along, bound), bound)
    noise = torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device)
    orthogonal = noise - (noise @ direction)[:, None] * direction
    return mean + math.sqrt(dt) * (orthogonal + along.to(x.dtype)[:, None] * direction)


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


def systematic_resample(weights, generator=None):
    """Indices of a systematic resampling of normalized weights [n]: one uniform draw, n evenly spaced points.

    Index i is drawn floor(n w_i) or that plus one times, the lowest-variance
    unbiased scheme among the usual ones (Kitagawa, 1996).
    """
    if weights.ndim != 1 or not len(weights) or (weights < 0).any() or not torch.isfinite(weights).all():
        raise ValueError('weights must be a nonempty vector of finite nonnegative values')
    n = len(weights)
    cumulative = (weights.double() / weights.double().sum()).cumsum(0)
    offset = torch.rand(1, generator=generator, dtype=torch.float64, device=weights.device)
    points = (offset + torch.arange(n, dtype=torch.float64, device=weights.device)) / n
    return torch.searchsorted(cumulative, points).clamp_max(n - 1)


def _effective_fraction(log_weights):
    """Effective sample size of unnormalized log weights, as a share of the particles (Kong, 1992)."""
    weights = torch.softmax(log_weights, 0)
    return float(1. / (weights.square().sum() * len(weights)))


@torch.no_grad()
def sample_guided_smc(h_model, num_samples, num_steps, *, generator=None, ess_threshold=.5,
                      batch_size=1024, max_proposals=None):
    """Guided sampler corrected by sequential Monte Carlo: num_samples equally weighted terminal states.

    The num_samples paths are the particles of one system. They move exactly as
    in the guided sampler: H-tilted start, guided Euler steps, exact final step.
    The intermediate target at step j <= J-2 is the unguided chain up to t_j
    times H(t_j, G_j x_j), and at step J-1 the unguided chain times the exact
    last-step probability p_last(x_{J-1}): H is the twisting function of a
    twisted particle filter (Wu et al., 2023). The weight of a particle is
    multiplied at each step by

        [twist(x_{j+1}) / twist(x_j)] * [unguided step density / guided step density],

    which is 1 for the exact H in continuous time. The product over all steps is
    the importance weight of the path against the DFM given its event. When the
    effective sample size falls below ess_threshold times num_samples, the
    particles are resampled in proportion to their weights (systematic
    resampling), so paths that the learned H sent wrong are replaced by copies
    of better ones, which the later noise spreads again. A last resampling
    before the exact final step returns equally weighted rows. As num_samples
    grows, the rows follow the unguided chain conditioned on its event, the law
    DFM rejection samples, whatever H is; H only sets how many particles this
    takes.

    Returns (states [num_samples, d], diagnostics). The diagnostics give the
    resampling times, the lowest and last effective sample sizes, the share of
    distinct paths at the last resampling and of distinct starting points, and
    the estimate of P(event) under the frozen DFM: the acceptance rate of the
    start times the product of the mean weights at the resampling times.
    """
    _check_h(h_model)
    _positive_int(num_steps, 'num_steps')
    if isinstance(ess_threshold, bool) or not 0 <= ess_threshold <= 1:
        raise ValueError('ess_threshold must lie in [0, 1]')
    reference = h_model.reference
    guidance = DFMHGuidance(h_model)
    x, initialization = sample_conditional_initial(h_model, num_samples, batch_size=batch_size,
                                                   max_proposals=max_proposals, generator=generator)
    n = num_samples
    dt = h_model.horizon / num_steps
    sqrt_dt = math.sqrt(dt)
    log_weights = torch.zeros(n, dtype=torch.float64, device=x.device)
    ancestors = torch.arange(n, device=x.device)
    log_normalizer, resampled_at, lowest = 0., [], 1.

    def resample(log_weights, *tensors):
        nonlocal log_normalizer, ancestors
        log_normalizer += float(torch.logsumexp(log_weights, 0)) - math.log(n)
        index = systematic_resample(torch.softmax(log_weights, 0), generator)
        ancestors = ancestors[index]
        return index, [tensor[index] for tensor in tensors]

    with evaluating(h_model):
        # The H-tilted start has equal weights: its twist cancels the tilt.
        twist = _log_probability(h_model, 0., x @ h_model.G(0.).T).squeeze(1).double()
        for j in range(num_steps - 1):
            t = torch.full((n, 1), j * dt, dtype=x.dtype, device=x.device)
            log_h, correction = guidance.value_and_gradient(x, t)
            if j:
                log_weights += log_h.double() - twist
            twist = log_h.double()
            fraction = _effective_fraction(log_weights)
            lowest = min(lowest, fraction)
            if j and fraction < ess_threshold:
                _, (x, t, twist, correction) = resample(log_weights, x, t, twist, correction)
                log_weights = torch.zeros_like(log_weights)
                resampled_at.append(j)
            increment = sqrt_dt * torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
            log_weights -= ((correction * increment).sum(1).double()
                            + .5 * dt * correction.square().sum(1).double())
            x = x + dt * (.5 * x + reference.score(x, reference.T - t) + correction) + increment
            if not torch.isfinite(x).all():
                raise ValueError('Sampling produced nonfinite values')
        last = h_model.exact_last_step_probability(x, dt, log=True).squeeze(1).double()
        log_weights += last - twist
        if not torch.isfinite(log_weights).any():
            raise ValueError('Every particle has zero weight before the final step')
        final_fraction = _effective_fraction(log_weights)
        index, (x,) = resample(log_weights, x)
        x = exact_final_step(h_model, x, num_steps, generator)
    if not torch.isfinite(x).all():
        raise ValueError('Sampling produced nonfinite values')
    return x, dict(
        particles=n, steps=num_steps, ess_threshold=float(ess_threshold), resampling_steps=resampled_at,
        resampling_count=len(resampled_at) + 1, lowest_effective_sample_fraction=min(lowest, final_fraction),
        final_effective_sample_fraction=final_fraction,
        distinct_final_paths_fraction=len(torch.unique(index)) / n,
        distinct_starts_fraction=len(torch.unique(ancestors)) / n,
        event_probability=initialization['acceptance_rate'] * math.exp(log_normalizer),
        initialization=initialization)
