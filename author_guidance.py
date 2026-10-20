"""Approximate DDPM mean-shift guidance; not part of the authors' DFM."""
import torch
from checkers import validate_guidance_scale
from author_h import DDPMHNetwork
from author_sampling import sample_author_ddpm
from guided_sampling import ConditionalSamples, evaluating



@torch.no_grad()
def _initial(h, count, batch_size, limit, generator):
    accepted, total, proposed = [], 0, 0
    while total < count and proposed < limit:
        x = h.reference.normal(min(batch_size, limit-proposed), generator)
        log_h = h.log_prob(h.reference.indexes[0], x)[:, 0]
        if torch.isnan(log_h).any() or (log_h > 0).any():
            raise ValueError('H must return log probabilities <= 0')
        uniform = torch.rand(len(x), dtype=x.dtype, device=x.device, generator=generator)
        keep = uniform.log() < log_h
        accepted.append(x[keep]); total += int(keep.sum()); proposed += len(x)
    if total < count:
        raise RuntimeError(f'Initial rejection exhausted {proposed} proposals: accepted {total}/{count}')
    return torch.cat(accepted)[:count], dict(method='h_rejection', proposals=proposed,
        accepted=total, returned=count, discarded=proposed-count)


def sample_guided_author(h, count, *, guidance_scale=1., initialization='h_rejection',
                         batch_size=64, max_proposals=100000, generator=None, initial_state=None):
    """Shift DDPM mean by scale * posterior_variance * grad_x log H(index,x).

    This is a first-order discrete guidance approximation, not an exact Doob
    transition or a published Xu-group algorithm. Initial H-tilting is retained
    when scale=0; use initialization='gaussian' for an unguided equivalence check.
    No terminal filtering is applied and event satisfaction is not guaranteed.
    """
    if not isinstance(h, DDPMHNetwork):
        raise TypeError('Expected DDPMHNetwork')
    scale = validate_guidance_scale(guidance_scale)
    if any(type(n) is not int or n < 1 for n in (count, batch_size, max_proposals)) or max_proposals < count:
        raise ValueError('Require positive counts and max_proposals >= count')
    if initialization not in ('h_rejection', 'gaussian'):
        raise ValueError('Unknown initialization')
    if any(p.requires_grad or p.grad is not None for p in h.reference.parameters()):
        raise ValueError('Reference must remain frozen')
    with evaluating(h), torch.no_grad():
        if initial_state is not None:
            h.reference.check_state(initial_state)
            if len(initial_state) != count:
                raise ValueError('Initial states must match count')
            starts = initial_state.detach().clone()
            info = {'method': 'supplied', 'target': 'caller supplied; unverified'}
        elif initialization == 'h_rejection':
            starts, info = _initial(h, count, batch_size, max_proposals, generator)
        else:
            starts = h.reference.normal(count, generator)
            info = {'method': 'gaussian', 'target': 'unconditional N(0,I)'}
        def correction(index, x):
            variance = h.reference.model.diffusion.posterior_variance[index]
            return scale * variance * h.gradient(index, x)
        samples = torch.cat([sample_author_ddpm(h.reference, len(x), initial_state=x,
            generator=generator, mean_shift_fn=correction if scale else None)
            for x in starts.split(batch_size)])
        return ConditionalSamples(samples, starts, dict(backend='author_unet_ddpm',
            guidance_scale=scale, initialization=info, terminal_filtering=False,
            save_index=h.reference.config.save_index,
            event_rate=float(h.terminal_event(samples).float().mean()),
            target='approximate DDPM event guidance; H, mean-shift and scaling errors remain'))
