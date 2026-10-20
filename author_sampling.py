"""Seedable DDPM sampling with the authors' after-transition save convention."""
import torch
from author_reference import FrozenAuthorReference


@torch.no_grad()
def sample_author_ddpm(reference, count, *, generator=None, initial_state=None,
                       return_trajectory=False, mean_shift_fn=None):
    """Return standardized [N,d], or [N,J+1,d] through the saved endpoint.

    mean_shift_fn(index, state) returns a mean shift, NOT a score or OU drift.
    Saved index 20 means the state AFTER transition 20 (forward index 19).
    Like upstream, finish the discarded tail to zero, preserving RNG advancement
    across successive sample batches. Guidance is used only through the endpoint.
    """
    if not isinstance(reference, FrozenAuthorReference):
        raise TypeError('Expected FrozenAuthorReference')
    if type(count) is not int or count < 1 or not isinstance(return_trajectory, bool):
        raise ValueError('Require a positive integer count and boolean return_trajectory')
    x = reference.normal(count, generator) if initial_state is None else initial_state.detach().clone()
    reference.check_state(x)
    if len(x) != count:
        raise ValueError('Initial state batch must match count')
    path = [x.clone()] if return_trajectory else None
    endpoint = reference.config.save_index
    saved = None
    for index in range(reference.config.timesteps - 1, -1, -1):
        shift = mean_shift_fn(index, x) if mean_shift_fn is not None and index >= endpoint else None
        x = reference.step(x, index, generator=generator, mean_shift=shift)
        if index >= endpoint and path is not None:
            path.append(x.clone())
        if index == endpoint:
            saved = x.clone()
    return torch.stack(path, 1) if path is not None else saved
