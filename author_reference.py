"""Frozen full-state DDPM reference; no fictitious factor projection."""
from copy import deepcopy
import torch
from torch import nn
from author_checkpoint import AuthorModel


class FrozenAuthorReference(nn.Module):
    def __init__(self, model, *, take_ownership=False):
        super().__init__()
        if not isinstance(model, AuthorModel):
            raise TypeError('Expected an AuthorModel')
        # Ownership transfer avoids a second billion-parameter copy when explicit.
        self.model = model if take_ownership else deepcopy(model)
        self.model.requires_grad_(False)
        for p in self.model.parameters():
            p.grad = None
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        return self

    @property
    def config(self):
        return self.model.config

    @property
    def d(self):
        return self.model.d

    @property
    def indexes(self):
        return tuple(range(self.config.timesteps - 1, self.config.save_index - 1, -1))

    def check_state(self, x):
        p = next(self.parameters())
        if (not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape != (len(x), self.d)
                or not len(x) or x.dtype != p.dtype or x.device != p.device
                or not torch.isfinite(x).all()):
            raise ValueError('State must be finite [N,d] and match reference dtype/device')

    def check_index(self, index):
        if type(index) is not int or not 0 <= index < self.config.timesteps:
            raise ValueError('Index must identify a DDPM transition')

    def normal(self, count, generator=None):
        if type(count) is not int or count < 1:
            raise ValueError('count must be a positive integer')
        p = next(self.parameters())
        return torch.randn(count, self.d, dtype=p.dtype, device=p.device, generator=generator)

    @torch.no_grad()
    def transition(self, x, index):
        self.check_state(x)
        self.check_index(index)
        batch = torch.full((len(x),), index, dtype=torch.long, device=x.device)
        mean, variance, log_variance, _ = self.model.diffusion.p_mean_variance(
            x.reshape(-1, 1, *self.config.image_size), batch, clip_denoised=False)
        return mean.flatten(1), variance.reshape(-1, 1), log_variance.reshape(-1, 1)

    @torch.no_grad()
    def step(self, x, index, *, generator=None, mean_shift=None):
        mean, _, log_variance = self.transition(x, index)
        if mean_shift is not None:
            self.check_state(mean_shift)
            if mean_shift.shape != x.shape:
                raise ValueError('Mean shift must match the state batch')
            mean = mean + mean_shift
        noise = self.normal(len(x), generator) if index > 0 else 0.
        result = mean + (.5 * log_variance).exp() * noise
        self.check_state(result)
        return result
