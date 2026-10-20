"""Full-state terminal-event H for a frozen DDPM (our guidance extension)."""
import torch
from torch import nn
from torch.nn import functional as F
from author_reference import FrozenAuthorReference
from conditioning import ConditioningConfig


class DDPMHNetwork(nn.Module):
    """H(index,x) predicts the event at the reference's saved endpoint.

    x is the full standardized ambient state BEFORE transition index. There is
    no factor G, reduced closed drift, or claimed U-Net latent sufficiency.
    This conditioning model is separate from the authors' unconditional DFM.
    """
    def __init__(self, reference, event, hidden_dims=(64, 64)):
        super().__init__()
        if not isinstance(reference, FrozenAuthorReference) or not isinstance(event, ConditioningConfig):
            raise TypeError('Require FrozenAuthorReference and ConditioningConfig')
        if len(event.w) != reference.d:
            raise ValueError('Event weights must match the reference asset order')
        self.reference, self.event = reference, event
        self.hidden_dims = tuple(hidden_dims)
        if not self.hidden_dims or any(type(n) is not int or n < 1 for n in self.hidden_dims):
            raise ValueError('Hidden widths must be positive integers')
        p = next(reference.parameters())
        self.register_buffer('w', torch.tensor(event.w, dtype=p.dtype, device=p.device))
        self.register_buffer('threshold', torch.tensor(event.C, dtype=p.dtype, device=p.device))
        if not torch.isfinite(self.w).all() or not (self.w != 0).any() or not torch.isfinite(self.threshold):
            raise ValueError('Event must remain finite and nonzero in reference precision')
        layers, width = [], reference.d + 1
        for hidden in self.hidden_dims:
            layers.extend((nn.Linear(width, hidden), nn.Tanh()))
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers).to(dtype=p.dtype, device=p.device)

    def _logits(self, index, x):
        self.reference.check_state(x)
        indexes = torch.as_tensor(index, device=x.device)
        if indexes.dtype == torch.bool or indexes.is_complex() or indexes.is_floating_point():
            raise ValueError('DDPM indices must be integers')
        if indexes.ndim == 0:
            indexes = indexes.expand(len(x))
        if (indexes.shape != (len(x),) or (indexes < self.reference.config.save_index).any()
                or (indexes >= self.reference.config.timesteps).any()):
            raise ValueError('Indices must match the batch and precede the saved endpoint')
        time = indexes.to(x.dtype)[:, None] / max(1, self.reference.config.timesteps - 1)
        return self.network(torch.cat((x, time), dim=1))

    def forward(self, index, x):
        return torch.sigmoid(self._logits(index, x))

    def log_prob(self, index, x):
        return F.logsigmoid(self._logits(index, x))

    def terminal_event(self, x):
        self.reference.check_state(x)
        return x @ self.w <= self.threshold

    def gradient(self, index, x):
        """Spatial log-H gradient; no reference or H parameter grads accumulate."""
        if torch.is_inference_mode_enabled():
            raise ValueError('H guidance requires autograd; use no_grad instead')
        with torch.enable_grad():
            state = x.detach().requires_grad_(True)
            log_h = self.log_prob(index, state)
            gradient = torch.autograd.grad(log_h.sum(), state)[0].detach()
        self.reference.check_state(gradient)
        return gradient
