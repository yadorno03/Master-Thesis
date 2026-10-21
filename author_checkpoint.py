"""Portable, weights-only checkpoints for the pinned numerical DFM."""
import json
from pathlib import Path
import torch
from torch import nn
from author_dfm import AUTHOR_COMMIT, AuthorDFMConfig, build_author_dfm
from author_data import ReturnTransform


class AuthorModel(nn.Module):
    def __init__(self, diffusion, transform, config):
        super().__init__()
        if (not isinstance(config, AuthorDFMConfig) or not isinstance(transform, ReturnTransform)
                or len(transform.assets) != config.d
                or tuple(diffusion.image_size) != tuple(config.image_size)
                or diffusion.num_timesteps != config.timesteps):
            raise ValueError('Diffusion, preprocessing and configuration must agree')
        self.diffusion, self.transform, self.config = diffusion, transform, config

    @property
    def d(self):
        return self.config.d

    def forward(self, standardized):
        """Training batches are already in the saved standardized coordinates."""
        return self.diffusion(standardized.reshape(-1, 1, *self.config.image_size))

    def checkpoint_payload(self, metadata=None):
        state = {k: v.detach().cpu().clone() for k, v in self.diffusion.state_dict().items()}
        if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in state.values()):
            raise ValueError('Author checkpoints require finite float32 weights and buffers')
        metadata = json.loads(json.dumps(metadata or {}, allow_nan=False))
        return dict(format='author-unet-ddpm-v1', **self.config.metadata(), state_dict=state,
                    mean=self.transform.mean.detach().cpu(), std=self.transform.std.detach().cpu(),
                    assets=self.transform.assets, metadata=metadata)

    def save_checkpoint(self, path, *, metadata=None):
        """Never overwrite an existing artifact."""
        payload = self.checkpoint_payload(metadata)
        with Path(path).open('xb') as handle:
            torch.save(payload, handle)

    @classmethod
    def from_payload(cls, payload, *, device='cpu'):
        if (payload.get('format') != 'author-unet-ddpm-v1'
                or payload.get('upstream_commit') != AUTHOR_COMMIT):
            raise ValueError('Unsupported checkpoint format or upstream revision')
        settings = dict(payload['config'])
        settings['image_size'] = tuple(settings['image_size'])
        config = AuthorDFMConfig(**settings)
        transform = ReturnTransform(payload['mean'], payload['std'], payload['assets'])
        state = payload['state_dict']
        if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in state.values()):
            raise ValueError('Checkpoint tensors must be finite float32')
        with torch.random.fork_rng(devices=[]):
            diffusion = build_author_dfm(config)
        diffusion.load_state_dict(state, strict=True)
        return cls(diffusion, transform, config).to(device).eval()

    @classmethod
    def load_checkpoint(cls, path, *, device='cpu'):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        return cls.from_payload(payload, device=device), payload['metadata']
