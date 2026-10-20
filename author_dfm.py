"""Pinned numerical DFM from Chen, Xu, Xu and Zhang (MIT upstream).

This is separate from DFM.UnconditionalDFM, the theoretical factor MLP.
No k argument is exposed: upstream latent_dim does not constrain the U-Net.
"""
from dataclasses import asdict, dataclass
import hashlib
from importlib import import_module, util
from pathlib import Path

AUTHOR_COMMIT = '99261df58dbc95315384ce34e3e4bb1a770a35fa'
SOURCE_HASHES = {
    'diffusion_factor_model.py': 'b46377d291426b1e5740be003590ab5dbaaf3a0c51e091335c9aa0502f081e29',
    'attend.py': '90d72c244acb09fa34358be43119e9fc541a7a432c9e6476ca2b8b80d2f89ede',
}


def author_module():
    """Reject an unpinned core before importing it; no implicit downloads."""
    spec = util.find_spec('diffusion_factor_model')
    if spec is None:
        raise ImportError('Install requirements-dfm.txt to use the author backend')
    folder = Path(spec.origin).parent
    for name, expected in SOURCE_HASHES.items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Author source differs from pinned commit: {name}')
    return import_module('diffusion_factor_model.diffusion_factor_model')


@dataclass(frozen=True)
class AuthorDFMConfig:
    image_size: tuple = (32, 64)
    model_dim: int = 256
    timesteps: int = 200
    save_index: int = 20
    epochs: int = 600
    batch_size: int = 32
    seed: int = 3407

    def __post_init__(self):
        values = (*self.image_size, self.model_dim, self.timesteps, self.epochs, self.batch_size)
        if len(self.image_size) != 2 or any(type(v) is not int or v < 1 for v in values):
            raise ValueError('Require positive integer dimensions and training settings')
        if self.model_dim < 4 or self.model_dim % 2:
            raise ValueError('model_dim must be even and >= 4 for time embeddings')
        if type(self.save_index) is not int or not 0 <= self.save_index < self.timesteps:
            raise ValueError('save_index must identify a diffusion transition')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('seed must be an integer in [0, 2**32)')
        if any(v % self.downsample_factor for v in self.image_size):
            raise ValueError('Image dimensions must be divisible by the downsampling factor')

    @property
    def dim_mults(self):
        minimum = min(self.image_size)
        for threshold, multipliers in ((32, (1, 2, 4, 16)), (16, (1, 2, 4, 8)),
                                       (8, (1, 2, 4)), (4, (1, 2))):
            if minimum >= threshold:
                return multipliers
        return (1,)

    @property
    def downsample_factor(self):
        return 2**(len(self.dim_mults) - 1)

    @property
    def d(self):
        return self.image_size[0] * self.image_size[1]

    def metadata(self):
        return dict(config=asdict(self), upstream_commit=AUTHOR_COMMIT,
                    backend='author_unet_ddpm', released_config_defaults=self == AuthorDFMConfig())


def build_author_dfm(config=AuthorDFMConfig()):
    """Use upstream layers, schedule (including cosine s=10), loss and DDPM.

    Overrides are explicit and recorded; small test models are not reproductions.
    Allocating the published d=2048 default requires substantial memory.
    """
    module = author_module()
    network = module.Unet(dim=config.model_dim, channels=1, filter_size=7,
                          dim_mults=config.dim_mults)
    return module.GaussianDiffusion(network, image_size=config.image_size,
        latent_dim=config.d, timesteps=config.timesteps, objective='pred_noise',
        beta_schedule='cosine', auto_normalize=False)
