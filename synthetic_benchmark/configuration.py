"""Experiment presets, named random streams, and JSON serialization."""
from dataclasses import dataclass, replace
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
import json

import numpy as np
import torch

from NeuralNets import SCORE_NETWORKS
from synthetic_conditional_truth import validate_dimensions

@dataclass
class Config:
    mode: str = 'standard'
    d: int = 100
    k: int = 8
    seeds: tuple = (42,)
    cases: tuple = ('gaussian',)
    n_train: int = 12000
    n_val: int = 3000
    n_eval: int = 3000
    dfm_epochs: int = 1000
    steps: int = 400
    T: float = 8.0
    t0: float = 0.005
    width: int = 64
    batch: int = 256
    lr: float = 0.002
    dfm_time_sampling: str = 'uniform'
    # 'per_asset': the DFM models returns standardized by their training mean and standard deviation.
    dfm_standardization: str = 'none'
    dfm_initialization: str = 'random'
    dfm_loading_lr_scale: float = 1.0
    dfm_loading_warmup_epochs: int = 0
    # 'mlp': the score network of Chen et al. (2025); 'gaussian_mlp': a Gaussian factor part plus that MLP;
    # 'regime_mlp': a scale mixture of Gaussians (volatility regimes) plus that MLP.
    dfm_network: str = 'mlp'
    # 'regime_mlp' only: numbers of factor and residual scales.
    dfm_regimes: tuple = (4, 3)
    # False trains the MLP g only: V, c and the analytic factor law (the Gaussian part's mean and covariance,
    # the regime scales and table) keep their initial values (PCA, moments, maximum likelihood).
    dfm_train_analytic: bool = True
    # Noise draws per validation row, fixed across epochs.
    dfm_validation_draws: int = 1
    # Decay of the exponential moving average of the weights that is validated and saved; 0 disables it.
    dfm_ema_decay: float = 0.0
    # Knots (in log time) of the linear refit of g after training, synthetic_benchmark.unconditional.
    # refit_linear_part; 0 trains without it.
    dfm_refit_knots: int = 0
    torch_threads: int = 2
    device: str = 'cpu'

    def __post_init__(self):
        validate_dimensions(self.d, self.k)
        if self.cases != ('gaussian',):
            raise ValueError("Only the Gaussian population is supported")
        if self.dfm_time_sampling not in ('uniform', 'log_importance', 'log_balanced'):
            raise ValueError('Unknown DFM time sampling setup')
        if self.dfm_standardization not in ('none', 'per_asset'):
            raise ValueError('DFM standardization must be none or per_asset')
        if self.dfm_initialization not in ('random', 'pca'):
            raise ValueError('DFM initialization must be random or pca')
        if not np.isfinite(self.dfm_loading_lr_scale) or self.dfm_loading_lr_scale <= 0:
            raise ValueError('DFM loading learning-rate scale must be positive and finite')
        if (isinstance(self.dfm_loading_warmup_epochs, bool)
                or not isinstance(self.dfm_loading_warmup_epochs, Integral)
                or self.dfm_loading_warmup_epochs < 0):
            raise ValueError('DFM loading warmup must be a nonnegative integer')
        if self.dfm_network not in SCORE_NETWORKS:
            raise ValueError(f'dfm_network must be one of {sorted(SCORE_NETWORKS)}')
        if (len(self.dfm_regimes) != 2 or any(isinstance(count, bool) or not isinstance(count, Integral)
                                               or count < 1 for count in self.dfm_regimes)):
            raise ValueError('dfm_regimes must be two positive integers')
        if (isinstance(self.dfm_validation_draws, bool) or not isinstance(self.dfm_validation_draws, Integral)
                or self.dfm_validation_draws < 1):
            raise ValueError('dfm_validation_draws must be a positive integer')
        if not isinstance(self.dfm_train_analytic, bool):
            raise ValueError('dfm_train_analytic must be a boolean')
        if not 0 <= self.dfm_ema_decay < 1:
            raise ValueError('dfm_ema_decay must be in [0, 1)')
        if (isinstance(self.dfm_refit_knots, bool) or not isinstance(self.dfm_refit_knots, Integral)
                or self.dfm_refit_knots < 0 or self.dfm_refit_knots == 1):
            raise ValueError('dfm_refit_knots must be 0 (no refit) or an integer of at least 2')
        if (isinstance(self.torch_threads, bool) or not isinstance(self.torch_threads, Integral)
                or self.torch_threads < 1):
            raise ValueError('torch_threads must be a positive integer')

    @classmethod
    def preset(cls, mode):
        if mode == 'smoke':
            return cls(mode=mode, d=12, k=8, n_train=2048, n_val=512, n_eval=512, dfm_epochs=15, steps=100, width=32)
        if mode == 'standard':
            return cls()
        raise ValueError(mode)


# The DFMs of the market notebook and their training recipe; the synthetic notebook trains DFM (MLP) and
# DFM (Gaussian + MLP) from Config.preset, with every parameter trained and no refit. Score matching barely
# constrains the factor variances, so the networks with an analytic factor part keep it at its fit to the
# training rows, train their MLP only and then refit the linear part of g; DFM (MLP), the network of
# Chen et al. (2025), trains every parameter and has no refit, which would make its factor variances overshoot.
DFM_NETWORKS = ('mlp', 'gaussian_mlp', 'regime_mlp')
DFM_RECIPE = dict(k=8, dfm_standardization='per_asset', dfm_initialization='pca', dfm_time_sampling='log_balanced',
                  dfm_loading_lr_scale=.1, dfm_loading_warmup_epochs=100, lr=1e-3, dfm_regimes=(4, 3),
                  # Eight fixed noise draws per validation row and averaged weights: the validation loss is
                  # flat near its minimum, so the selected epoch should not follow its noise.
                  dfm_validation_draws=8, dfm_ema_decay=.999, torch_threads=4)
NETWORK_SETTINGS = {network: dict(dfm_train_analytic=False, dfm_refit_knots=12)
                    for network in ('gaussian_mlp', 'regime_mlp')}


def notebook_config(mode='standard', **overrides):
    """The DFM configuration of the market notebook: d=500, k=8, 2,500 training rows; 'smoke' reduces every budget."""
    if mode == 'smoke':
        sizes = dict(d=12, n_train=2_048, n_val=512, n_eval=512, dfm_epochs=15, steps=100, width=32)
    elif mode == 'standard':
        sizes = dict(d=500, n_train=2_500, n_val=750, n_eval=3_000, dfm_epochs=1_500)
    else:
        raise ValueError(mode)
    return replace(Config(mode=mode, **DFM_RECIPE, **sizes), **overrides)


def network_config(config, network, settings=None):
    """The configuration of one DFM: ``config`` with ``network`` and its settings (default NETWORK_SETTINGS)."""
    settings = NETWORK_SETTINGS if settings is None else settings
    return replace(config, dfm_network=network, **settings.get(network, {}))


class Streams:
    """Named SeedSequence streams; labels cannot depend on call order."""
    # New names go last: a stream's seed depends on its position only.
    names = ('train', 'validation', 'dfm_noise', 'dfm_validation_noise', 'model_init', 'truth', 'truth2', 'generated',
             'dfm_refit')

    def __init__(self, seed):
        self.seeds = {n: int(s.generate_state(1)[0]) for n, s in zip(self.names, np.random.SeedSequence(seed).spawn(len(self.names)))}

    def np(self, n):
        return np.random.default_rng(self.seeds[n])

    def torch(self, n, device='cpu'):
        return torch.Generator(device=device).manual_seed(self.seeds[n])


def validate_h_seeds(seeds, names=('train', 'validation', 'test', 'shuffle')):
    """Copy the requested H stream seeds, rejecting reused random streams.

    PyTorch's CPU generator can repeat draws for seeds differing by 2**32, so
    distinct full integers alone do not establish separate trajectory streams.
    Callers sampling only some splits can supply just those names.
    """
    if not isinstance(seeds, Mapping):
        raise ValueError('H stream seeds must be a mapping from names to integers')
    names = tuple(names)
    if not names or len(set(names)) != len(names):
        raise ValueError('H stream names must be nonempty and distinct')
    missing = [name for name in names if name not in seeds]
    if missing:
        raise ValueError(f'Missing H stream seeds: {", ".join(missing)}')
    result = {}
    seen = {}
    for name in names:
        seed = seeds[name]
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**63:
            raise ValueError(f'H stream seed {name!r} must be an integer in [0, 2**63)')
        seed = int(seed)
        key = seed % 2**32
        if key in seen:
            raise ValueError('H streams require distinct seeds, including their lower 32 bits: '
                             f'{seen[key]} and {name} reuse the same random stream')
        seen[key] = name
        result[name] = seed
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))


@dataclass(frozen=True)
class ConditionalAssessmentConfig:
    """Bounded evaluation budgets, separate from training and existing streams.

    Caps are per rejection stream, not a presumed population event probability.
    Changing batching changes random-draw order; reproduce using all settings.
    """
    rejection_n: int = 1000
    rejection_batch_size: int = 256
    rejection_max_proposals: int = 50000
    rejection_reference_max_proposals: int = 50000
    rejection_seed: int = 41003
    rejection_reference_seed: int = 41004
    truth_reference_seed: int = 41005
    projection_seed: int = 41006
    matching_seed: int = 41007
    projections: int = 64
    # Noise levels of the readable distance table: pairs of exact samples, at the table's sample size.
    noise_seed: int = 41008
    noise_replicates: int = 100

    def __post_init__(self):
        for name in ('rejection_n', 'rejection_batch_size', 'rejection_max_proposals',
                     'rejection_reference_max_proposals', 'projections', 'noise_replicates'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        seeds = [getattr(self, name) for name in (
            'rejection_seed', 'rejection_reference_seed', 'truth_reference_seed',
            'projection_seed', 'matching_seed', 'noise_seed')]
        if any(isinstance(s, bool) or not isinstance(s, Integral) or not 0 <= s < 2**63 for s in seeds):
            raise ValueError('Assessment seeds must be integers in [0, 2**63)')
        # The supported CPU PyTorch generator aliases seeds differing by 2**32.
        if len({int(seed) % 2**32 for seed in seeds}) != len(seeds):
            raise ValueError('Assessment streams require distinct seeds, including their lower 32 bits')


def rejection_assessment(tail, rejection_n, *, cap_factor=4, **settings):
    """Assessment settings whose proposal caps are cap_factor times the proposals expected at this tail.

    A DFM rejection stream needs about rejection_n / tail proposals; the cap only
    stops a badly calibrated DFM from running indefinitely.
    """
    if isinstance(tail, bool) or not isinstance(tail, Real) or not 0 < tail < 1:
        raise ValueError('tail must be a probability in (0, 1)')
    if isinstance(cap_factor, bool) or not isinstance(cap_factor, Real) or cap_factor < 1:
        raise ValueError('cap_factor must be at least 1')
    cap = int(np.ceil(cap_factor * rejection_n / tail))
    return ConditionalAssessmentConfig(rejection_n=rejection_n, rejection_max_proposals=cap,
                                       rejection_reference_max_proposals=cap, **settings)
