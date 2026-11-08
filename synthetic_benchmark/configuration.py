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

# Local output of the market data cleaning step (2,516 training days of 800 stocks); not in git.
MARKET_PANEL = 'artifacts/market_data_cleaning/20261029T104817954744Z/train_complete.npz'

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
    dfm_initialization: str = 'random'
    dfm_loading_lr_scale: float = 1.0
    dfm_loading_warmup_epochs: int = 0
    # 'chen': the population of Chen et al. (2025); 'market': calibrated on real stocks.
    population: str = 'chen'
    market_panel: str = MARKET_PANEL
    market_cohort_seed: int = 500
    # 'factor': the score network of Chen et al. (2025); 'gaussian_residual': I3.
    dfm_network: str = 'factor'
    torch_threads: int = 2
    device: str = 'cpu'

    def __post_init__(self):
        validate_dimensions(self.d, self.k)
        if self.cases != ('gaussian',):
            raise ValueError("This branch supports only the Gaussian simulation")
        if self.dfm_time_sampling not in ('uniform', 'log_importance', 'log_balanced'):
            raise ValueError('Unknown DFM time sampling setup')
        if self.dfm_initialization not in ('random', 'pca'):
            raise ValueError('DFM initialization must be random or pca')
        if not np.isfinite(self.dfm_loading_lr_scale) or self.dfm_loading_lr_scale <= 0:
            raise ValueError('DFM loading learning-rate scale must be positive and finite')
        if (isinstance(self.dfm_loading_warmup_epochs, bool)
                or not isinstance(self.dfm_loading_warmup_epochs, Integral)
                or self.dfm_loading_warmup_epochs < 0):
            raise ValueError('DFM loading warmup must be a nonnegative integer')
        if self.population not in ('chen', 'market'):
            raise ValueError("population must be 'chen' or 'market'")
        if self.dfm_network not in SCORE_NETWORKS:
            raise ValueError(f'dfm_network must be one of {sorted(SCORE_NETWORKS)}')
        for name in ('market_cohort_seed', 'torch_threads'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < (name == 'torch_threads'):
                raise ValueError(f'{name} must be a {"positive" if name == "torch_threads" else "nonnegative"} integer')

    @classmethod
    def preset(cls, mode):
        if mode == 'smoke':
            return cls(mode=mode, d=12, k=8, n_train=2048, n_val=512, n_eval=512, dfm_epochs=15, steps=100, width=32)
        if mode == 'standard':
            return cls()
        if mode == 'research':
            return cls(mode=mode, seeds=(42, 314, 2718), n_train=40000, n_val=8000, n_eval=10000, dfm_epochs=1000, steps=1000, width=128)
        if mode == 'large':
            return replace(cls.preset('research'), mode=mode, d=100)
        raise ValueError(mode)

class Streams:
    """Named SeedSequence streams; labels cannot depend on call order."""
    names = ('train', 'validation', 'dfm_noise', 'dfm_validation_noise', 'model_init', 'truth', 'truth2', 'generated')

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


def validate_branch_seeds(branch_seeds, base_seeds, names=('train', 'validation', 'test')):
    """Copy named continuation-branch seeds, separate from every base H stream.

    validate_h_seeds returns only the names it is asked for, so branch seeds
    would be dropped silently if mixed into that mapping. Here the branch streams
    for the requested splits must differ from each other and from all supplied
    base/shuffle seeds, including in their lower 32 bits (CPU generator aliasing).
    Returns {split: seed}; the caller still owns generation of the generators.
    """
    if not isinstance(branch_seeds, Mapping) or not isinstance(base_seeds, Mapping):
        raise ValueError('Branch and base seeds must be mappings from names to integers')
    names = tuple(names)
    if not names or len(set(names)) != len(names):
        raise ValueError('Branch stream names must be nonempty and distinct')
    missing = [name for name in names if name not in branch_seeds]
    if missing:
        raise ValueError(f'Missing branch stream seeds: {", ".join(missing)}')
    taken = {}
    for name, seed in base_seeds.items():
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**63:
            raise ValueError(f'H stream seed {name!r} must be an integer in [0, 2**63)')
        taken[int(seed) % 2**32] = f'base:{name}'
    result = {}
    for name in names:
        seed = branch_seeds[name]
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**63:
            raise ValueError(f'Branch seed {name!r} must be an integer in [0, 2**63)')
        key = int(seed) % 2**32
        if key in taken:
            raise ValueError('Branch streams require seeds distinct from all base streams and each '
                             f'other, including their lower 32 bits: branch:{name} reuses {taken[key]}')
        taken[key] = f'branch:{name}'
        result[name] = int(seed)
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

    def __post_init__(self):
        for name in ('rejection_n', 'rejection_batch_size', 'rejection_max_proposals',
                     'rejection_reference_max_proposals', 'projections'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        seeds = [getattr(self, name) for name in (
            'rejection_seed', 'rejection_reference_seed', 'truth_reference_seed',
            'projection_seed', 'matching_seed')]
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
