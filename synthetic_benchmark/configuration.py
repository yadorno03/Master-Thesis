"""Experiment presets, named random streams, and JSON serialization."""
from dataclasses import dataclass, replace
from pathlib import Path
import json

import numpy as np
import torch

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
    device: str = 'cpu'

    def __post_init__(self):
        validate_dimensions(self.d, self.k)
        if self.cases != ('gaussian',):
            raise ValueError("This branch supports only the Gaussian simulation")

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
        from numbers import Integral
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
        if len(set(seeds)) != len(seeds):
            raise ValueError('Assessment streams require distinct seeds')
