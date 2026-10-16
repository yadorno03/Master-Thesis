"""Load the equity notebook's DFM checkpoint as a frozen conditional reference."""

import numpy as np
import torch

from conditioning import portfolio_event_in_model_coordinates
from DFM import UnconditionalDFM
from reference import FrozenDFMReference


def load_reference_checkpoint(path):
    """Return (reference, ordered assets, training mean, training std) on CPU.

    Restore saved float32/float64 weights and explicit architecture/horizon.
    Scaling is reused, never fitted here; its training-only provenance must be
    established by the producer. Loading does not advance the CPU RNG stream.
    """
    saved = torch.load(path, map_location='cpu', weights_only=True)
    config, state = saved['model_config'], saved['state_dict']
    required = {'d', 'k', 'sigma_max', 'T', 't0', 'hidden_dims'}
    if not required.issubset(config):
        raise ValueError('Checkpoint must specify the full architecture and horizon')
    loading = state['factor_score_network.V']
    if not isinstance(loading, torch.Tensor) or loading.dtype not in (torch.float32, torch.float64):
        raise ValueError('Checkpoint precision must be float32 or float64')
    dtype = loading.dtype
    if any(not isinstance(value, torch.Tensor) or value.dtype != dtype
           or not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Checkpoint weights must be finite and share one precision')
    with torch.random.fork_rng(devices=[]):
        model = UnconditionalDFM(**config).to(device='cpu', dtype=dtype)
        model.load_state_dict(state, strict=True)
    reference = FrozenDFMReference(model)
    assets = saved['assets']
    if (not isinstance(assets, (list, tuple)) or len(assets) != reference.d
            or not all(isinstance(asset, str) and asset.strip() for asset in assets)
            or len(set(assets)) != reference.d):
        raise ValueError('Checkpoint must have unique nonempty ordered asset identifiers')
    mean, std = saved['preprocessing']['mean'], saved['preprocessing']['std']
    # Validate the raw values before conversion can hide boolean/complex inputs.
    portfolio_event_in_model_coordinates(np.ones(reference.d) / reference.d, 0., mean, std)
    return reference, list(assets), np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
