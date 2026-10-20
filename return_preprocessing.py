"""Training-fitted return coordinates shared by the factor and U-Net models.

Splitting and OU population calculations belong to their respective callers.
The author recipe reduces float32 tensors; the NumPy recipe reduces float64
arrays. Choose once when fitting, then reuse the saved statistics everywhere.
"""
import numpy as np
import torch
from torch import nn
from conditioning import portfolio_event_in_model_coordinates


def _real_tensor(value, dtype=None):
    # Copy NumPy views (including negative strides/read-only arrays) safely.
    raw = torch.as_tensor(np.array(value, copy=True) if isinstance(value, np.ndarray) else value)
    if raw.is_complex() or raw.dtype == torch.bool:
        raise ValueError('Returns and preprocessing must be real-valued')
    if dtype is not None:
        raw = raw.to(dtype=dtype) if isinstance(value, (torch.Tensor, np.ndarray)) else torch.as_tensor(value, dtype=dtype)
    return raw


def _assets(assets, dimension):
    if isinstance(assets, (str, bytes)):
        raise ValueError('Supply an ordered sequence of asset names')
    names = tuple(assets)
    if (len(names) != dimension or any(not isinstance(a, str) or not a.strip() for a in names)
            or len(set(names)) != dimension):
        raise ValueError('Asset names must be nonempty, unique and match the return columns')
    return names


class ReturnTransform(nn.Module):
    """Saved per-asset statistics; tensor inputs/outputs use the buffer dtype/device.

    Pass assets when input column metadata is available: a mismatch is rejected,
    never silently reordered. Unlabelled arrays must already use self.assets.
    """
    def __init__(self, mean, std, assets):
        super().__init__()
        mean, std = (_real_tensor(v).detach().clone() for v in (mean, std))
        if (mean.ndim != 1 or not mean.numel() or std.shape != mean.shape
                or mean.dtype not in (torch.float32, torch.float64)
                or std.dtype != mean.dtype or mean.device != std.device
                or not torch.isfinite(mean).all() or not torch.isfinite(std).all()
                or not (std > 0).all()):
            raise ValueError('Require finite float32/float64 mean/std vectors and positive std')
        self.assets = _assets(assets, len(mean))
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    @classmethod
    def fit(cls, training_returns, assets, *, precision='author'):
        """Fit training rows only; author=float32 torch, numpy=float64 NumPy.

        Both use sample standard deviation (ddof=1). No split is inferred.
        """
        if precision not in ('author', 'numpy'):
            raise ValueError('precision must be author or numpy')
        dtype = torch.float32 if precision == 'author' else torch.float64
        data = _real_tensor(training_returns, dtype=dtype).detach().cpu()
        if data.ndim != 2 or len(data) < 2 or not data.shape[1] or not torch.isfinite(data).all():
            raise ValueError('Returns must be finite with shape [N>=2, d>=1]')
        if precision == 'numpy':
            array = data.numpy()
            mean, std = array.mean(axis=0), array.std(axis=0, ddof=1)
        else:
            # Exactly the released train.py order: cast before both reductions.
            mean, std = data.mean(0), data.std(0, correction=1)
        return cls(mean, std, assets)

    def check_assets(self, assets):
        if _assets(assets, len(self.mean)) != self.assets:
            raise ValueError('Asset order must match the fitted transformation')

    def _data(self, value, assets):
        if assets is not None:
            self.check_assets(assets)
        data = _real_tensor(value, dtype=self.mean.dtype).to(device=self.mean.device)
        if (data.ndim not in (1, 2) or data.numel() == 0 or data.shape[-1] != len(self.mean)
                or not torch.isfinite(data).all()):
            raise ValueError('Expected finite returns [d] or [N,d] in the saved asset order')
        return data

    def forward(self, returns, *, assets=None):
        return (self._data(returns, assets) - self.mean) / self.std

    transform = forward

    def inverse(self, standardized, *, assets=None):
        return self._data(standardized, assets) * self.std + self.mean

    def event(self, weights, threshold, *, assets=None):
        if assets is not None:
            self.check_assets(assets)
        return portfolio_event_in_model_coordinates(weights, threshold, self.mean, self.std)


class NumpyStandardizer:
    """NumPy facade over the same fitted ReturnTransform, including float32 fits.

    Use from_transform to compare both models without independently fitting.
    mean/std properties are copies, so callers cannot mutate saved buffers.
    """
    def __init__(self, mean, std, assets=None):
        dimension = len(mean)
        names = tuple(str(i) for i in range(dimension)) if assets is None else assets
        # Validate before a cast could erase bool/complex values.
        values = [_real_tensor(v, dtype=torch.float64) for v in (mean, std)]
        self.tensor_transform = ReturnTransform(*values, names)

    @classmethod
    def from_transform(cls, transform):
        if not isinstance(transform, ReturnTransform):
            raise TypeError('Expected ReturnTransform')
        result = cls.__new__(cls)
        result.tensor_transform = transform
        return result

    @classmethod
    def fit(cls, training_returns, assets=None):
        raw = _real_tensor(training_returns)
        if raw.ndim != 2:
            raise ValueError('Training returns must have shape [N,d]')
        names = tuple(str(i) for i in range(raw.shape[1])) if assets is None else assets
        return cls.from_transform(ReturnTransform.fit(training_returns, names, precision='numpy'))

    @property
    def mean(self):
        return self.tensor_transform.mean.detach().cpu().numpy().copy()

    @property
    def std(self):
        return self.tensor_transform.std.detach().cpu().numpy().copy()

    @property
    def assets(self):
        return self.tensor_transform.assets

    def transform(self, raw, *, assets=None):
        return self.tensor_transform(raw, assets=assets).detach().cpu().numpy()

    def inverse(self, model, *, assets=None):
        return self.tensor_transform.inverse(model, assets=assets).detach().cpu().numpy()

    def event(self, weights, threshold, *, assets=None):
        return self.tensor_transform.event(weights, threshold, assets=assets)
