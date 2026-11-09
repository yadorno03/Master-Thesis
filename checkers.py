import torch


def _parameter(value, name):
    try:
        tensor = torch.as_tensor(value)
        if tensor.is_complex() or tensor.dtype == torch.bool:
            raise ValueError(f"{name} must be real-valued")
        tensor = torch.as_tensor(value, device="cpu", dtype=torch.float64).detach().clone()
    except (TypeError, RuntimeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real numeric tensor") from exc
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite")
    return tensor

def validate_projection(B, d, *, device, dtype):
    """Return finite orthonormal ``[d, k]`` columns in the requested precision."""
    if not isinstance(d, int) or isinstance(d, bool) or d < 1:
        raise ValueError("d must be a positive integer")
    if not dtype.is_floating_point:
        raise ValueError("projection dtype must be floating point")
    basis = _parameter(B, "B").to(device=device, dtype=dtype)
    if basis.ndim != 2 or basis.shape[0] != d or not 1 <= basis.shape[1] <= d:
        raise ValueError("B must have shape [d, k] with 1 <= k <= d")
    tolerance = 1e-6 if dtype == torch.float64 else max(1e-5, 10 * torch.finfo(dtype).eps)
    gram = basis.T @ basis
    identity = torch.eye(basis.shape[1], device=device, dtype=dtype)
    if not torch.isfinite(basis).all() or not torch.allclose(
        gram, identity, atol=tolerance, rtol=0
    ):
        raise ValueError("B must have orthonormal columns")
    return basis


def checked_output(value, reference, name):
    if (not isinstance(value, torch.Tensor) or value.shape != reference.shape
            or value.dtype != reference.dtype or value.device != reference.device
            or not torch.isfinite(value).all()):
        raise ValueError(f"{name} must match shape, dtype, and device and be finite")
    return value
