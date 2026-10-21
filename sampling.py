import math
from numbers import Integral, Real
import torch


def generate_samples(
    model,
    num_samples,
    num_steps=100,
    t_0=None,
    guidance_fn=None,
    generator=None,
    initial_state=None,
    brownian_increments=None,
    return_trajectory=False,
    guidance_scale=1.0,
):
    """Sample the learned reverse OU SDE using Euler–Maruyama.

    Optional initial_state has shape [num_samples, d]. brownian_increments
    has shape [num_steps, num_samples, d] and contains actual dW increments
    (variance dt), not standard normal draws. Both must be finite tensors
    matching the model's device and dtype; no implicit conversion is performed.
    Supplied initial states are detached and copied, and neither supplied tensor
    is modified. Omitted inputs use N(0, I) initialization and sqrt(dt) times
    independent standard normal increments, respectively.
    guidance_fn receives elapsed reverse time s, while the score receives T-s.
    If return_trajectory=True, return [num_samples, num_steps+1, d] including
    the initial and terminal states, instead of just the terminal state.
    guidance_scale is a finite nonnegative multiplier on guidance_fn's drift
    correction. One preserves the original sampler; zero skips guidance_fn.
    It does not modify the initial distribution or diffusion noise.
    """
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype

    T = float(model.T)
    t_0 = float(model.t0 if t_0 is None else t_0)
    if not isinstance(return_trajectory, bool):
        raise ValueError("return_trajectory must be boolean")
    if (isinstance(guidance_scale, bool) or not isinstance(guidance_scale, Real)
            or not math.isfinite(guidance_scale) or guidance_scale < 0):
        raise ValueError("guidance_scale must be a finite nonnegative real number")
    guidance_scale = float(guidance_scale)

    for name, value in (("num_samples", num_samples), ("num_steps", num_steps),
                        ("model.d", model.d)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not dtype.is_floating_point:
        raise ValueError("model parameters must have floating-point dtype")
    if not math.isfinite(T) or not math.isfinite(t_0) or not 0 <= t_0 < T:
        raise ValueError("Require finite times with 0 <= t_0 < T")

    dt = (T - t_0) / num_steps
    sqrt_dt = math.sqrt(dt)

    def validate_tensor(value, shape, name):
        if (not isinstance(value, torch.Tensor) or value.shape != shape
                or value.device != device or value.dtype != dtype
                or not torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite with shape {shape} and match model dtype/device")

    if brownian_increments is not None:
        validate_tensor(brownian_increments, (num_steps, num_samples, model.d), "brownian_increments")
    if initial_state is None:
        x = torch.randn(
            num_samples, model.d,
            device=device, dtype=dtype, generator=generator,
        )
    else:
        validate_tensor(initial_state, (num_samples, model.d), "initial_state")
        x = initial_state.detach().clone()

    path = [x.clone()] if return_trajectory else None

    # Preserve each module's original training/evaluation mode.
    modes = [(module, module.training) for module in model.modules()]

    try:
        model.eval()

        with torch.no_grad():
            for j in range(num_steps):
                s = torch.full(
                    (num_samples, 1), j * dt,
                    device=device, dtype=dtype,
                )

                score = model.factor_score_network(x, T - s)
                drift = 0.5 * x + score

                if guidance_fn is not None and guidance_scale != 0.0:
                    drift = drift + guidance_scale * guidance_fn(x, s)

                if brownian_increments is None:
                    increment = sqrt_dt * torch.randn(
                        x.shape,
                        device=device, dtype=dtype, generator=generator,
                    )
                else:
                    increment = brownian_increments[j]
                x = x + dt * drift + increment

                if not torch.isfinite(x).all():
                    raise ValueError("Sampling produced nonfinite values")
                if path is not None:
                    path.append(x.clone())

        return torch.stack(path, dim=1) if path is not None else x

    finally:
        for module, training in modes:
            module.training = training
