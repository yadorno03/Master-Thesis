import torch
import numpy as np
from checkers import checked_output, guidance_inputs, validate_projection


class HGuidance:
    """Differentiate a positive H model, or a log-H model for rare events.

    Returns the supplied H-function's logarithmic gradient. Batch-coupled models
    (including training-mode BatchNorm) do not satisfy this adapter's pointwise
    contract.
    """

    def __init__(self, h_model, *, B=None, output_is_log=False):
        if not callable(h_model):
            raise ValueError("h_model must be callable")
        if not isinstance(output_is_log, bool):
            raise ValueError("output_is_log must be boolean")
        self.h_model = h_model
        self.B = None if B is None else validate_projection(
            B, len(B), device="cpu", dtype=torch.float64)
        self.output_is_log = output_is_log

    def _features(self, state):
        basis = None if self.B is None else validate_projection(
            self.B, state.shape[1], device=state.device, dtype=state.dtype)
        return (state if basis is None else state @ basis), basis

    def __call__(self, state, time):
        time = guidance_inputs(state, time)
        if torch.is_inference_mode_enabled():
            raise ValueError("CDG-ML needs autograd; use torch.no_grad(), not inference_mode()")
        with torch.enable_grad():
            features, basis = self._features(state.detach())
            features = features.detach().requires_grad_(True)
            h = checked_output(self.h_model(features, time.detach()), time, "H output")
            if not self.output_is_log and (h <= 0).any():
                raise ValueError("H must be strictly positive; use log-H for rare events")
            log_h = h if self.output_is_log else h.log()
            if not log_h.requires_grad:
                raise ValueError("H output is disconnected from the state; check the model's gradient graph")
            try:
                gradient = torch.autograd.grad(log_h.sum(), features, allow_unused=False)[0]
            except RuntimeError as exc:
                raise ValueError("H output must be differentiably connected to the state") from exc
            if basis is not None:
                gradient = gradient @ basis.T
        return checked_output(gradient.detach(), state, "guidance")


class QGuidance(HGuidance):
    """Return Q/H for Q=grad H; Q is not a drift or diffusion-scaled target."""

    def __init__(self, h_model, q_model, *, B=None):
        super().__init__(h_model, B=B)
        if not callable(q_model):
            raise ValueError("q_model must be callable")
        self.q_model = q_model

    def __call__(self, state, time):
        time = guidance_inputs(state, time)
        with torch.no_grad():
            features, basis = self._features(state)
            h = checked_output(self.h_model(features, time), time, "H output")
            if (h <= 0).any():
                raise ValueError("H must be strictly positive; Q/H is undefined otherwise")
            q = checked_output(self.q_model(features, time), features, "Q output")
            gradient = q / h
            if basis is not None:
                gradient = gradient @ basis.T
            return checked_output(gradient, state, "guidance").detach()
