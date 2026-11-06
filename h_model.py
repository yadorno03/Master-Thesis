"""Smooth terminal-event probability network and trajectory regression loss."""

from dataclasses import dataclass
import json
import math
import time
from numbers import Integral, Real

import torch
from torch import nn
from torch.nn import functional as F

from conditioning import ConditioningConfig
from DFM import UnconditionalDFM
from reference import FrozenDFMReference
from sampling import generate_samples


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


SOFT_TARGET_MODES = ('off', 'reuse_final', 'continuation')


def _check_soft_target_options(mode, window, use_last_step_targets):
    """Validate target options without touching data, optimizers or random streams.

    'off' keeps the legacy use_last_step_targets semantics and takes no window.
    Both new modes soften a late-time window of rows with final-step
    probabilities, so they require use_last_step_targets=True.
    """
    if mode not in SOFT_TARGET_MODES:
        raise ValueError(f'soft_target_mode must be one of {SOFT_TARGET_MODES}')
    if mode == 'off':
        if window is not None:
            raise ValueError("soft_target_window requires soft_target_mode other than 'off'")
        return
    if not use_last_step_targets:
        raise ValueError(f"soft_target_mode={mode!r} requires use_last_step_targets=True")
    if (isinstance(window, bool) or not isinstance(window, Real)
            or not math.isfinite(window) or window <= 0):
        raise ValueError('soft_target_window must be a finite positive remaining time')


def soft_target_indices(times, horizon, window):
    """Preterminal rows in the remaining-time window, always including the last.

    Row j is selected when horizon - times[j] <= window, with a tolerance of
    four machine epsilons of the stored precision times the horizon, so a
    cutoff that lies exactly on a grid time is included despite rounding. The
    rule depends only on time, never on a label or predicted probability.
    Returns increasing LongTensor indices on times' device.
    """
    if times.ndim != 1 or len(times) == 0:
        raise ValueError('times must be a nonempty vector')
    remaining = horizon - times.double()
    tolerance = 4 * torch.finfo(times.dtype).eps * horizon
    selected = remaining <= window + tolerance
    selected[-1] = True
    return selected.nonzero().flatten()


@dataclass
class TerminalLabelData:
    """Whole trajectories: features [N,J,k+1], binary labels [N,1].

    times and weights have shape [J]. Only preterminal times are stored;
    weights[j] = delta_t[j] / (T-t0). Split by trajectory, never by time row.
    """

    times: torch.Tensor
    features: torch.Tensor
    labels: torch.Tensor
    weights: torch.Tensor
    last_probabilities: torch.Tensor | None = None
    last_states: torch.Tensor | None = None
    # Continuation mode only: unguided branch-average final-step probabilities
    # at the selected late rows [N,K,1] and their standard errors (None if M=1).
    soft_target_indices: torch.Tensor | None = None
    soft_targets: torch.Tensor | None = None
    soft_target_standard_errors: torch.Tensor | None = None


class HNetwork(nn.Module):
    """Bind a frozen DFM and full portfolio event to a smooth H network.

    All times are generative in [0, T-t0]. Scalar times produce unbatched
    ell/G/A; [batch] or [batch, 1] times produce batched coefficients.
    features and mu take and return row batches of all (Z,U) coordinates.
    G may be rank deficient; no orthonormality or covariance inverse is assumed.
    Only network parameters are trainable; the reference stays frozen.
    """

    def __init__(self, reference, config, hidden_dims=(32, 32), *, log_remaining_time=False):
        super().__init__()
        if not isinstance(reference, FrozenDFMReference):
            raise TypeError("reference must be a FrozenDFMReference")
        if not isinstance(config, ConditioningConfig):
            raise TypeError("config must be a ConditioningConfig")
        if len(config.w) != reference.d:
            raise ValueError("w must contain one weight per asset")
        self.reference = reference
        self._config = config
        parameter = next(reference.parameters())
        self.register_buffer("_w", torch.tensor(
            config.w, device=parameter.device, dtype=parameter.dtype))
        if not torch.isfinite(self._w).all() or not (self._w != 0).any():
            raise ValueError("w must remain finite and nonzero in reference precision")
        if not torch.isfinite(torch.tensor(config.C, dtype=parameter.dtype)):
            raise ValueError("C must remain finite in reference precision")

        self.hidden_dims = tuple(hidden_dims)
        self.log_remaining_time = bool(log_remaining_time)
        dimension_y = reference._score.k + 1
        self.register_buffer('normalization_times', self._w.new_empty(0))
        self.register_buffer('normalization_mean', self._w.new_empty((0, dimension_y)))
        self.register_buffer('normalization_scale', self._w.new_empty((0, dimension_y)))
        if not self.hidden_dims:
            raise ValueError("At least one smooth hidden layer is required")
        layers = []
        dimension = reference._score.k + 2 + int(self.log_remaining_time)
        for width in self.hidden_dims:
            _positive_int(width, "hidden width")
            layers.extend((nn.Linear(dimension, width), nn.Tanh()))
            dimension = width
        layers.append(nn.Linear(dimension, 1))
        self.network = nn.Sequential(*layers).to(device=parameter.device, dtype=parameter.dtype)

    @property
    def config(self):
        return self._config

    @property
    def w(self):
        """Return a copy so callers cannot mutate the configured portfolio."""
        return self._w.clone()

    @property
    def C(self):
        return self.config.C

    def G(self, t):
        """G_t: [k+1, d] or [batch, k+1, d], retaining the complete U row."""
        return self.reference.G(t, self._w)

    def ell(self, t):
        """ell_t = (alpha_(T-t)/alpha_t0) D_(T-t) Lambda_t0 w."""
        return self.G(t)[..., -1, :]

    def _batch(self, t, state, dimension):
        self.reference._check_tensor(state, "state")
        if state.ndim != 2 or state.shape[0] == 0 or state.shape[1] != dimension:
            raise ValueError(f"state must have nonempty shape [batch, {dimension}]")
        t = self.reference._time(t, 0, self.reference.T - self.reference.t0)
        if t.shape[0] == 1:
            t = t.expand(state.shape[0], 1)
        if t.shape[0] != state.shape[0]:
            raise ValueError("Time batch must match state batch")
        return t

    def features(self, t, x):
        """Y=(Z,U)=G_t X, returned as [batch, k+1]."""
        t = self._batch(t, x, self.reference.d)
        return torch.bmm(self.G(t), x.unsqueeze(-1)).squeeze(-1)

    def mu(self, t, y):
        """Explicit closed reduced drift (Rework.pdf, equation 2.5).

        The U drift has no linear U term, but the U coordinate is retained.
        No ambient state reconstruction or separate drift network is used.
        """
        network = self.reference._score
        t = self._batch(t, y, network.k + 1)
        s, alpha, D, lambda0 = self.reference._schedule(t)
        z = y[:, :network.k]
        n = network.g_zeta(torch.cat((z, s), dim=-1))
        D2Bn = D.square() * (n @ network.V.T)
        mu_z = -0.5 * z + alpha * (D2Bn @ network.V)
        mu_u = (alpha.square() / math.exp(-self.reference.t0 / 2)
                * (D2Bn * lambda0 * self._w).sum(dim=-1, keepdim=True))
        return torch.cat((mu_z, mu_u), dim=-1)

    def A(self, t):
        """Reduced noise covariance G_t G_t^T; may be singular."""
        G = self.G(t)
        return G @ G.transpose(-1, -2)

    def terminal_event(self, y):
        """Evaluate U <= C on terminal reduced states [batch, k+1]."""
        self._batch(self.reference.T - self.reference.t0, y,
                    self.reference._score.k + 1)
        return y[:, -1] <= self.C

    @property
    def horizon(self):
        return self.reference.T - self.reference.t0

    def _logits(self, t, y):
        t = self._batch(t, y, self.reference._score.k + 1)
        return self.network(self.network_inputs(t, y))

    @torch.no_grad()
    def fit_normalization(self, data):
        """Fit coordinate-wise moments at each time using training paths only."""
        _validate_data(self, data)
        self.normalization_times = data.times.detach().clone()
        self.normalization_mean = data.features.mean(0)
        # A tiny absolute floor handles deterministic coordinates without undoing
        # the scaling of the small portfolio coordinate.
        self.normalization_scale = data.features.std(0, unbiased=False).clamp_min(1e-6)

    def network_inputs(self, t, y):
        """Standardize inside H, so guidance retains the spatial chain rule.

        Interpolate moments between training times; clamp outside that grid.
        Y=G_t X itself, and the reference dynamics, are never modified.
        """
        t = self._batch(t, y, self.reference._score.k + 1)
        grid = self.normalization_times
        if grid.numel():
            hi = torch.searchsorted(grid, t[:, 0].contiguous()).clamp(0, len(grid)-1)
            lo = (hi-1).clamp_min(0)
            span = (grid[hi]-grid[lo]).clamp_min(torch.finfo(y.dtype).eps)
            fraction = ((t[:, 0]-grid[lo])/span).clamp(0, 1)[:, None]
            mean = self.normalization_mean[lo].lerp(self.normalization_mean[hi], fraction)
            scale = self.normalization_scale[lo].lerp(self.normalization_scale[hi], fraction)
            y = (y-mean)/scale
        inputs = [y, t/self.horizon]
        if self.log_remaining_time:
            floor = self.horizon-grid[-1] if grid.numel() else self.horizon*1e-4
            inputs.append(torch.log((self.horizon-t).clamp_min(floor)/self.horizon))
        return torch.cat(inputs, dim=-1)

    def exact_last_step_probability(self, x, dt, *, log=False):
        """Exact event probability of one final ambient Euler step, not of the SDE.

        Keep x differentiable for independent checks of guidance gradients. The
        reduced coordinates need not suffice for the ambient Euler transition.
        """
        if not math.isfinite(dt) or not 0 < dt <= self.horizon:
            raise ValueError('dt must lie in (0, horizon]')
        t = self.horizon-dt
        self._batch(t, x, self.reference.d)
        drift = .5*x + self.reference.score(x, self.reference.T-t)
        z = (self.C-(x+dt*drift) @ self._w)/(math.sqrt(dt)*self._w.norm())
        return (torch.special.log_ndtr(z) if log else torch.special.ndtr(z))[:, None]

    def forward(self, t, y):
        return torch.sigmoid(self._logits(t, y))

    def log_prob(self, t, y):
        return F.logsigmoid(self._logits(t, y))

    @torch.no_grad()
    def sample_trajectories(self, num_trajectories, num_steps=100, *, generator=None,
                            batch_size=None, soft_target_mode='off', soft_target_window=None,
                            branch_count=None, branch_generator=None, branch_batch_size=None,
                            diagnostics=None):
        """Collect unguided Gaussian-start Euler-Maruyama paths and terminal labels.

        Labels are evaluated at T-t0; that endpoint is excluded from regression
        inputs. This learns the discretized reference event probability.
        Optional batch_size bounds ambient path memory; only reduced features
        are retained across batches. Seeds reproduce a fixed batching setup.

        soft_target_mode='continuation' also stores, for each late row of the
        window, the mean of branch_count fresh unguided continuations' exact
        final-step probabilities (see _continuation_final_probability). Branches
        draw only from branch_generator, never from generator, so base paths,
        labels and final-step probabilities do not depend on the branch setup.
        branch_batch_size bounds how many branch states are integrated at once.
        branch_count=1 is useful for validating the implementation only: a fresh
        single continuation has the same conditional target variance as reusing the
        original path's, so regenerating it does not by itself reduce variance.
        'reuse_final' needs no extra collection and returns the ordinary data.
        An optional diagnostics dict accumulates branch_seconds, branch_state_steps
        (intermediate Euler updates over all branch states) and
        branch_final_evaluations for cost accounting.
        """
        _positive_int(num_trajectories, 'num_trajectories')
        _positive_int(num_steps, 'num_steps')
        if batch_size is not None:
            _positive_int(batch_size, 'batch_size')
        _check_soft_target_options(soft_target_mode, soft_target_window, soft_target_mode != 'off')
        branching = soft_target_mode == 'continuation'
        if branching:
            _positive_int(branch_count, 'branch_count')
            if not isinstance(branch_generator, torch.Generator):
                raise ValueError('continuation mode requires a dedicated branch_generator')
            if branch_batch_size is not None:
                _positive_int(branch_batch_size, 'branch_batch_size')
        elif (branch_count is not None or branch_generator is not None
              or branch_batch_size is not None):
            raise ValueError("branch options require soft_target_mode='continuation'")
        if batch_size is not None:
            options = dict(soft_target_mode=soft_target_mode, soft_target_window=soft_target_window,
                           branch_count=branch_count, branch_generator=branch_generator,
                           branch_batch_size=branch_batch_size, diagnostics=diagnostics)
            chunks = [self.sample_trajectories(min(batch_size, num_trajectories - start),
                      num_steps, generator=generator, **options)
                      for start in range(0, num_trajectories, batch_size)]
            soft = {}
            if branching:
                if any(not torch.equal(chunk.soft_target_indices, chunks[0].soft_target_indices)
                       for chunk in chunks):
                    raise ValueError('Chunks selected different soft-target time indices')
                has_errors = [chunk.soft_target_standard_errors is not None for chunk in chunks]
                if any(has_errors) and not all(has_errors):
                    raise ValueError('Chunks disagree on whether standard errors exist')
                soft = dict(soft_target_indices=chunks[0].soft_target_indices,
                    soft_targets=torch.cat([chunk.soft_targets for chunk in chunks]),
                    soft_target_standard_errors=torch.cat(
                        [chunk.soft_target_standard_errors for chunk in chunks]) if all(has_errors) else None)
            return TerminalLabelData(chunks[0].times,
                torch.cat([chunk.features for chunk in chunks]),
                torch.cat([chunk.labels for chunk in chunks]), chunks[0].weights,
                torch.cat([chunk.last_probabilities for chunk in chunks]),
                torch.cat([chunk.last_states for chunk in chunks]), **soft)
        ref = self.reference
        paths = generate_samples(ref, num_trajectories, num_steps=num_steps,
                                 generator=generator, return_trajectory=True)
        dt = self.horizon / num_steps
        times = torch.arange(num_steps, dtype=paths.dtype, device=paths.device) * dt
        features = torch.stack([self.features(t, paths[:, j])
                                for j, t in enumerate(times)], dim=1)
        labels = (paths[:, -1] @ self.w <= self.C)
        weights = torch.full_like(times, dt / self.horizon)
        last = self.exact_last_step_probability(paths[:, -2], dt)
        soft = {}
        if branching:
            started = time.perf_counter()
            soft = self._branch_targets(paths, times, last, dt, soft_target_window, branch_count,
                                        branch_generator, branch_batch_size, diagnostics)
            if diagnostics is not None:
                diagnostics['branch_seconds'] = (diagnostics.get('branch_seconds', 0.)
                                                 + time.perf_counter() - started)
        return TerminalLabelData(times, features, labels[:, None].to(paths.dtype), weights,
                                 last, paths[:, -2].clone(), **soft)

    def _branch_targets(self, paths, times, last, dt, window, branch_count, generator, batch_size,
                        diagnostics=None):
        """Late-row continuation targets for one batch of original paths."""
        n, steps = len(paths), len(times)
        indices = soft_target_indices(times, self.horizon, window)
        targets = paths.new_empty(n, len(indices), 1)
        errors = paths.new_zeros(n, len(indices), 1)
        for position, j in enumerate(indices.tolist()):
            if j == steps - 1:
                # The analytic final step is reused: no simulation, no RNG draw.
                targets[:, position] = last
                continue
            rows = n * branch_count
            if diagnostics is not None:
                diagnostics['branch_state_steps'] = (diagnostics.get('branch_state_steps', 0)
                                                     + rows * (steps - 1 - j))
                diagnostics['branch_final_evaluations'] = (
                    diagnostics.get('branch_final_evaluations', 0) + rows)
            chunk = rows if batch_size is None else batch_size
            values = torch.cat([
                _continuation_final_probability(self, paths[:, j][order // branch_count], j, steps,
                                                generator=generator)
                for order in torch.arange(rows, device=paths.device).split(chunk)])
            values = values.reshape(n, branch_count)
            targets[:, position, 0] = values.mean(1)
            if branch_count > 1:
                errors[:, position, 0] = values.std(1, unbiased=True) / math.sqrt(branch_count)
        return dict(soft_target_indices=indices.to(paths.device), soft_targets=targets,
                    soft_target_standard_errors=errors if branch_count > 1 else None)

    def save_checkpoint(self, path, *, metadata=None):
        """Bundle H with its exact reference weights, architecture, times and event.

        Metadata records experiment context such as asset order, preprocessing,
        seeds and rollout/training settings, using JSON-compatible values.
        Loading returns this context alongside the model. Optimizer state is not stored.
        """
        ref = self.reference
        score = ref._score
        dtype = next(self.network.parameters()).dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Checkpoint precision must be float32 or float64")
        if any(parameter.dtype != dtype or not torch.isfinite(parameter).all()
               for parameter in self.parameters()):
            raise ValueError("H and reference weights must be finite and share one precision")
        layers = list(score.g_zeta)
        if (len(layers) % 2 != 1
                or any(type(layer) is not nn.Linear for layer in layers[::2])
                or any(type(layer) is not nn.ReLU for layer in layers[1::2])):
            raise ValueError("Checkpoint supports the standard DFM Linear/ReLU architecture")
        metadata = json.loads(json.dumps({} if metadata is None else metadata, allow_nan=False))
        def cpu_state(module):
            return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        torch.save({
            "format_version": 1,
            "dtype": str(next(self.network.parameters()).dtype).removeprefix("torch."),
            "reference": {
                "architecture": {"d": ref.d, "k": score.k, "sigma_max": score.sigma_max,
                                 "hidden_dims": [layer.out_features for layer in layers[:-1:2]]},
                "T": ref.T, "t0": ref.t0, "state_dict": cpu_state(score)},
            "event": {"w": self.config.w, "C": self.C},
            "h": {"hidden_dims": self.hidden_dims, "state_dict": cpu_state(self.network),
                  "log_remaining_time": self.log_remaining_time,
                  "normalization": {name: getattr(self, name).detach().cpu().clone()
                      for name in ('normalization_times', 'normalization_mean', 'normalization_scale')}},
            "metadata": metadata,
        }, path)

    @classmethod
    def load_checkpoint(cls, path, *, device="cpu"):
        """Return (H model in eval mode, metadata), restoring the saved precision."""
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("format_version") != 1 or saved.get("dtype") not in ("float32", "float64"):
            raise ValueError("Unsupported H checkpoint format or precision")
        dtype = {"float32": torch.float32, "float64": torch.float64}[saved["dtype"]]
        reference = saved["reference"]
        for state in (reference["state_dict"], saved["h"]["state_dict"]):
            if any(value.dtype != dtype or not torch.isfinite(value).all() for value in state.values()):
                raise ValueError("Checkpoint weights must be finite and match saved precision")
        with torch.random.fork_rng(devices=[]):
            dfm = UnconditionalDFM(**reference["architecture"], T=reference["T"],
                                   t0=reference["t0"]).to(dtype=dtype)
            dfm.factor_score_network.load_state_dict(reference["state_dict"], strict=True)
            model = cls(FrozenDFMReference(dfm), ConditioningConfig(**saved["event"]),
                        hidden_dims=saved["h"]["hidden_dims"],
                        log_remaining_time=saved['h'].get('log_remaining_time', False))
        normalization = saved['h'].get('normalization')
        if normalization is not None:
            grid, mean, scale = (normalization[name] for name in
                                ('normalization_times', 'normalization_mean', 'normalization_scale'))
            if (grid.ndim != 1 or mean.shape != (len(grid), model.reference._score.k+1)
                    or scale.shape != mean.shape
                    or any(v.dtype != dtype or not torch.isfinite(v).all() for v in (grid, mean, scale))
                    or (scale <= 0).any()
                    or (len(grid) and (grid[0] != 0 or grid[-1] >= model.horizon
                                      or (grid[1:] <= grid[:-1]).any()))):
                raise ValueError('Invalid H normalization statistics')
            for name, value in normalization.items():
                setattr(model, name, value)
        model.network.load_state_dict(saved["h"]["state_dict"], strict=True)
        model.to(device=device).eval()
        return model, saved["metadata"]


def _continuation_final_probability(model, x, start, num_steps, *, generator=None,
                                    increments=None):
    """Exact final-step event probability after one unguided continuation per row.

    x holds ambient Euler states X_j at absolute grid index j=start of a
    num_steps-step grid on the original step size dt=(T-t0)/num_steps. Updates
    use absolute generative times t_j=j*dt, j=start..num_steps-2, matching
    sampling.generate_samples; the final Gaussian increment is integrated out
    by HNetwork.exact_last_step_probability, so no final Bernoulli is drawn.
    At start=num_steps-1 this is that exact probability with no RNG use.
    Optional increments [num_steps-1-start, n, d] are actual dW increments
    (variance dt) for deterministic tests. Autograd is respected, so callers
    may differentiate through x; the collector runs it under no_grad.
    """
    reference = model.reference
    if not 0 <= start <= num_steps - 1:
        raise ValueError('start must be a preterminal grid index')
    dt = model.horizon / num_steps
    sqrt_dt = math.sqrt(dt)
    n, d = x.shape
    if increments is not None and increments.shape != (num_steps - 1 - start, n, d):
        raise ValueError('increments must have shape [num_steps-1-start, n, d]')
    modes = [(module, module.training) for module in reference.modules()]
    try:
        reference.eval()
        for j in range(start, num_steps - 1):
            s = torch.full((n, 1), j * dt, device=x.device, dtype=x.dtype)
            drift = 0.5 * x + reference.score(x, reference.T - s)
            noise = (sqrt_dt * torch.randn(x.shape, device=x.device, dtype=x.dtype,
                                           generator=generator)
                     if increments is None else increments[j - start])
            x = x + dt * drift + noise
            if not torch.isfinite(x).all():
                raise ValueError('Continuation produced nonfinite values')
        return model.exact_last_step_probability(x, dt)
    finally:
        for module, training in modes:
            module.training = training


def terminal_label_loss(probabilities, labels, weights):
    """Sum time-weighted squared errors, then average over trajectories.

    probabilities: [N,J,1], labels: [N,1], weights: [J]. No class balancing,
    terminal-time term, or extra division by the number of time points is used.
    """
    if (probabilities.ndim != 3 or probabilities.shape[-1] != 1
            or labels.shape != (probabilities.shape[0], 1)
            or weights.shape != (probabilities.shape[1],)):
        raise ValueError("Expected probabilities [N,J,1], labels [N,1], weights [J]")
    return ((probabilities - labels[:, None]).square()
            * weights[None, :, None]).sum(dim=1).mean()


def _validate_data(model, data):
    dimension = model.reference._score.k + 1
    if (data.features.ndim != 3 or min(data.features.shape) == 0
            or data.features.shape[2] != dimension):
        raise ValueError("features must have nonempty shape [N,J,k+1]")
    n, steps, _ = data.features.shape
    for tensor in (data.times, data.features, data.labels, data.weights):
        model.reference._check_tensor(tensor, "trajectory data")
        if tensor.requires_grad:
            raise ValueError("Training trajectories must be detached from autograd")
    if data.times.shape != (steps,) or data.labels.shape != (n, 1) or data.weights.shape != (steps,):
        raise ValueError("Inconsistent trajectory/time/label/weight dimensions")
    if not ((data.labels == 0) | (data.labels == 1)).all():
        raise ValueError("Terminal labels must be binary")
    if data.times[0] != 0 or (data.times[1:] <= data.times[:-1]).any() or data.times[-1] >= model.horizon:
        raise ValueError("Times must increase from zero and exclude the terminal endpoint")
    edges = torch.cat((data.times, data.times.new_tensor([model.horizon])))
    expected_weights = edges.diff() / model.horizon
    if not torch.allclose(data.weights, expected_weights) or not (data.weights > 0).all():
        raise ValueError("Each time weight must be delta_t / (T-t0)")
    if data.last_probabilities is not None:
        model.reference._check_tensor(data.last_probabilities, 'last probabilities')
        if (data.last_probabilities.shape != (n, 1) or data.last_probabilities.requires_grad
                or ((data.last_probabilities < 0) | (data.last_probabilities > 1)).any()):
            raise ValueError('Last probabilities must be detached [N,1] values in [0,1]')
    _validate_soft_arrays(model, data)


def _validate_soft_arrays(model, data):
    """Structural checks of optional continuation targets, whatever the mode."""
    arrays = (data.soft_target_indices, data.soft_targets, data.soft_target_standard_errors)
    if all(a is None for a in arrays):
        return
    n, steps, _ = data.features.shape
    indices, targets, errors = arrays
    if indices is None or targets is None:
        raise ValueError('Soft targets require both soft_target_indices and soft_targets')
    reference = model.reference
    if (not isinstance(indices, torch.Tensor) or indices.dtype != torch.long
            or indices.ndim != 1 or len(indices) == 0 or indices.requires_grad
            or indices.device != data.features.device):
        raise ValueError('soft_target_indices must be a nonempty detached LongTensor vector')
    if (indices[0] < 0 or indices[-1] != steps - 1 or (indices[1:] <= indices[:-1]).any()):
        raise ValueError('soft_target_indices must be unique, increasing preterminal rows '
                         'ending at the last preterminal row')
    for name, tensor in (('soft_targets', targets), ('soft_target_standard_errors', errors)):
        if tensor is None:
            continue
        reference._check_tensor(tensor, name)
        if tensor.requires_grad:
            raise ValueError(f'{name} must be detached from autograd')
        if tensor.shape != (n, len(indices), 1):
            raise ValueError(f'{name} must have shape [N,K,1]')
    if ((targets < 0) | (targets > 1)).any():
        raise ValueError('soft_targets must be probabilities in [0,1]')
    if errors is not None:
        if (errors < 0).any():
            raise ValueError('soft_target_standard_errors must be nonnegative')
        if (errors[:, -1] != 0).any():
            raise ValueError('The final-step standard error must be zero')
    if data.last_probabilities is not None and not torch.equal(targets[:, -1], data.last_probabilities):
        raise ValueError('Final-row soft targets must equal last_probabilities')


def _predict_trajectories(model, times, features):
    n, steps, dimension = features.shape
    times = times[None, :].expand(n, -1).reshape(-1, 1)
    return model(times, features.reshape(-1, dimension)).reshape(n, steps, 1)


def h_time_weights(model, data, late_time_fraction=0., last_step_fraction=0.):
    """Mix interval weights with inverse-remaining-time weights and a last atom.

    These weights depend only on time, never on the event class. Hence both
    Brier and BCE retain the conditional-probability population optimum.
    """
    if (not math.isfinite(late_time_fraction) or not math.isfinite(last_step_fraction)
            or min(late_time_fraction, last_step_fraction) < 0
            or late_time_fraction + last_step_fraction >= 1):
        raise ValueError('Time mixture fractions must be nonnegative and sum to less than one')
    late = data.weights/(model.horizon-data.times)
    weights = ((1-late_time_fraction-last_step_fraction)*data.weights
               + late_time_fraction*late/late.sum())
    weights = weights.clone()
    weights[-1] += last_step_fraction
    return weights


def _resolve_soft_rows(model, data, mode, window, use_last_step_targets=True):
    """Validate target options against one split; return the resolved row indices.

    Returns None for mode 'off'. Everything is checked before any optimizer or
    random draw, and both new modes need the sampled final-step probabilities.
    """
    _check_soft_target_options(mode, window, use_last_step_targets)
    if mode == 'off':
        return None
    if data.last_probabilities is None:
        raise ValueError('Exact last-step targets require sampled last probabilities')
    rows = soft_target_indices(data.times, model.horizon, window)
    if mode == 'continuation':
        if data.soft_targets is None:
            raise ValueError('continuation mode requires data collected with continuation targets')
        if (len(rows) != len(data.soft_target_indices)
                or not torch.equal(rows, data.soft_target_indices)):
            raise ValueError('Stored soft-target time indices do not match soft_target_window')
    return rows


def _training_targets(data, indices, use_last_step_targets, soft_target_mode='off', soft_rows=None):
    """The single rule choosing the regression target for every (path, time) row.

    off: binary label everywhere, except the final row when use_last_step_targets.
    reuse_final: rows in soft_rows take their own path's final-step probability.
    continuation: rows in soft_rows take the stored branch-average probability.
    All other rows keep the binary terminal outcome. soft_rows come from
    _resolve_soft_rows, so training, validation and sampled-time gathering agree.
    """
    targets = data.labels[indices, None].expand(-1, len(data.times), -1).clone()
    if soft_target_mode == 'off':
        if use_last_step_targets:
            if data.last_probabilities is None:
                raise ValueError('Exact last-step targets require sampled last probabilities')
            targets[:, -1] = data.last_probabilities[indices]
    elif soft_target_mode == 'reuse_final':
        targets[:, soft_rows] = data.last_probabilities[indices][:, None, :]
    elif soft_target_mode == 'continuation':
        targets[:, soft_rows] = data.soft_targets[indices]
    else:
        raise ValueError(f'soft_target_mode must be one of {SOFT_TARGET_MODES}')
    return targets


@torch.no_grad()
def evaluate_h_objective(model, data, *, batch_size=256, loss='bce',
                         late_time_fraction=0., last_step_fraction=0.,
                         use_last_step_targets=False, soft_target_mode='off',
                         soft_target_window=None):
    soft_rows = _resolve_soft_rows(model, data, soft_target_mode, soft_target_window,
                                   use_last_step_targets)
    weights = h_time_weights(model, data, late_time_fraction, last_step_fraction)
    total = 0.
    for indices in torch.arange(len(data.features), device=data.features.device).split(batch_size):
        features = data.features[indices]
        n, steps, dim = features.shape
        logits = model._logits(data.times[None].expand(n, -1).reshape(-1, 1),
                               features.reshape(-1, dim)).reshape(n, steps, 1)
        targets = _training_targets(data, indices, use_last_step_targets, soft_target_mode, soft_rows)
        errors = (F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
                  if loss == 'bce' else (logits.sigmoid()-targets).square())
        total += (errors*weights[None, :, None]).sum().item()
    return total/len(data.features)


@torch.no_grad()
def evaluate_h_scores(model, data, *, batch_size=256, late_time_fraction=0., last_step_fraction=0.,
                      use_last_step_targets=False, soft_target_mode='off', soft_target_window=None):
    """Common binary-label scores beside the configured-target scores.

    binary_* use only the independent terminal outcomes with the interval time
    weights; they are the cross-arm comparison. *_weighted keep binary labels but
    use the configured training time weights. configured_* use the configured
    soft targets and time weights; their noise floor differs from the binary
    scores', so a smaller configured value alone is not an improvement.
    Does not replace evaluate_h, which keeps its original binary Brier meaning.
    """
    _validate_data(model, data)
    _positive_int(batch_size, 'batch_size')
    soft_rows = _resolve_soft_rows(model, data, soft_target_mode, soft_target_window,
                                   use_last_step_targets)
    configured = h_time_weights(model, data, late_time_fraction, last_step_fraction)
    sums = dict.fromkeys(('binary_brier', 'binary_bce', 'binary_brier_weighted',
                          'binary_bce_weighted', 'configured_brier', 'configured_bce'), 0.)
    for indices in torch.arange(len(data.features), device=data.features.device).split(batch_size):
        n, steps, dim = data.features[indices].shape
        logits = model._logits(data.times[None].expand(n, -1).reshape(-1, 1),
                               data.features[indices].reshape(n*steps, dim)).reshape(n, steps, 1)
        binary = data.labels[indices, None].expand(-1, steps, -1)
        soft = _training_targets(data, indices, use_last_step_targets, soft_target_mode, soft_rows)
        for prefix, target in (('binary', binary), ('configured', soft)):
            brier = (logits.sigmoid()-target).square()
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
            if prefix == 'binary':
                sums['binary_brier'] += (brier*data.weights[None, :, None]).sum().item()
                sums['binary_bce'] += (bce*data.weights[None, :, None]).sum().item()
                sums['binary_brier_weighted'] += (brier*configured[None, :, None]).sum().item()
                sums['binary_bce_weighted'] += (bce*configured[None, :, None]).sum().item()
            else:
                sums['configured_brier'] += (brier*configured[None, :, None]).sum().item()
                sums['configured_bce'] += (bce*configured[None, :, None]).sum().item()
    return {name: value/len(data.features) for name, value in sums.items()}


def train_h(model, data, *, epochs=100, batch_size=64, lr=1e-3, generator=None,
            validation_data=None, patience=None, selection=None, loss='brier',
            normalize_inputs=False, late_time_fraction=0., last_step_fraction=0.,
            use_last_step_targets=False, times_per_trajectory=None,
            soft_target_mode='off', soft_target_window=None):
    """Train only H on complete trajectories; keep partial minibatches.

    data must come from this model's unguided reference and event. An optional
    shuffle generator must match the data device. Optional independent validation
    selects the lowest configured probability-loss checkpoint; patience stops
    stale epochs. Defaults preserve uniform-time Brier training. Optional BCE,
    time sampling, scaling and last-step soft targets are independent switches.
    selection receives the validation history and selected epoch.
    soft_target_mode ('off', 'reuse_final', 'continuation') with a positive
    remaining-time soft_target_window replaces binary targets on late rows by
    probabilities (see _training_targets); both require use_last_step_targets.
    Returns sample-weighted epoch
    losses; evaluation should use independent held-out trajectories.
    """
    _check_soft_target_options(soft_target_mode, soft_target_window, use_last_step_targets)
    _validate_data(model, data)
    if validation_data is not None:
        _validate_data(model, validation_data)
    if loss not in ('brier', 'bce'):
        raise ValueError('loss must be brier or bce')
    soft_rows = _resolve_soft_rows(model, data, soft_target_mode, soft_target_window, use_last_step_targets)
    if validation_data is not None:
        _resolve_soft_rows(model, validation_data, soft_target_mode, soft_target_window,
                           use_last_step_targets)
    time_weights = h_time_weights(model, data, late_time_fraction, last_step_fraction)
    if times_per_trajectory is not None:
        _positive_int(times_per_trajectory, 'times_per_trajectory')
    if use_last_step_targets:
        for split in (data, validation_data):
            if split is not None and split.last_probabilities is None:
                raise ValueError('Exact last-step targets require sampled last probabilities')
    if normalize_inputs:
        model.fit_normalization(data)
    legacy = (loss == 'brier' and late_time_fraction == last_step_fraction == 0
              and not use_last_step_targets and times_per_trajectory is None
              and soft_target_mode == 'off')
    if patience is not None:
        _positive_int(patience, "patience")
        if validation_data is None:
            raise ValueError("patience requires validation_data")
    best, best_state, best_epoch = float("inf"), None, 0
    validation_history = []
    _positive_int(epochs, "epochs")
    _positive_int(batch_size, "batch_size")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    ref = model.reference
    if any(parameter.requires_grad or parameter.grad is not None for parameter in ref.parameters()):
        raise ValueError("The DFM reference must remain frozen with no parameter gradients")
    optimizer = torch.optim.Adam(model.network.parameters(), lr=lr)
    history = []
    modes = [(module, module.training) for module in model.modules()]
    n = data.features.shape[0]
    try:
        model.train()
        for epoch in range(epochs):
            order = torch.randperm(n, device=data.features.device, generator=generator)
            total = 0.0
            for indices in order.split(batch_size):
                if legacy:
                    predictions = _predict_trajectories(model, data.times, data.features[indices])
                    objective = terminal_label_loss(predictions, data.labels[indices], data.weights)
                else:
                    targets = _training_targets(data, indices, use_last_step_targets,
                                                soft_target_mode, soft_rows)
                    if times_per_trajectory is None:
                        features = data.features[indices]
                        times = data.times[None].expand(len(indices), -1)
                    else:
                        # Independent draws for each trajectory. No importance
                        # correction: the mixture IS the intended time objective.
                        js = torch.multinomial(time_weights, len(indices)*times_per_trajectory,
                                               replacement=True, generator=generator)
                        js = js.reshape(len(indices), times_per_trajectory)
                        features = data.features[indices[:, None], js]
                        times = data.times[js]
                        targets = targets.gather(1, js[:, :, None])
                    logits = model._logits(times.reshape(-1, 1), features.reshape(-1, features.shape[-1]))
                    errors = (F.binary_cross_entropy_with_logits(logits, targets.reshape(-1, 1), reduction='none')
                              if loss == 'bce' else (logits.sigmoid()-targets.reshape(-1, 1)).square())
                    objective = (errors.mean() if times_per_trajectory is not None else
                                 (errors.reshape(len(indices), -1)*time_weights).sum(1).mean())
                if not torch.isfinite(objective):
                    raise ValueError("Nonfinite H training loss")
                optimizer.zero_grad()
                objective.backward()
                optimizer.step()
                total += objective.item() * len(indices)
            history.append(total / n)
            if validation_data is not None:
                if legacy:
                    score = evaluate_h(model, validation_data,
                        constant_probability=float(data.labels.mean()), batch_size=batch_size)["h_loss"]
                else:
                    score = evaluate_h_objective(model, validation_data, batch_size=batch_size,
                        loss=loss, late_time_fraction=late_time_fraction,
                        last_step_fraction=last_step_fraction, use_last_step_targets=use_last_step_targets,
                        soft_target_mode=soft_target_mode, soft_target_window=soft_target_window)
                if not math.isfinite(score):
                    raise ValueError("Nonfinite H validation loss")
                validation_history.append(score)
                if score < best:
                    best, best_epoch = score, epoch + 1
                    best_state = {k: v.detach().clone() for k, v in model.network.state_dict().items()}
                if patience is not None and epoch + 1 - best_epoch >= patience:
                    break
            if (epoch+1) % 10 == 0 and not legacy:
                print(f'H epoch {epoch+1}/{epochs}: train={history[-1]:.6f}, '
                      f'validation={validation_history[-1] if validation_history else float("nan"):.6f}', flush=True)
        if best_state is not None:
            model.network.load_state_dict(best_state)
        if selection is not None:
            selection.update(best_epoch=best_epoch if best_state is not None else None,
                             validation_history=validation_history,
                             best_loss=best if best_state is not None else None)
    finally:
        for module, training in modes:
            module.training = training
    return history


@torch.no_grad()
def evaluate_h(model, data, *, constant_probability, batch_size=256):
    """Held-out weighted squared loss and a supplied constant-probability baseline.

    Estimate the baseline probability from training labels only. These losses
    measure probability prediction, not accuracy of spatial/log-probability gradients.
    """
    _validate_data(model, data)
    _positive_int(batch_size, "batch_size")
    if not math.isfinite(constant_probability) or not 0 <= constant_probability <= 1:
        raise ValueError("constant_probability must lie in [0,1]")
    total = baseline = 0.0
    n = data.features.shape[0]
    for start in range(0, n, batch_size):
        features = data.features[start:start+batch_size]
        labels = data.labels[start:start+batch_size]
        predictions = _predict_trajectories(model, data.times, features)
        total += terminal_label_loss(predictions, labels, data.weights).item() * len(features)
        constant = torch.full_like(predictions, constant_probability)
        baseline += terminal_label_loss(constant, labels, data.weights).item() * len(features)
    return {"h_loss": total / n, "constant_loss": baseline / n}


def diagnose_h_terminal(model, data, *, batch_size=256, late_time_fraction=0., last_step_fraction=0.,
                        soft_target_mode='off', soft_target_window=None):
    """Independent late-time calibration and ambient last-step gradient checks.

    Calibration targets follow the configured soft-target mode. With the default
    'off' they are the terminal label, except the exact final-step probability.

    Exact Euler targets condition on full x. A reduced H may have irreducible
    residual error here, since G_t sufficiency for the continuous dynamics does
    not imply sufficiency for their ambient Euler discretization.
    """
    _validate_data(model, data)
    soft_rows = _resolve_soft_rows(model, data, soft_target_mode, soft_target_window)
    configured = (None if soft_rows is None else
                  _training_targets(data, torch.arange(len(data.features), device=data.features.device),
                                    True, soft_target_mode, soft_rows))
    in_window = set() if soft_rows is None else set(soft_rows.tolist())
    weights = h_time_weights(model, data, late_time_fraction, last_step_fraction)
    remaining = model.horizon-data.times
    selected = sorted(set([len(data.times)-1] + [int(torch.argmin((remaining-v).abs()))
                                                for v in (.1, .5, 1.)]))
    rows, calibration = [], []
    with torch.no_grad():
        for j in selected:
            p = torch.cat([model(data.times[j], chunk) for chunk in data.features[:, j].split(batch_size)])
            if configured is not None:
                target = configured[:, j]
                kind = (soft_target_mode if j in in_window else 'terminal_label')
                if j == len(data.times)-1:
                    kind = 'exact_euler_probability'
            else:
                target = data.last_probabilities if j == len(data.times)-1 else data.labels
                kind = 'exact_euler_probability' if j == len(data.times)-1 else 'terminal_label'
            rows.append(dict(time=float(data.times[j]), remaining=float(remaining[j]),
                n=len(p), prediction_mean=float(p.mean()), target_mean=float(target.mean()),
                brier=float((p-target).square().mean()), target=kind))
            for b in range(10):
                mask = (p[:, 0] >= b/10) & (p[:, 0] < (b+1)/10 if b < 9 else p[:, 0] <= 1)
                if mask.any():
                    calibration.append(dict(remaining=float(remaining[j]), bin_lower=b/10,
                        n=int(mask.sum()), predicted=float(p[mask].mean()), observed=float(target[mask].mean())))
        last_y = data.features[:, -1]
        p = torch.cat([model(data.times[-1], chunk) for chunk in last_y.split(batch_size)])
        activations = torch.cat([torch.tanh(model.network[0](model.network_inputs(data.times[-1], chunk)))
                                 for chunk in last_y.split(batch_size)])
        summary = dict(late_rows=rows, calibration=calibration,
                       first_layer_saturation_fraction=float((activations.abs() > .99).float().mean()),
                       final_0_1_time_weight=float(weights[remaining <= .1].sum()))
        if data.last_probabilities is not None:
            q = data.last_probabilities
            high = q[:, 0] > .9
            summary.update(last_probability_rmse=float((p-q).square().mean().sqrt()),
                high_probability_n=int(high.sum()),
                high_probability_prediction_mean=float(p[high].mean()) if high.any() else None,
                high_probability_target_mean=float(q[high].mean()) if high.any() else None)
    if data.last_states is not None:
        squared_error = squared_exact = cosine_sum = 0.
        count = 0
        for x in data.last_states.split(batch_size):
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                log_exact = model.exact_last_step_probability(x, float(remaining[-1]), log=True)
                exact = torch.autograd.grad(log_exact.sum(), x)[0]
                learned = torch.autograd.grad(model.log_prob(data.times[-1], model.features(data.times[-1], x)).sum(), x)[0]
            # Exclude nearly impossible/already certain events from the relative
            # score diagnostic; report the included count explicitly.
            mask = (log_exact[:, 0] > math.log(.01)) & (log_exact[:, 0] < math.log(.99))
            squared_error += float((learned[mask]-exact[mask]).square().sum())
            squared_exact += float(exact[mask].square().sum())
            cosine_sum += float(F.cosine_similarity(learned[mask], exact[mask]).sum())
            count += int(mask.sum())
        summary.update(gradient_n=count, gradient_probability_range=[.01, .99],
            gradient_relative_rmse=math.sqrt(squared_error/squared_exact) if squared_exact else None,
            gradient_mean_cosine=cosine_sum/count if count else None)
    return summary
