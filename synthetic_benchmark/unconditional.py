"""Unconditional DFM fitting and per-case benchmark artifacts."""
from copy import deepcopy
from dataclasses import asdict
from numbers import Integral
import platform
import time

import numpy as np
import pandas as pd
import torch

from DFM import UnconditionalDFM
from synthetic_conditional_truth import (
    validate_dimensions, make_population, population_metadata, population_details,
    check_population_model,
)
from .configuration import Streams, write_json
from .reporting import DFM_LABELS, compare, factor_variance_summary, plot_comparison


def population_for_config(config):
    """The Chen et al. (2025) population at the configured dimensions."""
    return make_population(config.d, 'gaussian', k=config.k)


def dfm_training_times(n, cfg, generator):
    """Time proposal; log_importance corrects back to uniform-time DSM."""
    u = torch.rand(n, 1, generator=generator, device=cfg.device)
    if cfg.dfm_time_sampling == 'uniform':
        return cfg.t0 + (cfg.T - cfg.t0) * u
    return cfg.t0 * torch.exp(np.log(cfg.T / cfg.t0) * u)


def dfm_training_loss(model, x, t, z, cfg):
    h = -torch.expm1(-t)
    pred = model.factor_score_network(model.q_sample(x, t, z), t)
    errors = (pred + z / torch.sqrt(h)).square().sum(-1)
    if cfg.dfm_time_sampling == 'log_importance':
        errors = errors * t[:, 0] * np.log(cfg.T / cfg.t0) / (cfg.T - cfg.t0)
    elif cfg.dfm_time_sampling == 'log_balanced':
        # An explicit alternative objective: E_log-time[h(t) * DSM error].
        # Noise-variance weighting of Song et al. (2021) and Ho et al. (2020).
        errors = errors * h[:, 0]
    return errors.mean()


@torch.no_grad()
def initialize_dfm_from_observations(model, observations):
    """PCA warm start from training returns only, in the DFM's model coordinates.

    The principal-components factor estimate (Bai, 2003) initializes V and c.

    Center solely to estimate covariance. Neither the true loadings nor any
    validation data enter this initializer. All parameters are subsequently fit.
    """
    if not torch.isfinite(observations).all() or len(observations) < 2:
        raise ValueError('PCA initialization requires at least two finite training rows')
    centered = observations - observations.mean(0)
    covariance = centered.T @ centered / (len(observations) - 1)
    _, vectors = torch.linalg.eigh(covariance)
    score = model.factor_score_network
    loadings = vectors[:, -score.k:]
    residual = centered - (centered @ loadings) @ loadings.T
    variances = residual.square().mean(0).clamp(min=1e-4, max=.95 * score.sigma_max)
    score.V.copy_(loadings)
    score.c_raw.copy_(torch.logit(variances / score.sigma_max))


def fit_dfm(train, val, cfg, streams):
    """Only observations, k, and algorithm configuration enter this boundary."""
    if isinstance(cfg.dfm_epochs, bool) or not isinstance(cfg.dfm_epochs, Integral) or cfg.dfm_epochs <= 0:
        raise ValueError('dfm_epochs must be a positive integer')
    validate_dimensions(cfg.d, cfg.k)
    torch.manual_seed(streams.seeds['model_init'])
    model = UnconditionalDFM(cfg.d, cfg.k, sigma_max=2.0, T=cfg.T, t0=cfg.t0, hidden_dims=[cfg.width, cfg.width],
                             score_network=cfg.dfm_network).to(cfg.device)
    x = torch.tensor(train, dtype=torch.float32, device=cfg.device)
    v = torch.tensor(val, dtype=torch.float32, device=cfg.device)
    if cfg.dfm_standardization == 'per_asset':
        # Statistics of the training rows only; validation rows use the same map.
        model.standardization.fit(train)
        x, v = model.standardization.standardize(x), model.standardization.standardize(v)
    if cfg.dfm_initialization == 'pca':
        initialize_dfm_from_observations(model, x)
    if cfg.dfm_network == 'gaussian_mlp':
        # The Gaussian part starts from the Gaussian factor model of the training rows, given V and c.
        model.factor_score_network.initialize_prior(x)
    rng = streams.torch('dfm_noise', cfg.device)
    vrng = streams.torch('dfm_validation_noise', cfg.device)
    vt = dfm_training_times(len(v), cfg, vrng)
    vz = torch.randn(v.shape, generator=vrng, device=cfg.device)

    def loss(x, t, z):
        return dfm_training_loss(model, x, t, z, cfg)
    score = model.factor_score_network
    # Every parameter except V (with a Gaussian part, also its mean and covariance) trains at the base rate.
    opt = torch.optim.Adam([
        {'params': [score.V], 'lr': cfg.lr * cfg.dfm_loading_lr_scale},
        {'params': [p for name, p in score.named_parameters() if name != 'V'], 'lr': cfg.lr},
    ])
    # Even reduced-budget runs retain at least one joint training epoch.
    warmup = min(cfg.dfm_loading_warmup_epochs, cfg.dfm_epochs - 1)
    best = float('inf')
    history = []
    for epoch in range(cfg.dfm_epochs):
        score.V.requires_grad_(epoch >= warmup)
        total = 0.0
        for ix in torch.randperm(len(x), generator=rng, device=cfg.device).split(cfg.batch):
            t = dfm_training_times(len(ix), cfg, rng)
            z = torch.randn(len(ix), cfg.d, generator=rng, device=cfg.device)
            opt.zero_grad()
            l = loss(x[ix], t, z)
            if not torch.isfinite(l):
                raise ValueError(f'Nonfinite DFM training loss at epoch {epoch + 1}')
            l.backward()
            opt.step()
            model.factor_score_network.project_orthogonal()
            total += l.item() * len(ix)
        with torch.no_grad():
            vl = loss(v, vt, vz).item()
        if not np.isfinite(vl):
            raise ValueError(f'Nonfinite DFM validation loss at epoch {epoch + 1}')
        history.append(dict(epoch=epoch + 1, train=total / len(x), validation=vl))
        if vl < best:
            best = vl
            state = deepcopy(model.state_dict())
            selected = epoch + 1
    model.load_state_dict(state)
    model.eval()
    model.requires_grad_(False)
    for p in model.parameters():
        p.grad = None
    return (model, history, selected)


def dfm_folder_name(case, seed, network):
    """Run sub-folder of one trained DFM."""
    return f"{case}_seed{seed}_{network}"


def initialize_unconditional_run(output, config):
    """Create a fresh run directory and record configuration and environment."""
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config.torch_threads)
    write_json(output / "config.json", asdict(config))
    write_json(output / "environment.json", dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, device=config.device))


def prepare_unconditional_case(output, config, case, seed):
    """Create the case directory, streams, and population provenance before drawing data."""
    print(f"Running {DFM_LABELS[config.dfm_network]}, seed={seed}, d={config.d}", flush=True)
    folder = output / dfm_folder_name(case, seed, config.dfm_network)
    folder.mkdir()
    streams = Streams(seed)
    pop = population_for_config(config)
    write_json(folder / "streams.json", streams.seeds)
    print(f"Assets: {pop.d}; population factors: {pop.k}; DFM factors: {config.k}", flush=True)
    write_json(folder / "population.json", population_details(pop, case))
    return folder, streams, pop


def fit_and_save_dfm(train, val, config, streams, pop, case, folder):
    fit_start = time.perf_counter()
    model, history, selected = fit_dfm(train, val, config, streams)
    check_population_model(pop, model)
    fit_seconds = time.perf_counter() - fit_start
    pd.DataFrame(history).to_csv(folder / "dfm_curve.csv", index=False)
    torch.save(dict(state=model.state_dict(), config=asdict(config), selected_epoch=selected, **population_metadata(pop, case)), folder / "dfm.pt")
    return model, history, selected, fit_seconds


def evaluate_unconditional_samples(generated, truth, truth2, pop, reference, config, case, seed, fit_seconds, generation_seconds):
    """Metrics of generated returns; the learned loadings are compared in the DFM's model coordinates."""
    result = compare(generated, truth, pop)
    V = reference._score.V.detach().cpu().double().numpy()
    # Standardizing asset i divides its loadings by scale_i, so the factor span becomes span(B / scale).
    scale = reference.standardization.scale.detach().cpu().double().numpy()
    basis = np.linalg.qr(pop.B / scale[:, None])[0]
    result.update(case=case, seed=seed, method=DFM_LABELS[config.dfm_network], dfm_network=config.dfm_network,
                  fit_seconds=fit_seconds, generation_seconds=generation_seconds,
                  loading_projection_error=np.linalg.norm(V @ V.T - basis @ basis.T) / np.sqrt(2 * config.k),
                  **factor_variance_summary(generated, pop))
    baseline = dict(compare(truth2, truth, pop), case=case, seed=seed, method="truth-vs-truth",
                    dfm_network=config.dfm_network, **factor_variance_summary(truth2, pop))
    return result, baseline


def save_unconditional_case(folder, case, generated, truth, truth2, pop, result, baseline):
    """Save metrics, the four-panel comparison, and all unconditional draws."""
    pd.DataFrame([result, baseline]).to_csv(folder / "metrics.csv", index=False)
    plot_comparison(generated, truth, pop, None, folder / "unconditional.png", f"{case}: unconditional")
    np.savez_compressed(folder / "unconditional_samples.npz", generated=generated, truth=truth, truth2=truth2)


def save_unconditional_summary(rows, output, started):
    # Every DFM of a seed shares the same truth draws, hence the same truth-vs-truth row.
    metrics = pd.DataFrame(rows).drop_duplicates(subset=['case', 'seed', 'method'], ignore_index=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    numeric = ['mean_error', 'covariance_error', 'correlation_error', 'portfolio_ks', 'portfolio_w1', 'sliced_w1',
               'mmd2_unbiased', 'marginal_pit_ks_mean', 'marginal_pit_ks_std', 'leading_factor_variance_ratio',
               'factor_variance_ratio_min', 'factor_variance_ratio_max', 'residual_variance_ratio',
               'event_0.05', 'event_0.01']
    metrics.groupby(['case', 'method'])[numeric].agg(['mean', 'std', 'count']).to_csv(output / "seed_summary.csv")
    write_json(output / "runtime.json", dict(seconds=time.perf_counter() - started))
    print("Finished:", output)
    return metrics, numeric
