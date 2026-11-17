"""Unconditional DFM fitting and per-case benchmark artifacts."""
from copy import deepcopy
from dataclasses import asdict
from numbers import Integral
import logging
import platform
import time

import numpy as np
import pandas as pd
import torch

from DFM import UnconditionalDFM
from NeuralNets import REGIME_PARAMETERS
from synthetic_conditional_truth import (
    validate_dimensions, make_population, population_metadata, population_details,
    check_population_model,
)
from .configuration import Streams, write_json
from .reporting import DFM_LABELS, compare, factor_variance_summary, plot_comparison

# Progress messages; the notebook shows results only.
logger = logging.getLogger(__name__)


def population_for_config(config):
    """The Chen et al. (2025) population at the configured dimensions."""
    return make_population(config.d, 'gaussian', k=config.k)


def dfm_training_times(n, cfg, generator):
    """Time proposal; log_importance corrects back to uniform-time DSM."""
    u = torch.rand(n, 1, generator=generator, device=cfg.device)
    if cfg.dfm_time_sampling == 'uniform':
        return cfg.t0 + (cfg.T - cfg.t0) * u
    return cfg.t0 * torch.exp(np.log(cfg.T / cfg.t0) * u)


def _time_weighted(errors, t, h, cfg):
    if cfg.dfm_time_sampling == 'log_importance':
        errors = errors * t[:, 0] * np.log(cfg.T / cfg.t0) / (cfg.T - cfg.t0)
    elif cfg.dfm_time_sampling == 'log_balanced':
        # An explicit alternative objective: E_log-time[h(t) * DSM error].
        # Noise-variance weighting of Song et al. (2021) and Ho et al. (2020).
        errors = errors * h[:, 0]
    return errors


def dfm_loss_parts(model, x, t, z, cfg, score=None):
    """The training loss and its factor part, the squared error inside span(V), for a score network.

    ``score`` defaults to the model's network; fit_dfm passes its averaged copy.
    """
    score = model.factor_score_network if score is None else score
    h = -torch.expm1(-t)
    error = score(model.q_sample(x, t, z), t) + z / torch.sqrt(h)
    return (_time_weighted(error.square().sum(-1), t, h, cfg).mean(),
            _time_weighted((error @ score.V).square().sum(-1), t, h, cfg).mean())


def dfm_training_loss(model, x, t, z, cfg):
    return dfm_loss_parts(model, x, t, z, cfg)[0]


@torch.no_grad()
def refit_linear_part(model, x, cfg, generator, draws=64, chunk=2048, ridge=1e-6):
    """Fit the linear refit Gamma(t) u of the network's g by least squares on the training rows x.

    Score matching barely constrains the scale of the score along the factors: the factor part of the
    loss is a few percent of it, so a trained DFM can generate factors with the wrong variances. Given
    the trained network, the training objective is quadratic in Gamma, which is piecewise linear in log t
    between the network's refit knots, spread evenly over [log t0, log T]. Its minimizer solves the normal
    equations, estimated on ``draws`` noised copies of each row with the times and weights of
    cfg.dfm_time_sampling. With u = V^T D x, the noise term has a closed-form average,
    E[(z / sqrt(h)) u^T | x0, t] = D V, so only the network's scores are averaged. ``ridge`` times the
    mean diagonal of the normal matrix keeps it invertible. Returns the objective on these draws
    before and after the refit.
    """
    score = model.factor_score_network
    if not score.refit_knots:
        raise ValueError('The score network has no refit knots')
    k, K = score.k, score.refit_knots
    score.refit_log_times.copy_(torch.linspace(np.log(cfg.t0), np.log(cfg.T), K, dtype=score.refit_log_times.dtype))
    score.refit_gamma.zero_()
    V = score.V.double()
    c = (torch.sigmoid(score.c_raw) * score.sigma_max).double()
    gram = torch.zeros(K * k * k, K * k * k, dtype=torch.float64, device=V.device)
    rhs = torch.zeros(K * k * k, dtype=torch.float64, device=V.device)
    rows, before = len(x) * draws, 0.0
    for start in range(0, rows, chunk):
        # Rows start..start+chunk of the rows repeated draws times.
        x0 = x[torch.arange(start, min(start + chunk, rows), device=x.device) % len(x)]
        t = dfm_training_times(len(x0), cfg, generator).to(x0.dtype)
        z = torch.randn(x0.shape, generator=generator, device=x0.device).to(x0.dtype)
        s = score(model.q_sample(x0, t, z), t).double()
        t, z = t.double(), z.double()
        alpha, h = torch.exp(-t / 2), -torch.expm1(-t)
        weight = _time_weighted(torch.ones_like(h[:, 0]), t, h, cfg)
        before += float((weight * (s + z / h.sqrt()).square().sum(-1)).sum())
        D = 1 / (h + alpha ** 2 * c)                                            # [N, d]
        u = (D * (alpha * x0.double() + h.sqrt() * z)) @ V                      # [N, k]
        VD2V = torch.einsum('nd,di,dj->nij', D ** 2, V, V)                      # V^T D^2 V
        # Quadratic term: E[w phi phi^T (alpha^2 V^T D^2 V) kron (u u^T)], over the (knot, i, j) entries of Gamma.
        QB = ((alpha ** 2)[:, :, None, None, None] * VD2V[:, :, None, :, None]
              * (u[:, None, :, None, None] * u[:, None, None, None, :])).reshape(len(x0), k * k, k * k)
        phi = score.refit_hats(t)                                               # [N, K]
        PP = (weight[:, None, None] * phi[:, :, None] * phi[:, None, :]).reshape(len(x0), K * K)
        gram += ((PP.T @ QB.reshape(len(x0), -1)).reshape(K, K, k * k, k * k)
                 .permute(0, 2, 1, 3).reshape(K * k * k, K * k * k))
        # Linear term: alpha (V^T D s u^T + V^T D^2 V), the second part from the noise.
        F = alpha[:, :, None] * (((D * s) @ V).unsqueeze(-1) * u.unsqueeze(-2) + VD2V)
        rhs -= ((weight[:, None] * phi).T @ F.reshape(len(x0), k * k)).reshape(-1)
    gram, rhs, before = gram / rows, rhs / rows, before / rows
    gram += ridge * gram.diagonal().mean() * torch.eye(len(gram), dtype=gram.dtype, device=gram.device)
    gamma = torch.linalg.solve(gram, rhs)
    if not torch.isfinite(gamma).all():
        raise ValueError('Nonfinite linear refit')
    score.refit_gamma.copy_(gamma.reshape(K, k, k))
    return dict(objective_before=before, objective_after=before - float(gamma @ rhs))


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
                             score_network=cfg.dfm_network, regimes=cfg.dfm_regimes,
                             refit_knots=cfg.dfm_refit_knots).to(cfg.device)
    x = torch.tensor(train, dtype=torch.float32, device=cfg.device)
    v = torch.tensor(val, dtype=torch.float32, device=cfg.device)
    if cfg.dfm_standardization == 'per_asset':
        # Statistics of the training rows only; validation rows use the same map.
        model.standardization.fit(train)
        x, v = model.standardization.standardize(x), model.standardization.standardize(v)
    if cfg.dfm_initialization == 'pca':
        initialize_dfm_from_observations(model, x)
    score = model.factor_score_network
    if cfg.dfm_network in ('gaussian_mlp', 'regime_mlp'):
        # The Gaussian part starts from the Gaussian factor model of the training rows, given V and c.
        score.initialize_prior(x)
    if cfg.dfm_network == 'regime_mlp':
        # Scales and table by maximum likelihood: they carry too little score-matching loss to be found by it.
        logger.info('Regime initialization: %s', score.initialize_regimes(x))
    rng = streams.torch('dfm_noise', cfg.device)
    vrng = streams.torch('dfm_validation_noise', cfg.device)
    # The same noising times and noise at every epoch, dfm_validation_draws per validation row.
    v = v.repeat(cfg.dfm_validation_draws, 1)
    vt = dfm_training_times(len(v), cfg, vrng)
    vz = torch.randn(v.shape, generator=vrng, device=cfg.device)

    def loss(x, t, z):
        return dfm_training_loss(model, x, t, z, cfg)
    # V and the regime scales and table train at a reduced rate, after the loading warm-up; every other
    # parameter (with a Gaussian part, also its mean and covariance) trains at the base rate. Without
    # dfm_train_analytic only the MLP g trains: V, c and the analytic factor law keep their initial values.
    frozen = set() if cfg.dfm_train_analytic else {name for name, _ in score.named_parameters()
                                                   if not name.startswith('g_zeta.')}
    for name, p in score.named_parameters():
        p.requires_grad_(name not in frozen)
    loading = [score.V] if 'V' not in frozen else []
    regime = [p for name, p in score.named_parameters() if name in REGIME_PARAMETERS and name not in frozen]
    opt = torch.optim.Adam([
        *([{'params': loading, 'lr': cfg.lr * cfg.dfm_loading_lr_scale}] if loading else []),
        {'params': [p for name, p in score.named_parameters() if name not in ('V', *REGIME_PARAMETERS, *frozen)],
         'lr': cfg.lr},
        *([{'params': regime, 'lr': cfg.lr * cfg.dfm_loading_lr_scale}] if regime else []),
    ])
    # Even reduced-budget runs retain at least one joint training epoch.
    warmup = min(cfg.dfm_loading_warmup_epochs, cfg.dfm_epochs - 1)
    # Exponential moving average of the weights (Ho et al., 2020): the model validated and returned.
    average = deepcopy(score).requires_grad_(False) if cfg.dfm_ema_decay else None
    best = float('inf')
    history = []
    updates = 0
    for epoch in range(cfg.dfm_epochs):
        for p in (*loading, *regime):
            p.requires_grad_(epoch >= warmup)
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
            if average is not None:
                updates += 1
                # Early averages follow the weights closely, so the start does not dominate them.
                rate = 1 - min(cfg.dfm_ema_decay, (1 + updates) / (10 + updates))
                with torch.no_grad():
                    for averaged, parameter in zip(average.parameters(), score.parameters()):
                        averaged.lerp_(parameter, rate)
        evaluated = score
        if average is not None:
            # An average of orthonormal loadings is not orthonormal: retract the copy that is evaluated.
            evaluated = deepcopy(average)
            evaluated.project_orthogonal()
        with torch.no_grad():
            vl, vf = (part.item() for part in dfm_loss_parts(model, v, vt, vz, cfg, evaluated))
        if not np.isfinite(vl):
            raise ValueError(f'Nonfinite DFM validation loss at epoch {epoch + 1}')
        # validation_factor is the part of the validation loss inside span(V): a diagnostic.
        history.append(dict(epoch=epoch + 1, train=total / len(x), validation=vl, validation_factor=vf))
        if vl < best:
            best = vl
            state = deepcopy(evaluated.state_dict())
            selected = epoch + 1
    score.load_state_dict(state)
    model.eval()
    model.requires_grad_(False)
    for p in model.parameters():
        p.grad = None
    if cfg.dfm_refit_knots:
        before = dfm_loss_parts(model, v, vt, vz, cfg)
        fitted = refit_linear_part(model, x, cfg, streams.torch('dfm_refit', cfg.device))
        after = dfm_loss_parts(model, v, vt, vz, cfg)
        logger.info('Linear refit: training objective %.4f -> %.4f; validation loss %.4f -> %.4f, factor part '
                    '%.4f -> %.4f', fitted['objective_before'], fitted['objective_after'], before[0], after[0],
                    before[1], after[1])
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
    logger.info("Running %s, seed=%d, d=%d", DFM_LABELS[config.dfm_network], seed, config.d)
    folder = output / dfm_folder_name(case, seed, config.dfm_network)
    folder.mkdir()
    streams = Streams(seed)
    pop = population_for_config(config)
    write_json(folder / "streams.json", streams.seeds)
    logger.info("Assets: %d; population factors: %d; DFM factors: %d", pop.d, pop.k, config.k)
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
    logger.info("Finished: %s", output)
    return metrics, numeric
