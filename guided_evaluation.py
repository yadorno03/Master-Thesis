"""Distribution diagnostics against an independently rejected reference sample."""

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


def conditional_diagnostics(samples, benchmark, weights, threshold, *, seed=0,
                            projections=32):
    """Compare ALL guided draws, including violations, in model coordinates.

    This reports distances, not a pass/fail certificate of conditional-law
    accuracy. Projection directions use independent seeded randomness. Neither
    the event nor the fitted model is selected using the benchmark.
    """
    x, target, w = (np.asarray(value, dtype=float) for value in (samples, benchmark, weights))
    if (x.ndim != 2 or target.ndim != 2 or min(x.shape) < 1
            or len(x) < 2 or len(target) < 2 or x.shape[1] != target.shape[1]
            or w.shape != (x.shape[1],) or not np.any(w)
            or not all(np.isfinite(v).all() for v in (x, target, w))
            or np.ndim(threshold) != 0 or not np.isfinite(threshold)):
        raise ValueError('Need finite sample matrices (at least two rows) and a nonzero portfolio')
    if isinstance(projections, bool) or not isinstance(projections, int) or projections < 1:
        raise ValueError('projections must be a positive integer')
    pnl, target_pnl = x @ w, target @ w
    directions = np.random.default_rng(seed).normal(size=(x.shape[1], projections))
    directions /= np.linalg.norm(directions, axis=0)
    a, b = x @ directions, target @ directions
    quantiles = np.quantile(pnl, [.01, .05, .5, .95, .99])
    target_quantiles = np.quantile(target_pnl, [.01, .05, .5, .95, .99])
    covariance_error = (np.atleast_2d(np.cov(x, rowvar=False))
                        - np.atleast_2d(np.cov(target, rowvar=False)))
    return {
        'samples': len(x), 'benchmark_samples': len(target),
        'event_rate': float(np.mean(pnl <= threshold)),
        'benchmark_event_rate': float(np.mean(target_pnl <= threshold)),
        'portfolio_mean': float(pnl.mean()),
        'benchmark_portfolio_mean': float(target_pnl.mean()),
        'portfolio_std': float(pnl.std(ddof=1)),
        'benchmark_portfolio_std': float(target_pnl.std(ddof=1)),
        'portfolio_quantiles': quantiles.tolist(),
        'benchmark_portfolio_quantiles': target_quantiles.tolist(),
        'portfolio_wasserstein': float(wasserstein_distance(pnl, target_pnl)),
        'portfolio_ks_statistic': float(ks_2samp(pnl, target_pnl).statistic),
        'portfolio_mean_bias': float(pnl.mean() - target_pnl.mean()),
        'portfolio_quantile_rmse': float(np.sqrt(np.mean((quantiles - target_quantiles)**2))),
        'mean_vector_rmse': float(np.sqrt(np.mean((x.mean(0) - target.mean(0))**2))),
        'covariance_rmse': float(np.sqrt(np.mean(covariance_error**2))),
        'sliced_wasserstein': float(np.mean([wasserstein_distance(a[:, j], b[:, j])
                                            for j in range(projections)])),
        'projection_seed': seed, 'projections': projections,
        'terminal_filtering': False,
    }
