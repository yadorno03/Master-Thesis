"""Distribution metrics, figures, and the benchmark's written interpretation."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
from scipy.stats import ks_2samp, wasserstein_distance, kstest

def correlation(c):
    s = np.sqrt(np.diag(c))
    return c / np.outer(s, s)

def compare(x, truth, pop, c=None, seed=781):
    """All rows enter every metric. RBF MMD uses at most 512 deterministic rows."""
    mean, cov = pop.moments(None)
    empirical = np.cov(x, rowvar=False)
    scale = np.sqrt(np.trace(pop.moments()[1]) / len(mean))
    p = x @ pop.w
    q = truth @ pop.w
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(64, x.shape[1]))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    sw = np.mean([wasserstein_distance(x @ a, truth @ a) for a in directions])
    bandwidth2 = np.median(pdist(truth[:min(512, len(truth))]) ** 2)
    a = x[:512]
    b = truth[:512]
    xx = np.exp(-cdist(a, a, 'sqeuclidean') / (2 * bandwidth2))
    yy = np.exp(-cdist(b, b, 'sqeuclidean') / (2 * bandwidth2))
    xy = np.exp(-cdist(a, b, 'sqeuclidean') / (2 * bandwidth2))
    mmd = (xx.sum() - np.trace(xx)) / (len(a) * (len(a) - 1)) + (yy.sum() - np.trace(yy)) / (len(b) * (len(b) - 1)) - 2 * xy.mean()
    out = dict(n=len(x), mean_error=np.linalg.norm(x.mean(0) - mean) / (np.sqrt(len(mean)) * scale), covariance_error=np.linalg.norm(empirical - cov) / np.linalg.norm(cov), correlation_error=np.linalg.norm(correlation(empirical) - correlation(cov)) / np.linalg.norm(correlation(cov)), portfolio_ks=ks_2samp(p, q).statistic, portfolio_population_ks=kstest(p, lambda v: pop.cdf(v, None)).statistic, portfolio_w1=wasserstein_distance(p, q), sliced_w1=sw, mmd2_unbiased=mmd, mmd_bandwidth=np.sqrt(bandwidth2), diversity_trace_ratio=np.trace(empirical) / np.trace(cov))
    out['marginal_pit_ks_mean'], out['marginal_pit_ks_std'] = pop.evaluate_generated_samples(x)
    for quant in (0.01, 0.05, 0.5, 0.95, 0.99):
        out[f'portfolio_q{quant}'] = np.quantile(p, quant)
        target_p = quant
        out[f'portfolio_true_q{quant}'] = pop.threshold(target_p)
    for tail in (0.1, 0.05, 0.01):
        prob = np.mean(p <= pop.threshold(tail))
        out[f'event_{tail}'] = prob
        out[f'event_{tail}_se'] = np.sqrt(prob * (1 - prob) / len(p))
    return out

def plot_comparison(x, truth, pop, c, path, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    p = x @ pop.w
    q = truth @ pop.w
    for v, label in ((q, 'Truth'), (p, 'Model (all outputs)')):
        ax[0, 0].plot(np.sort(v), np.arange(1, len(v) + 1) / len(v), label=label)
    grid = np.linspace(min(p.min(), q.min()), max(p.max(), q.max()), 400)
    ax[0, 0].plot(grid, pop.cdf(grid, None), 'k--', label='Exact CDF')
    ax[0, 0].legend()
    ax[0, 0].set_title('Portfolio CDF')
    quant = np.linspace(0.005, 0.995, 100)
    ax[0, 1].scatter(np.quantile(q, quant), np.quantile(p, quant), s=8)
    limits = [min(p.min(), q.min()), max(p.max(), q.max())]
    ax[0, 1].plot(limits, limits, 'k--')
    ax[0, 1].set(xlabel='Truth portfolio quantile', ylabel='Model quantile', title='QQ: all outputs')
    bins = np.histogram_bin_edges(np.concatenate([truth[:, 0], x[:, 0]]), bins=35)
    ax[1, 0].hist(truth[:, 0], bins=bins, density=True, histtype='step', linewidth=1.7, label='Truth')
    ax[1, 0].hist(x[:, 0], bins=bins, density=True, histtype='step', linewidth=1.7, label='DFM')
    ax[1, 0].legend()
    ax[1, 0].set(title='Asset 0 return distribution', xlabel='Return', ylabel='Density')
    tx = truth @ pop.B[:, :2]
    gx = x @ pop.B[:, :2]
    ax[1, 1].scatter(tx[:, 0], tx[:, 1], s=5, alpha=0.2, label='Truth')
    colors = 'tab:orange'
    ax[1, 1].scatter(gx[:, 0], gx[:, 1], s=5, alpha=0.3, c=colors, label='Model')
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title('Two population loading projections (evaluation only)')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

def plot_dfm_validation(folder, case):
    """Display the saved validation curve for one unconditional case."""
    pd.read_csv(folder / "dfm_curve.csv").set_index("epoch")[["validation"]].plot(title=f"{case}: DFM validation loss", ylabel="Denoising score-matching loss")
    plt.tight_layout()
    plt.show()


def unconditional_recovery_report(b, u0, config):
    """Markdown summary of unconditional recovery against the Monte Carlo baseline."""
    text = f"""
1. **Unconditional recovery:** the first Gaussian training seed has covariance error {b.covariance_error:.3f}, portfolio KS {b.portfolio_ks:.3f} and sliced W1 {b.sliced_w1:.3f}. Matched true-versus-true values are {u0.covariance_error:.3f}, {u0.portfolio_ks:.3f}, {u0.sliced_w1:.3f}. For this execution, these gaps are {"substantially above" if b.covariance_error > 2*u0.covariance_error else "near"} the Monte Carlo covariance baseline. {"The unconditional joint law is not recovered accurately at this budget." if b.covariance_error > 2*u0.covariance_error else "The remaining metrics and seed variability must also be assessed before claiming recovery."}

Executed mode: **{config.mode}**, dimensions **{config.d} × {config.k}**, training seeds **{config.seeds}**, samples per comparison **{config.n_eval}**. One-seed smoke results do not estimate training variability.
"""
    return text


def plot_h_validation(h_curve, selected_epoch, output, *, title="H validation loss", show=True):
    """Save validation loss, marking the selected H epoch, and optionally display."""
    # Plot validation loss only; retain training losses in the saved CSV.
    ax = h_curve.set_index("epoch")[["validation"]].plot(
        title=title, ylabel="Configured time-weighted probability loss",
    )
    ax.axvline(selected_epoch, color="gray", linestyle="--", label="Selected epoch")
    ax.legend()
    fig = ax.figure
    fig.tight_layout()
    fig.savefig(output / "validation_loss.png", dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig




PORTFOLIO_CONVENTION = 'CPU NumPy float64 matrix-vector product; portfolio <= threshold, no tolerance'
MATCHING_RULE = 'Fixed seeded permutation per source; first n indices, without replacement, independent of values'

# The four sources of conditional returns compared by the benchmark, in display order.
SOURCES = ('truth', 'cdg_ml', 'rejection', 'refined_exact')
SOURCE_LABELS = {
    'truth': 'Conditional ground truth',
    'cdg_ml': 'CDG-ML',
    'rejection': 'DFM rejection sampling',
    'refined_exact': 'Refined CDG-ML + exact final step',
}
SOURCE_COLORS = {'truth': '#52514d', 'cdg_ml': '#eb6834', 'rejection': '#1baf7a', 'refined_exact': '#2a78d6'}
MODELS = ('cdg_ml', 'refined_exact')


def portfolio_values(samples, weights):
    """Common evaluation/filter convention, independent of sampler dtype/device."""
    if np.iscomplexobj(samples) or np.iscomplexobj(weights):
        raise ValueError('Portfolio samples and weights must be real')
    x = np.asarray(samples, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or w.shape != (x.shape[1],) or not np.isfinite(w).all() or not np.any(w):
        raise ValueError('Require a sample matrix and finite nonzero event weights')
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite evaluation samples')
    with np.errstate(over='ignore', invalid='ignore'):
        values = x @ w
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite portfolio values')
    return values


def conditional_comparisons(samples, weights, *, projection_seed, matching_seed, projections=64,
                            models=MODELS):
    """Sample-to-sample conditional distances with matched sample sizes.

    samples maps 'truth', 'truth_reference', 'rejection', 'rejection_reference'
    and every name in ``models`` to [n, d] arrays. Each source gets one
    independent permutation; a comparison of n rows uses the first n indices of
    both permutations, so the Monte Carlo rows (truth vs independent truth and
    rejection vs independent rejection) are computed at exactly the sample
    sizes of the comparisons. Insufficient references are reported as
    unavailable, never silently downsized. Returns (metrics, directions).
    """
    names = (*models, 'rejection', 'truth', 'truth_reference', 'rejection_reference')
    arrays = {name: np.asarray(samples[name], dtype=np.float64) for name in names}
    w = np.asarray(weights, dtype=np.float64)
    for array in arrays.values():
        portfolio_values(array, w)  # Validate even empty inputs.
    if isinstance(projections, bool) or not isinstance(projections, (int, np.integer)) or projections <= 0:
        raise ValueError('projections must be a positive integer')
    rng = np.random.default_rng(projection_seed)
    directions = rng.normal(size=(projections, len(w)))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    streams = np.random.SeedSequence(matching_seed).spawn(len(names))
    orders = {name: np.random.default_rng(stream).permutation(len(arrays[name]))
              for name, stream in zip(names, streams)}
    projected = {name: array @ directions.T for name, array in arrays.items()}
    portfolios = {name: portfolio_values(array, w) for name, array in arrays.items()}

    def row(left, right, n, kind):
        available = n <= min(len(arrays[left]), len(arrays[right]))
        used = n if available else 0
        status = ('reference_shortfall' if not available else
                  'empty' if n == 0 else 'singleton' if n == 1 else 'ok')
        result = dict(left=left, right=right, row_type=kind, n_left=len(arrays[left]),
                      n_right=len(arrays[right]), target_n=n, n_used=used, status=status,
                      portfolio_w1=np.nan, portfolio_ks=np.nan, sliced_w1=np.nan)
        if used:
            a, b = orders[left][:used], orders[right][:used]
            p, q = portfolios[left][a], portfolios[right][b]
            result.update(portfolio_w1=float(wasserstein_distance(p, q)),
                          portfolio_ks=float(ks_2samp(p, q).statistic),
                          sliced_w1=float(np.mean([
                              wasserstein_distance(projected[left][a, j], projected[right][b, j])
                              for j in range(projections)])))
        return result

    pairs = [(m, reference) for m in models for reference in ('rejection', 'truth')] + [('rejection', 'truth')]
    rows = [row(a, b, min(len(arrays[a]), len(arrays[b])), 'comparison') for a, b in pairs]
    for n in sorted({r['n_used'] for r in rows}):
        rows.append(row('truth', 'truth_reference', n, 'MC reference'))
        rows.append(row('rejection', 'rejection_reference', n, 'MC reference'))
    return pd.DataFrame(rows), directions


def conditional_moment_table(samples, population, threshold):
    """Errors of the sample mean and covariance against the exact conditional moments."""
    mean, covariance = population.moments(c=threshold)
    scale = np.sqrt(np.trace(population.moments()[1]))
    rows = []
    for source, values in samples.items():
        row = dict(source=source, n=len(values), mean_error=np.nan, covariance_error=np.nan,
                   diversity_trace_ratio=np.nan, analytic_portfolio_ks=np.nan)
        if len(values) > 1:
            values = np.asarray(values, dtype=np.float64)
            sample_covariance = np.cov(values, rowvar=False)
            row.update(
                mean_error=float(np.linalg.norm(values.mean(0) - mean) / scale),
                covariance_error=float(np.linalg.norm(sample_covariance - covariance) / np.linalg.norm(covariance)),
                diversity_trace_ratio=float(np.trace(sample_covariance) / np.trace(covariance)),
                analytic_portfolio_ks=float(kstest(portfolio_values(values, population.w), population.cdf,
                                                   args=(threshold,)).statistic))
        rows.append(row)
    return pd.DataFrame(rows)


def conditional_summary(samples, comparisons, moments, threshold, weights):
    """One column per source, one row per metric.

    The distance of a source to its own kind is a Monte Carlo level: DFM
    rejection 'vs DFM rejection' compares two independent rejection streams, and
    the ground truth 'vs ground truth' compares two independent truth draws, at
    the sample size used by the models' comparisons.
    """
    comparisons = comparisons.set_index(['left', 'right'])
    moments = moments.set_index('source')
    model_n = max(len(samples[m]) for m in MODELS)

    def lookup(left, right, n=None):
        rows = comparisons.loc[[(left, right)]]
        if n is not None:
            rows = rows[rows.n_used == min(n, rows.n_used.max())]
        return rows.iloc[0]

    pairs = {('truth', 'rejection'): ('rejection', 'truth'), ('truth', 'truth'): ('truth', 'truth_reference', model_n),
             ('rejection', 'rejection'): ('rejection', 'rejection_reference', len(samples['rejection'])),
             ('rejection', 'truth'): ('rejection', 'truth')}
    table = {}
    for source in SOURCES:
        p = portfolio_values(samples[source], weights)
        column = {'Rows': len(p), 'Event rate': float(np.mean(p <= threshold))}
        for reference, label in (('rejection', 'DFM rejection'), ('truth', 'ground truth')):
            values = lookup(*pairs.get((source, reference), (source, reference)))
            for metric, name in (('portfolio_ks', 'Portfolio KS'), ('portfolio_w1', 'Portfolio W1'),
                                 ('sliced_w1', 'Sliced W1')):
                column[f'{name} vs {label}'] = float(values[metric])
            column[f'Rows compared vs {label}'] = int(values.n_used)
        column['Analytic portfolio KS'] = float(moments.loc[source, 'analytic_portfolio_ks'])
        column['Mean error'] = float(moments.loc[source, 'mean_error'])
        column['Covariance error'] = float(moments.loc[source, 'covariance_error'])
        table[SOURCE_LABELS[source]] = column
    return pd.DataFrame(table)


def plot_conditional_performance(samples, population, threshold, output, *, title=None, show=True):
    """Portfolio CDF, portfolio Q–Q against the ground truth, and two loading projections."""
    portfolios = {s: portfolio_values(samples[s], population.w) for s in SOURCES}
    figure = plt.figure(figsize=(13, 10))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.15, 1))
    cdf_ax, qq_ax, projection_ax = (figure.add_subplot(grid[0, :]), figure.add_subplot(grid[1, 0]),
                                    figure.add_subplot(grid[1, 1]))
    values = np.concatenate(list(portfolios.values()))
    lo, hi = min(threshold, values.min()), max(threshold, values.max())
    pad = max((hi - lo) * .035, .001)
    for source in SOURCES:
        ordered = np.sort(portfolios[source])
        cdf_ax.step(np.r_[lo - pad, ordered, hi + pad], np.r_[0., np.arange(1, len(ordered) + 1) / len(ordered), 1.],
                    where='post', color=SOURCE_COLORS[source], linewidth=1.8,
                    label=f'{SOURCE_LABELS[source]} (n={len(ordered):,})')
    x = np.linspace(lo - pad, hi + pad, 600)
    cdf_ax.plot(x, population.cdf(x, c=threshold), 'k--', linewidth=1.2, label='Exact conditional CDF')
    cdf_ax.axvline(threshold, color='#9b9a96', linestyle=':', label='Event threshold C')
    cdf_ax.set(title='Conditional portfolio CDF', xlim=(lo - pad, hi + pad), ylim=(-.02, 1.02),
               xlabel='Portfolio return', ylabel='Cumulative probability')
    cdf_ax.legend(fontsize=8, loc='lower right', frameon=False)

    quantiles = np.linspace(.005, .995, 100)
    truth_q = np.quantile(portfolios['truth'], quantiles)
    for source, marker in (('cdg_ml', 'o'), ('rejection', 'x'), ('refined_exact', '^')):
        qq_ax.scatter(truth_q, np.quantile(portfolios[source], quantiles), s=16, marker=marker,
                      color=SOURCE_COLORS[source], label=SOURCE_LABELS[source])
    span = np.r_[truth_q, *(np.quantile(portfolios[s], quantiles) for s in MODELS)]
    qq_ax.plot([span.min(), span.max()], [span.min(), span.max()], 'k--', linewidth=1, label='Perfect agreement')
    qq_ax.axhline(threshold, color='#9b9a96', linestyle=':')
    qq_ax.set(title='Portfolio Q–Q against the conditional ground truth', xlabel='Ground-truth quantile',
              ylabel='Sample quantile')
    qq_ax.legend(fontsize=8, frameon=False)

    for source, alpha in (('truth', .25), ('cdg_ml', .25), ('refined_exact', .25)):
        projection = np.asarray(samples[source], dtype=np.float64) @ population.B[:, :2]
        projection_ax.scatter(projection[:, 0], projection[:, 1], s=4, alpha=alpha, color=SOURCE_COLORS[source],
                              label=SOURCE_LABELS[source])
    projection_ax.set(title='Two population loading projections', xlabel='Projection on loading 1',
                      ylabel='Projection on loading 2')
    projection_ax.legend(fontsize=8, frameon=False, markerscale=3)
    for ax in (cdf_ax, qq_ax, projection_ax):
        ax.grid(alpha=.2)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    figure.suptitle(title or 'Conditional distribution performance')
    figure.tight_layout()
    path = Path(output) / 'conditional_performance.png'
    figure.savefig(path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return path
