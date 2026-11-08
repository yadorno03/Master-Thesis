"""Distribution metrics, figures, and the benchmark's written interpretation."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.spatial.distance import cdist, pdist
from scipy.special import log_ndtr, ndtri_exp
from scipy.stats import ks_2samp, wasserstein_distance, kstest, norm

from synthetic_conditional_truth import MARKET_POPULATION_DESIGN

def correlation(c):
    s = np.sqrt(np.diag(c))
    return c / np.outer(s, s)


def random_directions(seed, count, d):
    """Seeded random unit directions in R^d, the projections of the sliced W1 distance."""
    directions = np.random.default_rng(seed).normal(size=(count, d))
    return directions / np.linalg.norm(directions, axis=1)[:, None]


def compare(x, truth, pop, c=None, seed=781):
    """All rows enter every metric. RBF MMD uses at most 512 deterministic rows."""
    mean, cov = pop.moments(None)
    empirical = np.cov(x, rowvar=False)
    scale = np.sqrt(np.trace(pop.moments()[1]) / len(mean))
    p = x @ pop.w
    q = truth @ pop.w
    directions = random_directions(seed, 64, x.shape[1])
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

DFM_LABELS = {'mlp': 'DFM (MLP)', 'gaussian_mlp': 'DFM (Gaussian + MLP)'}
IDENTIFIABLE_RATIO = 5.0


def factor_variance_ratios(x, pop, identifiable_ratio=IDENTIFIABLE_RATIO):
    """Generated over true return variance along the population's factor directions.

    The directions are E = B U, with U the eigenvectors of the factor covariance
    S = U diag(lambda) U^T, by decreasing lambda. Only factors with
    lambda >= identifiable_ratio * mean(D) are kept: smaller ones cannot be told
    apart from the residual noise. Also returns the generated over true residual
    variance, the variance outside span(B). The target of every ratio is 1.
    """
    if len(pop.pi) != 1:
        raise ValueError('Factor variance ratios need a single-component population')
    _, true_covariance = pop.moments()
    eigenvalues, U = np.linalg.eigh(pop.S[0])
    order = eigenvalues.argsort()[::-1]
    eigenvalues, E = eigenvalues[order], pop.B @ U[:, order]
    keep = eigenvalues >= identifiable_ratio * pop.D.mean()
    covariance = np.cov(np.asarray(x, dtype=np.float64), rowvar=False)
    ratios = (np.einsum('dk,de,ek->k', E, covariance, E)[keep]
              / np.einsum('dk,de,ek->k', E, true_covariance, E)[keep])

    def outside(c):
        return np.trace(c) - np.einsum('dk,de,ek->', pop.B, c, pop.B)
    return ratios, float(outside(covariance) / outside(true_covariance))


def factor_variance_summary(x, pop):
    """Scalar summary of factor_variance_ratios for the metrics table."""
    ratios, residual = factor_variance_ratios(x, pop)
    nan = float('nan')
    return dict(identifiable_factors=len(ratios),
                leading_factor_variance_ratio=float(ratios[0]) if len(ratios) else nan,
                factor_variance_ratio_min=float(ratios.min()) if len(ratios) else nan,
                factor_variance_ratio_max=float(ratios.max()) if len(ratios) else nan,
                residual_variance_ratio=residual)


INK, INK_SECONDARY, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
DFM_COLORS = {'gaussian_mlp': '#2a78d6', 'mlp': '#eb6834'}  # Categorical slots 1 and 2.


def _style(ax, title, xlabel, ylabel):
    ax.grid(True, color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#c3c2b7')
    ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, labelsize=8)
    ax.set_title(title, color=INK, fontsize=10, loc='left')
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)


def returns_in_percent(population):
    """Chen et al. (2025) populations hold raw returns, shown in percent; market ones are standardized units."""
    return (population.construction or {}).get('design') != MARKET_POPULATION_DESIGN


def format_return(value, percent):
    return f'{100 * value:.2f}%' if percent else f'{value:.3f}'


def largest_factor_direction(population):
    """Unit return direction B u of the largest factor variance, signed so that w'B u >= 0.

    u is the leading eigenvector of the factor covariance, the first direction of
    factor_variance_ratios. With the positive sign, portfolio losses are also
    negative along this direction.
    """
    m = population.pi @ population.m
    factor_covariance = (np.einsum('j,jik->ik', population.pi, population.S)
                         + np.einsum('j,ji,jk->ik', population.pi, population.m - m, population.m - m))
    direction = population.B @ np.linalg.eigh(factor_covariance)[1][:, -1]
    return direction if direction @ population.w >= 0 else -direction


def exact_portfolio_quantiles(population, probabilities, c=None):
    """Exact quantiles of w'R, or of w'R given w'R <= c.

    Given the event, the q-quantile is the unconditional q P(w'R <= c) quantile.
    """
    scale = 1. if c is None else float(np.exp(population.log_event(c)))
    return np.array([population.threshold(q * scale) for q in np.ravel(probabilities)])


def exact_projection_cdf(population, direction, x, c=None, *, nodes=2000):
    """Exact CDF of a'R, or of a'R given w'R <= c, for a Gaussian factor population.

    Within a component, a'R = mean + beta (w'R - portfolio mean) + independent
    Gaussian noise. Given the event, the expectation over the truncated w'R uses
    midpoint quantile nodes of its exact truncated law.
    """
    a = np.asarray(direction, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    portfolio_mean, portfolio_variance = population.portfolio
    weights = population.pi if c is None else population.component_probabilities(c)
    u = (np.arange(nodes) + .5) / nodes
    cdf = np.zeros(x.shape)
    for j, weight in enumerate(weights):
        covariance_a = population.B @ (population.S[j] @ (population.B.T @ a)) + population.D * a
        mean_a = population.m[j] @ (population.B.T @ a)
        variance_a = a @ covariance_a
        if c is None:
            cdf += weight * norm.cdf((x - mean_a) / np.sqrt(variance_a))
            continue
        sd = np.sqrt(portfolio_variance[j])
        beta = (population.w @ covariance_a) / portfolio_variance[j]
        residual_sd = np.sqrt(max(variance_a - beta ** 2 * portfolio_variance[j], 0.))
        p = portfolio_mean[j] + sd * ndtri_exp(np.log(u) + log_ndtr((c - portfolio_mean[j]) / sd))
        centers = mean_a + beta * (p - portfolio_mean[j])
        if residual_sd > 0:
            cdf += weight * norm.cdf((x[..., None] - centers) / residual_sd).mean(-1)
        else:
            cdf += weight * (centers <= x[..., None]).mean(-1)
    return cdf


def exact_projection_quantiles(population, direction, probabilities, c=None, *, points=2000):
    """Exact quantiles of a'R (optionally given the event), inverting exact_projection_cdf on a fine grid."""
    a = np.asarray(direction, dtype=np.float64)
    means = population.m @ (population.B.T @ a)
    sds = np.sqrt([a @ (population.B @ (S @ (population.B.T @ a)) + population.D * a) for S in population.S])
    grid = np.linspace((means - 10 * sds).min(), (means + 10 * sds).max(), points)
    cdf = np.maximum.accumulate(exact_projection_cdf(population, a, grid, c))
    return np.interp(probabilities, cdf, grid)


EXACT_LABEL = 'Exact law'
QUANTILE_LEVELS = np.linspace(.005, .995, 120)


def _percent_axes(ax, percent):
    if percent:
        ax.xaxis.set_major_formatter(PercentFormatter(1., decimals=1))
        ax.yaxis.set_major_formatter(PercentFormatter(1., decimals=1))


def _qq(ax, exact, lines, title, percent):
    """Sample quantiles against exact ones; the exact law is the diagonal y = x."""
    ax.plot(exact[[0, -1]], exact[[0, -1]], color=INK, linewidth=1.2, label=EXACT_LABEL)
    for label, color, values in lines:
        ax.plot(exact, values, color=color, linewidth=1.8, label=label)
    _style(ax, title, 'Exact quantile', 'Sample quantile')
    _percent_axes(ax, percent)


def _finish_figure(figure, path, title, show):
    """One legend below the panels, labels deduplicated in drawing order."""
    handles = {}
    for ax in figure.axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    figure.legend(list(handles.values()), list(handles), loc='lower center', ncol=len(handles), frameon=False,
                  fontsize=9)
    if title:
        figure.suptitle(title, color=INK_SECONDARY, fontsize=10, x=.01, ha='left')
    figure.tight_layout(rect=(0, .08, 1, .94 if title else 1))
    figure.savefig(path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return path


def plot_dfm_distributions(generated, population, path, *, show=True):
    """Q–Q plots of each DFM against the exact law: the equal-weight portfolio and the largest factor direction.

    generated maps a DFM network name ('mlp', 'gaussian_mlp') to its samples.
    """
    networks = [n for n in DFM_LABELS if n in generated]
    direction = largest_factor_direction(population)
    figure, (portfolio_ax, factor_ax) = plt.subplots(1, 2, figsize=(11, 4.8))
    figure.patch.set_facecolor('white')
    _qq(portfolio_ax, exact_portfolio_quantiles(population, QUANTILE_LEVELS),
        [(DFM_LABELS[n], DFM_COLORS[n], np.quantile(portfolio_values(generated[n], population.w), QUANTILE_LEVELS))
         for n in networks],
        'Equal-weight portfolio return: Q–Q against the exact law', returns_in_percent(population))
    _qq(factor_ax, exact_projection_quantiles(population, direction, QUANTILE_LEVELS),
        [(DFM_LABELS[n], DFM_COLORS[n],
          np.quantile(np.asarray(generated[n], dtype=np.float64) @ direction, QUANTILE_LEVELS)) for n in networks],
        'Largest factor direction: Q–Q against the exact law', False)
    return _finish_figure(figure, path, None, show)


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
    pd.read_csv(folder / "dfm_curve.csv").set_index("epoch")[["validation"]].plot(title=f"{case}: validation loss", ylabel="Denoising score-matching loss")
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

# The sources of conditional returns compared by the benchmark, in display order. The three
# sampled models all draw their last step exactly; they differ by the guidance before it.
SOURCES = ('truth', 'rejection', 'refined_exact', 'cdg_ml_exact', 'unguided_exact')
SOURCE_LABELS = {
    'truth': 'Conditional ground truth',
    'rejection': 'DFM rejection sampling',
    'refined_exact': 'Refined CDG-ML + exact final step',
    'cdg_ml_exact': 'CDG-ML + exact final step',
    'unguided_exact': 'No guidance + exact final step',
}
# Categorical slots 1-4 in fixed order; the ground truth is neutral ink.
SOURCE_COLORS = {'truth': '#52514d', 'refined_exact': '#2a78d6', 'cdg_ml_exact': '#eb6834',
                 'rejection': '#1baf7a', 'unguided_exact': '#eda100'}
MODELS = ('refined_exact', 'cdg_ml_exact', 'unguided_exact')
# The exact final step can place a row on C; float32 sampling then exceeds C by at most about 1e-6.
EVENT_TOLERANCE = 1e-5


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


def matched_orders(lengths, matching_seed, models=MODELS):
    """One fixed seeded permutation per source; a comparison of n rows uses the first n indices.

    lengths maps source names to sample sizes. A source's permutation depends only
    on its position among the comparison names, so tables computed from any subset
    of sources reuse the rows of conditional_comparisons.
    """
    names = (*models, 'rejection', 'truth', 'truth_reference', 'rejection_reference')
    streams = np.random.SeedSequence(matching_seed).spawn(len(names))
    return {name: np.random.default_rng(stream).permutation(lengths[name])
            for name, stream in zip(names, streams) if name in lengths}


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
    directions = random_directions(projection_seed, projections, len(w))
    orders = matched_orders({name: len(array) for name, array in arrays.items()}, matching_seed, models)
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


def conditional_summary(samples, comparisons, moments, threshold, weights, *, near_band=None):
    """One column per source, one row per metric.

    The distance of a source to its own kind is a Monte Carlo level: DFM
    rejection 'vs DFM rejection' compares two independent rejection streams, and
    the ground truth 'vs ground truth' compares two independent truth draws, at
    the sample size used by the models' comparisons.

    The event rate and the rows on the threshold use EVENT_TOLERANCE, so a row the
    exact final step places on C counts as inside the event despite float32
    rounding. With near_band (3 standard deviations of the last step along the
    portfolio), 'Rows just below the threshold' counts the rows within near_band of C: how many rows the
    exact final step pushed just below C, against the share in the ground truth.
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
        column = {'Rows': len(p), 'Event rate': float(np.mean(p <= threshold + EVENT_TOLERANCE)),
                  'Rows on the threshold': float(np.mean(np.abs(p - threshold) <= EVENT_TOLERANCE))}
        if near_band is not None:
            column['Rows just below the threshold'] = float(np.mean(
                (p > threshold - near_band) & (p <= threshold + EVENT_TOLERANCE)))
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


SHORT_LABELS = {'truth': 'Ground truth', 'rejection': 'DFM rejection', 'refined_exact': 'Refined CDG-ML',
                'cdg_ml_exact': 'CDG-ML', 'unguided_exact': 'No guidance', **DFM_LABELS}


def shown_sources(sources):
    """Validate the sources drawn and tabulated beside the exact law; the ground-truth sample is not drawn."""
    sources = tuple(sources)
    if len(set(sources)) != len(sources) or not set(sources).issubset(SOURCES):
        raise ValueError(f'Shown sources must be distinct names from {SOURCES}')
    sources = tuple(s for s in sources if s != 'truth')
    if not sources:
        raise ValueError('Show at least one sampled source besides the ground truth')
    return sources


def plot_conditional_distributions(samples, population, threshold, output, *, sources=('rejection', 'refined_exact'),
                                   title=None, show=True):
    """Three distribution plots of the conditional law, each against the exact law given the event.

    The portfolio CDF, the portfolio Q–Q plot and the Q–Q plot along the largest
    factor direction, which checks the joint law beyond the portfolio. Only
    ``sources`` are drawn; the ground-truth sample is replaced by the exact law.
    """
    sources = shown_sources(sources)
    percent = returns_in_percent(population)
    portfolios = {s: portfolio_values(samples[s], population.w) for s in sources}
    direction = largest_factor_direction(population)
    figure, (cdf_ax, portfolio_ax, factor_ax) = plt.subplots(1, 3, figsize=(15, 4.9))
    figure.patch.set_facecolor('white')

    lo = min(min(np.quantile(p, .002) for p in portfolios.values()),
             exact_portfolio_quantiles(population, [.002], c=threshold)[0])
    grid = np.linspace(lo, threshold, 400)
    cdf_ax.plot(grid, population.cdf(grid, c=threshold), color=INK, linewidth=1.2, label=EXACT_LABEL)
    for s in sources:
        ordered = np.sort(portfolios[s])
        cdf_ax.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered), where='post', linewidth=1.8,
                    color=SOURCE_COLORS[s], label=SOURCE_LABELS[s])
    cdf_ax.axvline(threshold, color=MUTED, linewidth=.9, linestyle='-.', label='Event threshold C')
    cdf_ax.set_xlim(lo, threshold + .03 * (threshold - lo))
    _style(cdf_ax, 'Portfolio return given the event: CDF', 'Portfolio return', 'Cumulative probability')
    if percent:
        cdf_ax.xaxis.set_major_formatter(PercentFormatter(1., decimals=1))

    _qq(portfolio_ax, exact_portfolio_quantiles(population, QUANTILE_LEVELS, c=threshold),
        [(SOURCE_LABELS[s], SOURCE_COLORS[s], np.quantile(portfolios[s], QUANTILE_LEVELS)) for s in sources],
        'Portfolio return given the event: Q–Q', percent)
    _qq(factor_ax, exact_projection_quantiles(population, direction, QUANTILE_LEVELS, c=threshold),
        [(SOURCE_LABELS[s], SOURCE_COLORS[s],
          np.quantile(np.asarray(samples[s], dtype=np.float64) @ direction, QUANTILE_LEVELS)) for s in sources],
        'Largest factor direction given the event: Q–Q', False)
    return _finish_figure(figure, Path(output) / 'conditional_distributions.png', title, show)


# Readable tables: values beside the exact law, then distances beside their noise level.
NOISE_LEVEL = .95
DISTANCE_LABELS = {
    'portfolio_ks': 'Portfolio: KS distance to the exact law',
    'sliced_w1': 'Joint law: sliced W1 distance to the ground truth',
    'covariance_error': 'Covariance: relative error to the exact covariance',
}


def sample_distances(x, truth, population, c, directions, covariance):
    """Portfolio KS to the exact (conditional) CDF, sliced W1 to a truth sample, relative covariance error."""
    x, truth = np.asarray(x, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    projected_x, projected_truth = x @ directions.T, truth @ directions.T
    return dict(
        portfolio_ks=float(kstest(portfolio_values(x, population.w), population.cdf, args=(c,)).statistic),
        sliced_w1=float(np.mean([wasserstein_distance(projected_x[:, j], projected_truth[:, j])
                                 for j in range(len(directions))])),
        covariance_error=float(np.linalg.norm(np.cov(x, rowvar=False) - covariance) / np.linalg.norm(covariance)))


def noise_levels(population, n, c, directions, covariance, *, replicates, seed, level=NOISE_LEVEL):
    """The level-quantile of each distance when the compared sample is itself exact.

    Each replicate draws two independent exact samples of n rows: the first plays
    the compared source, the second the ground truth. A value at or below this
    level is consistent with sampling noise at that sample size.
    """
    if isinstance(replicates, bool) or not isinstance(replicates, (int, np.integer)) or replicates <= 0:
        raise ValueError('replicates must be a positive integer')
    rng = np.random.default_rng(seed)
    draws = [sample_distances(population.sample(n, rng, c=c), population.sample(n, rng, c=c), population, c,
                              directions, covariance) for _ in range(replicates)]
    return {key: float(np.quantile([draw[key] for draw in draws], level)) for key in DISTANCE_LABELS}


def _distance_table(results, noise, n, verdict):
    """Distances as text, the noise level at the same size, and whether ``verdict`` is within it."""
    table = {label: [f'{values[key]:.3f}' for key in DISTANCE_LABELS] for label, values in results.items()}
    table[f'Noise level ({NOISE_LEVEL:.0%}, n={n:,})'] = [f'{noise[key]:.3f}' for key in DISTANCE_LABELS]
    table[f'{SHORT_LABELS[verdict]} within noise?'] = [
        'yes' if results[verdict][key] <= noise[key] else 'no' for key in DISTANCE_LABELS]
    frame = pd.DataFrame(table, index=list(DISTANCE_LABELS.values()))
    return frame.rename(columns={key: SHORT_LABELS[key] for key in results})


def dfm_tables(generated, truth, population, *, verdict='gaussian_mlp', projection_seed=781, projections=64,
               noise_seed=42010, replicates=100):
    """Readable unconditional tables: values beside the exact law, then distances beside their noise level.

    generated maps a DFM network name to samples of the same size as the
    ground-truth sample ``truth``; every row enters every value and distance.
    The projections match the sliced W1 of ``compare`` (seed 781), so the
    distances equal those in metrics.csv.
    """
    networks = [n for n in DFM_LABELS if n in generated]
    if verdict not in networks:
        raise ValueError(f'verdict must be one of {networks}')
    n = len(truth)
    if any(len(generated[network]) != n for network in networks):
        raise ValueError('Each DFM sample must have the size of the ground-truth sample')
    percent = returns_in_percent(population)
    mean, covariance = population.moments()
    w = population.w
    values = {'Exact': [format_return(w @ mean, percent), format_return(np.sqrt(w @ covariance @ w), percent),
                        format_return(population.threshold(.05), percent),
                        format_return(population.threshold(.01), percent), '1.00']}
    for network in networks:
        p = portfolio_values(generated[network], w)
        ratios = factor_variance_ratios(generated[network], population)[0]
        values[DFM_LABELS[network]] = [
            format_return(p.mean(), percent), format_return(p.std(ddof=1), percent),
            format_return(np.quantile(p, .05), percent), format_return(np.quantile(p, .01), percent),
            f'{ratios.min():.2f}–{ratios.max():.2f}' if len(ratios) else '—']
    values = pd.DataFrame(values, index=[
        'Mean portfolio return', 'Portfolio volatility', '5% quantile of the portfolio return',
        '1% quantile of the portfolio return', 'Factor variances relative to the exact ones (min–max)'])
    directions = random_directions(projection_seed, projections, population.d)
    results = {network: sample_distances(generated[network], truth, population, None, directions, covariance)
               for network in networks}
    noise = noise_levels(population, n, None, directions, covariance, replicates=replicates, seed=noise_seed)
    return values, _distance_table(results, noise, n, verdict)


def conditional_tables(samples, population, threshold, *, directions, matching_seed, noise_seed,
                       sources=('rejection', 'refined_exact'), verdict='refined_exact', replicates=100):
    """Readable conditional tables: values beside the exact law, then distances beside their noise level.

    Values use every row of each source. Distances use the same number n of
    rows for every source, the smallest shown sample, taken with the seeded
    permutations of conditional_comparisons, so a single noise level applies.
    """
    sources = shown_sources(sources)
    if verdict not in sources:
        raise ValueError(f'verdict must be one of {sources}')
    percent = returns_in_percent(population)
    mean, covariance = population.moments(c=threshold)
    w = population.w
    values = {'Exact': [format_return(w @ mean, percent), format_return(np.sqrt(w @ covariance @ w), percent),
                        f'{1:.1%}', '—']}
    for s in sources:
        p = portfolio_values(samples[s], w)
        values[SHORT_LABELS[s]] = [format_return(p.mean(), percent), format_return(p.std(ddof=1), percent),
                                    f'{np.mean(p <= threshold + EVENT_TOLERANCE):.1%}', f'{len(p):,}']
    values = pd.DataFrame(values, index=['Expected shortfall: mean return given the event',
                                         'Volatility given the event', 'Rows in the event', 'Rows generated'])
    n = min(len(samples[s]) for s in sources)
    orders = matched_orders({name: len(samples[name]) for name in (*sources, 'truth')}, matching_seed)
    truth = np.asarray(samples['truth'], dtype=np.float64)[orders['truth'][:n]]
    results = {s: sample_distances(np.asarray(samples[s], dtype=np.float64)[orders[s][:n]], truth, population,
                                   threshold, directions, covariance) for s in sources}
    noise = noise_levels(population, n, threshold, directions, covariance, replicates=replicates, seed=noise_seed)
    return values, _distance_table(results, noise, n, verdict)


def table_notes(n, tail=None):
    """Two short reading guides, shown above the value and distance tables."""
    event = f'{tail:.0%} event' if tail is not None else 'Unconditional returns'
    return (f'**{event}: values next to the exact law.**',
            f'**{event}: distances (lower is better).** Each source uses {n:,} rows. The noise level is the '
            f'{NOISE_LEVEL:.0%} quantile of the same distance between exact samples of that size: '
            'a value at or below it cannot be told apart from sampling noise.')
