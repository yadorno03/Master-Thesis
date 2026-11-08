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

DFM_LABELS = {'mlp': 'DFM (MLP)', 'gaussian_mlp': 'DFM (Gaussian + MLP)'}
# Readable names for the unconditional metrics shown in the notebook.
UNCONDITIONAL_METRIC_LABELS = {
    'covariance_error': 'Covariance error', 'sliced_w1': 'Sliced W1', 'portfolio_ks': 'Portfolio KS',
    'leading_factor_variance_ratio': 'Largest factor: variance ratio',
    'factor_variance_ratio_min': 'All factors: smallest variance ratio',
    'factor_variance_ratio_max': 'All factors: largest variance ratio',
    'residual_variance_ratio': 'Residual variance ratio',
    'event_0.05': 'Share below the 5% quantile', 'event_0.01': 'Share below the 1% quantile',
    'fit_seconds': 'Training time (s)',
}
TRUTH_REFERENCE_LABEL = 'Independent truth draw (noise)'
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
TRUTH_COLOR = '#52514d'


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


def plot_dfm_comparison(generated, truth_reference, pop, path, *, show=True):
    """The two DFMs against the truth: factor variances, the equal-weight portfolio, its tail frequencies.

    generated maps a DFM network name ('mlp', 'gaussian_mlp') to its
    samples; truth_reference is an independent population draw of the same size,
    which shows the Monte Carlo noise of each measure.
    """
    networks = [n for n in ('mlp', 'gaussian_mlp') if n in generated]
    figure, (variance_ax, qq_ax, tail_ax) = plt.subplots(1, 3, figsize=(15, 4.8))
    figure.patch.set_facecolor('white')
    bars = [(DFM_LABELS[n], generated[n], DFM_COLORS[n]) for n in networks]
    bars.append((TRUTH_REFERENCE_LABEL, truth_reference, TRUTH_COLOR))
    positions = np.arange(len(factor_variance_ratios(truth_reference, pop)[0]))
    width = .8 / len(bars)
    for j, (name, values, color) in enumerate(bars):
        variance_ax.bar(positions + (j - (len(bars) - 1) / 2) * width, factor_variance_ratios(values, pop)[0],
                        width * .92, color=color, label=name)
    variance_ax.axhline(1., color=INK, linewidth=1, label='Target')
    variance_ax.set_xticks(positions, [str(i + 1) for i in positions])
    _style(variance_ax, 'Variance along each factor (target 1)', 'Factor, by decreasing variance',
           'Generated / true variance')

    quantiles = np.linspace(.005, .995, 100)
    exact = np.array([pop.threshold(q) for q in quantiles])
    for network in networks:
        qq_ax.plot(exact, np.quantile(generated[network] @ pop.w, quantiles), color=DFM_COLORS[network],
                   linewidth=2, label=DFM_LABELS[network])
    qq_ax.plot(exact, np.quantile(truth_reference @ pop.w, quantiles), color=TRUTH_COLOR, linewidth=1.2,
               linestyle=(0, (1, 2)), label=TRUTH_REFERENCE_LABEL)
    qq_ax.plot(exact[[0, -1]], exact[[0, -1]], color=INK, linewidth=1, label='Exact quantiles')
    _style(qq_ax, 'Equal-weight portfolio: Q–Q against the exact law', 'Exact quantile', 'Sample quantile')
    qq_ax.legend(fontsize=8, frameon=False, loc='upper left')

    tails = (.1, .05, .01)
    for j, (name, values, color) in enumerate(bars):
        portfolio = values @ pop.w
        frequencies = [np.mean(portfolio <= pop.threshold(tail)) / tail for tail in tails]
        tail_ax.bar(np.arange(len(tails)) + (j - (len(bars) - 1) / 2) * width, frequencies, width * .92,
                    color=color, label=name)
    tail_ax.axhline(1., color=INK, linewidth=1, label='Target')
    tail_ax.set_xticks(np.arange(len(tails)), [f'{tail:.0%} quantile' for tail in tails])
    _style(tail_ax, 'How often the portfolio falls below its quantile', 'Exact portfolio quantile',
           'Observed / nominal frequency')
    handles, labels = variance_ax.get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False, fontsize=8)
    figure.tight_layout(rect=(0, .07, 1, 1))
    figure.savefig(path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return path


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
# Categorical slots 1-4 in fixed order; the ground truth is neutral ink. Line style is a second cue.
SOURCE_COLORS = {'truth': '#52514d', 'refined_exact': '#2a78d6', 'cdg_ml_exact': '#eb6834',
                 'rejection': '#1baf7a', 'unguided_exact': '#eda100'}
SOURCE_LINESTYLES = {'truth': '-', 'rejection': '-', 'refined_exact': '-', 'cdg_ml_exact': '--',
                     'unguided_exact': ':'}
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


def plot_conditional_performance(samples, population, threshold, output, *, comparisons, title=None, show=True):
    """Six panels: the conditional portfolio law, two distances with their noise levels, two factor coordinates.

    Top: the portfolio CDF given the event against the exact conditional CDF, and
    portfolio Q–Q plots against DFM rejection (the target of guidance) and against
    the ground truth. Bottom: sliced W1 and portfolio KS of the two guided models,
    each beside the Monte Carlo level at the same sample size, and the two leading
    factor coordinates. The no-guidance source is far off the bar scale (its values
    are in the summary table); it appears in the CDF and Q–Q panels.
    """
    portfolios = {s: portfolio_values(samples[s], population.w) for s in SOURCES}

    def style(source):
        return dict(color=SOURCE_COLORS[source], linestyle=SOURCE_LINESTYLES[source], label=SOURCE_LABELS[source])

    def distance(left, right, metric, n=None):
        rows = comparisons[(comparisons.left == left) & (comparisons.right == right)]
        if n is not None:
            rows = rows[rows.n_used == n]
        return float(rows.iloc[0][metric]), int(rows.iloc[0].n_used)

    figure, axes = plt.subplots(2, 3, figsize=(15, 9.4))
    figure.patch.set_facecolor('white')

    ax = axes[0, 0]
    lo = min(np.quantile(portfolios['truth'], .002), np.quantile(portfolios['rejection'], .002))
    grid = np.linspace(lo, threshold, 400)
    ax.plot(grid, population.cdf(grid, c=threshold), color=INK, linewidth=1, label='Exact conditional CDF')
    for source in ('rejection', *MODELS):
        ordered = np.sort(portfolios[source])
        ax.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered), where='post', linewidth=1.8, **style(source))
    ax.axvline(threshold, color=MUTED, linewidth=.9, linestyle='-.', label='Event threshold C')
    ax.set_xlim(lo, threshold + .03 * (threshold - lo))
    _style(ax, 'Portfolio given the event: CDF', 'Portfolio return', 'Cumulative probability')

    quantiles = np.linspace(.005, .995, 120)
    for ax, reference, panel in ((axes[0, 1], 'rejection', 'Q–Q against DFM rejection (the target)'),
                                 (axes[0, 2], 'truth', 'Q–Q against the conditional ground truth')):
        reference_quantiles = np.quantile(portfolios[reference], quantiles)
        for source in SOURCES:
            if source != reference:
                ax.plot(reference_quantiles, np.quantile(portfolios[source], quantiles), linewidth=1.8,
                        **style(source))
        ax.plot(reference_quantiles[[0, -1]], reference_quantiles[[0, -1]], color=MUTED, linewidth=1,
                label='y = x')
        ax.axhline(threshold, color=MUTED, linewidth=.9, linestyle='-.')
        _style(ax, panel, f'{SOURCE_LABELS[reference]}: quantile', 'Source quantile')

    guided = ('refined_exact', 'cdg_ml_exact')
    width = .32
    for ax, metric, panel in ((axes[1, 0], 'sliced_w1', 'Joint law: sliced W1 (lower is better)'),
                              (axes[1, 1], 'portfolio_ks', 'Portfolio: KS distance (lower is better)')):
        sizes = [distance(guided[0], reference, metric)[1] for reference in ('rejection', 'truth')]
        for j, source in enumerate(guided):
            values = [distance(source, reference, metric)[0] for reference in ('rejection', 'truth')]
            ax.bar(np.arange(2) + (j - .5) * width, values, width * .92, color=SOURCE_COLORS[source],
                   label=SOURCE_LABELS[source])
        noise = (distance('rejection', 'rejection_reference', metric, sizes[0])[0],
                 distance('truth', 'truth_reference', metric, sizes[1])[0])
        for i, value in enumerate(noise):
            ax.hlines(value, i - .42, i + .42, color=INK, linewidth=1.4, linestyle=(0, (4, 2)),
                      label='Monte Carlo noise at the same sample size' if i == 0 else None)
        dfm, n_dfm = distance('rejection', 'truth', metric)
        ax.hlines(dfm, 1 - .42, 1 + .42, color=SOURCE_COLORS['rejection'], linewidth=1.6,
                  label='DFM rejection vs ground truth')
        ax.text(.99, .99, f'Green line: DFM rejection vs ground truth, n={n_dfm:,}\n'
                f'(noise at that size: {distance("truth", "truth_reference", metric, n_dfm)[0]:.3f})',
                transform=ax.transAxes, ha='right', va='top', fontsize=7, color=INK_SECONDARY)
        ax.set_xticks([0, 1], [f'vs DFM rejection (n={sizes[0]:,})', f'vs ground truth (n={sizes[1]:,})'])
        ax.set_ylim(0, 1.3 * max(ax.get_ylim()[1], dfm))
        _style(ax, panel, '', {'sliced_w1': 'Sliced W1', 'portfolio_ks': 'Portfolio KS'}[metric])

    ax = axes[1, 2]
    for source in ('rejection', 'refined_exact', 'cdg_ml_exact'):
        coordinates = np.asarray(samples[source], dtype=np.float64) @ population.B[:, :2]
        ax.scatter(coordinates[:, 0], coordinates[:, 1], s=6, alpha=.35, linewidths=0,
                   color=SOURCE_COLORS[source], label=SOURCE_LABELS[source])
    _style(ax, 'Two leading factor coordinates given the event', 'Loading 1', 'Loading 2')

    handles = {}
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    figure.legend(list(handles.values()), list(handles), loc='lower center', ncol=4, frameon=False, fontsize=8)
    figure.suptitle(title or 'Conditional distribution performance', color=INK_SECONDARY, fontsize=10,
                    x=.01, ha='left')
    figure.tight_layout(rect=(0, .08, 1, .97))
    path = Path(output) / 'conditional_performance.png'
    figure.savefig(path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return path
