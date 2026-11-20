"""Figures and readable tables of the market benchmark.

Market returns have no exact law. The unconditional reference is the hold-out
market days; the conditional ones are the hold-out market days in the event and
DFM rejection, the law guidance targets. The tables are those of the synthetic
benchmark (synthetic_benchmark.reporting), with the market days in place of the
exact law and noise levels from bootstrap samples of market days.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import ks_2samp, kurtosis, wasserstein_distance

from synthetic_benchmark.reporting import (
    CONDITIONAL_VALUE_ROWS, DFM_COLORS, DFM_LABELS, DFM_VALUE_ROWS, INK, INK_SECONDARY, MODELS,
    MUTED, NOISE_LEVEL, QUANTILE_LEVELS, SHORT_LABELS, SOURCE_COLORS, SOURCE_LABELS, VERDICT, _distance_table,
    _finish_figure, _percent_axes, _style, conditional_values, matched_rows, portfolio_values,
    portfolio_value_text, random_directions, rejection_label, rejection_table, rejection_values,
)

MARKET_LABEL = 'Hold-out market days'
MARKET_EVENT_LABEL = 'Hold-out market days in the event'
MARKET_COLOR = SOURCE_COLORS['truth']
# Fewer hold-out event days than this are not drawn as a Q-Q line.
MIN_QQ_ROWS = 20
# The distances of synthetic_benchmark.reporting.DISTANCE_LABELS, to the hold-out days instead of the exact law.
DFM_DISTANCE_LABELS = {'portfolio_ks': 'Portfolio: KS distance to the hold-out days',
                       'sliced_w1': 'Joint law: sliced W1 distance to the hold-out days',
                       'covariance_error': 'Covariance: relative error to the hold-out covariance'}
EVENT_DISTANCE_LABELS = {'portfolio_ks': 'Portfolio: KS distance to the hold-out days in the event',
                         'sliced_w1': 'Joint law: sliced W1 distance to the hold-out days in the event',
                         'covariance_error': 'Covariance: relative error to the hold-out covariance in the event'}
# The value rows of the synthetic notebook, then the excess kurtosis of the portfolio return, which shows
# whether the heavy tails of market returns are learned.
MARKET_VALUE_ROWS = (*DFM_VALUE_ROWS, 'Excess kurtosis of the portfolio return')


def market_value_text(p):
    """The MARKET_VALUE_ROWS of a sample of portfolio returns p, formatted."""
    return [*portfolio_value_text(p), f'{kurtosis(np.asarray(p, dtype=np.float64)):.2f}']


def leading_direction(returns, weights):
    """Unit first principal direction of the standardized training returns, oriented along the portfolio."""
    returns = np.asarray(returns, dtype=np.float64)
    scale = returns.std(0)
    direction = np.linalg.eigh(np.corrcoef(returns, rowvar=False))[1][:, -1] / scale
    direction /= np.linalg.norm(direction)
    return direction if direction @ np.asarray(weights) >= 0 else -direction


def _qq_reference(ax, reference, lines, title, reference_label, percent):
    """Sample quantiles against those of a reference sample, the diagonal y = x."""
    ax.plot(reference[[0, -1]], reference[[0, -1]], color=INK, linewidth=1.2, label=reference_label)
    for label, color, values, style in lines:
        ax.plot(reference, values, color=color, linewidth=1.8, linestyle=style, label=label)
    _style(ax, title, f'{reference_label}: quantile', 'Sample quantile')
    _percent_axes(ax, percent)


def sample_distances(x, reference, weights, directions):
    """Portfolio KS, sliced W1 and relative covariance error of a sample against a reference sample."""
    x, reference = np.asarray(x, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    projected_x, projected_reference = x @ directions.T, reference @ directions.T
    covariance = np.cov(reference, rowvar=False)
    return dict(
        portfolio_ks=float(ks_2samp(portfolio_values(x, weights), portfolio_values(reference, weights)).statistic),
        sliced_w1=float(np.mean([wasserstein_distance(projected_x[:, j], projected_reference[:, j])
                                 for j in range(len(directions))])),
        covariance_error=float(np.linalg.norm(np.cov(x, rowvar=False) - covariance) / np.linalg.norm(covariance)))


def bootstrap_noise_levels(pool, n, weights, directions, *, replicates, seed, level=NOISE_LEVEL):
    """The level-quantile of each distance between two bootstrap samples of n market days.

    pool holds the market days of the compared law: every day, or every day in
    the event. Each replicate draws two samples of n days with replacement, the
    first playing the compared source and the second the hold-out days. A value at
    or below this level is consistent with the sampling variation between two
    market samples of that size.
    """
    if isinstance(replicates, bool) or not isinstance(replicates, (int, np.integer)) or replicates <= 0:
        raise ValueError('replicates must be a positive integer')
    pool = np.asarray(pool, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = [sample_distances(pool[rng.integers(len(pool), size=n)], pool[rng.integers(len(pool), size=n)],
                              weights, directions) for _ in range(replicates)]
    return {key: float(np.quantile([draw[key] for draw in draws], level)) for key in DFM_DISTANCE_LABELS}


def market_dfm_tables(generated, evaluation, insample, weights, *, verdict='regime_mlp', projection_seed=781,
                      projections=64, matching_seed=42010, noise_seed=42012, replicates=100):
    """Readable unconditional tables: values beside the hold-out market days, then distances beside their noise level.

    The market counterpart of synthetic_benchmark.reporting.dfm_tables. generated
    maps a DFM network name to its samples; values use every row. Each distance
    compares n rows of a DFM with n hold-out days, n being the smallest sample,
    chosen by seeded permutations. The noise level is
    bootstrap_noise_levels over every market day (``insample``, the training and
    validation days, and the hold-out days), the law the hold-out days are drawn from.
    """
    networks = [n for n in DFM_LABELS if n in generated]
    if verdict not in networks:
        raise ValueError(f'verdict must be one of {networks}')
    samples = {MARKET_LABEL: evaluation, **{DFM_LABELS[n]: generated[n] for n in networks}}
    values = pd.DataFrame({label: market_value_text(portfolio_values(sample, weights))
                           for label, sample in samples.items()}, index=list(MARKET_VALUE_ROWS))
    n = min(len(evaluation), *(len(generated[network]) for network in networks))
    streams = np.random.SeedSequence(matching_seed).spawn(len(networks) + 1)
    hold_out = np.asarray(evaluation, dtype=np.float64)[np.random.default_rng(streams[0]).permutation(len(evaluation))[:n]]
    directions = random_directions(projection_seed, projections, len(weights))
    results = {}
    for stream, network in zip(streams[1:], networks):
        rows = np.random.default_rng(stream).permutation(len(generated[network]))[:n]
        results[network] = sample_distances(np.asarray(generated[network])[rows], hold_out, weights, directions)
    noise = bootstrap_noise_levels(np.concatenate((insample, evaluation)), n, weights, directions,
                                   replicates=replicates, seed=noise_seed)
    return values, _distance_table(results, noise, n, verdict, labels=DFM_DISTANCE_LABELS)


def noise_rows(table):
    """The number of rows n in the header of the noise-level column of a readable distance table."""
    column = next(column for column in table.columns if column.startswith('Noise level'))
    return int(column.rsplit('n=', 1)[1].rstrip(')').replace(',', ''))


def market_table_notes(n, tail=None):
    """Two short reading guides, shown above the value and distance tables (synthetic_benchmark.reporting.table_notes)."""
    if tail is None:
        title, reference, pool = 'Unconditional returns', 'the hold-out market days', 'every market day'
    else:
        title, reference, pool = f'{tail:.0%} event', 'the hold-out market days in the event', 'every market day in the event'
    return (f'**{title}: values next to {reference}.**',
            f'**{title}: distances to {reference} (lower is better).** Each source uses {n:,} rows. The noise level '
            f'is the {NOISE_LEVEL:.0%} quantile of the same distance between two bootstrap samples of {n:,} days '
            f'drawn from {pool} (training, validation and hold-out): a value at or below it cannot be told apart '
            'from the sampling variation between two market samples.')


def plot_market_dfm_distributions(generated, evaluation, weights, direction, path, *, show=True):
    """Q–Q plots of each DFM against the hold-out market days: the portfolio and the leading direction."""
    networks = [n for n in DFM_LABELS if n in generated]
    figure, (portfolio_ax, direction_ax) = plt.subplots(1, 2, figsize=(11, 4.8))
    figure.patch.set_facecolor('white')
    evaluation = np.asarray(evaluation, dtype=np.float64)
    _qq_reference(portfolio_ax, np.quantile(portfolio_values(evaluation, weights), QUANTILE_LEVELS),
                  [(DFM_LABELS[n], DFM_COLORS[n],
                    np.quantile(portfolio_values(generated[n], weights), QUANTILE_LEVELS), '-') for n in networks],
                  'Equal-weight portfolio return: Q–Q', MARKET_LABEL, True)
    _qq_reference(direction_ax, np.quantile(evaluation @ direction, QUANTILE_LEVELS),
                  [(DFM_LABELS[n], DFM_COLORS[n],
                    np.quantile(np.asarray(generated[n], dtype=np.float64) @ direction, QUANTILE_LEVELS), '-')
                   for n in networks],
                  'Leading direction of the training returns: Q–Q', MARKET_LABEL, False)
    return _finish_figure(figure, path, None, show)



# Conditional returns.
def market_conditional_tables(samples, weights, threshold, *, pool, directions, matching_seed, noise_seed,
                              market_frequency, dfm_frequency, sources=('rejection', *MODELS), verdict=VERDICT,
                              replicates=100):
    """Readable conditional tables: values beside the hold-out market days in the event, then distances.

    The market counterpart of synthetic_benchmark.reporting.conditional_tables.
    samples holds 'market', the hold-out days in the event, and every source.
    Values use every row; market_frequency and dfm_frequency are the event
    frequencies without conditioning of the hold-out days and of the DFM (the
    acceptance of DFM rejection). Distances use the same number n of rows for
    every source and for the market days, the smallest sample, taken with the
    seeded permutations of conditional_comparisons (the market days in the role of
    the ground truth); the noise level is bootstrap_noise_levels over ``pool``,
    every market day in the event.
    """
    sources = tuple(sources)
    if verdict not in sources:
        raise ValueError(f'verdict must be one of {sources}')
    values = {MARKET_EVENT_LABEL: conditional_values(samples['market'], weights, threshold, market_frequency),
              **{SHORT_LABELS[s]: conditional_values(samples[s], weights, threshold,
                                                     dfm_frequency if s == 'rejection' else None) for s in sources}}
    values = pd.DataFrame(values, index=list(CONDITIONAL_VALUE_ROWS))
    n = min(len(samples['market']), *(len(samples[s]) for s in sources))
    market = matched_rows(samples['market'], 'truth', n, matching_seed)
    results = {s: sample_distances(matched_rows(samples[s], s, n, matching_seed), market, weights, directions)
               for s in sources}
    noise = bootstrap_noise_levels(pool, n, weights, directions, replicates=replicates, seed=noise_seed)
    return values, _distance_table(results, noise, n, verdict, labels=EVENT_DISTANCE_LABELS)


def market_rejection_table(rejections, frequencies, market, pool, weights, *, market_frequency, directions,
                           matching_seed, noise_seed, replicates=100):
    """DFM rejection of several frozen DFMs beside the hold-out market days in the event, as text.

    rejections and frequencies map a DFM network name to its DFM rejection sample
    and to its event frequency without conditioning. Each distance compares n rows
    of a DFM rejection sample with n hold-out days in the event, n being the
    smallest sample, all chosen by the matched permutations of market_conditional_tables,
    whose noise level (over ``pool``) it shares.
    """
    networks = [network for network in DFM_LABELS if network in rejections]
    n = min(len(market), *(len(rejections[network]) for network in networks))
    hold_out = matched_rows(market, 'truth', n, matching_seed)
    values = {MARKET_EVENT_LABEL: rejection_values(market, weights, market_frequency),
              **{rejection_label(network): rejection_values(rejections[network], weights, frequencies[network])
                 for network in networks}}
    distances = {rejection_label(network): sample_distances(
        matched_rows(rejections[network], 'rejection', n, matching_seed), hold_out, weights, directions)
        for network in networks}
    noise = bootstrap_noise_levels(pool, n, weights, directions, replicates=replicates, seed=noise_seed)
    return rejection_table(values, distances, noise, n, labels=EVENT_DISTANCE_LABELS)


def market_rejection_note(table, tail):
    """Reading note of market_rejection_table."""
    n = noise_rows(table)
    return (f'**{tail:.0%} event: DFM rejection of each frozen DFM next to the hold-out market days in the '
            'event.** This is the error of the DFM itself, before any conditioning method. Distances (lower is '
            f'better) use {n:,} rows of each sample; the noise level is the {NOISE_LEVEL:.0%} quantile of the same '
            f'distance between two bootstrap samples of {n:,} days drawn from every market day in the event.')


def plot_market_conditional(samples, weights, threshold, direction, output, *, sources, title=None, show=True):
    """Three plots of the returns given the event, each against DFM rejection, the law guidance targets.

    The portfolio CDF, the portfolio Q–Q plot and the Q–Q plot along the
    leading direction of the training returns, which checks the joint law
    beyond the portfolio. ``sources`` are the sampled models drawn; the
    hold-out market days in the event are dashed.
    """
    sources = tuple(sources)
    drawn = {s: np.asarray(samples[s], dtype=np.float64) for s in ('rejection', 'market', *sources)}
    portfolios = {s: portfolio_values(x, weights) for s, x in drawn.items()}
    figure, (cdf_ax, portfolio_ax, direction_ax) = plt.subplots(1, 3, figsize=(15, 4.9))
    figure.patch.set_facecolor('white')
    styles = {'rejection': (INK, '-', 1.2, SOURCE_LABELS['rejection']),
              'market': (MARKET_COLOR, '--', 1.4, MARKET_EVENT_LABEL),
              **{s: (SOURCE_COLORS[s], '-', 1.8, SOURCE_LABELS[s]) for s in sources}}
    lo = min(np.quantile(p, .002) for p in portfolios.values() if len(p))
    for s, p in portfolios.items():
        if len(p):
            color, style, width, label = styles[s]
            ordered = np.sort(p)
            cdf_ax.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered), where='post', linewidth=width,
                        color=color, linestyle=style, label=label)
    cdf_ax.axvline(threshold, color=MUTED, linewidth=.9, linestyle='-.', label='Event threshold C')
    cdf_ax.set_xlim(lo, threshold + .03 * (threshold - lo))
    _style(cdf_ax, 'Portfolio return given the event: CDF', 'Portfolio return', 'Cumulative probability')
    cdf_ax.xaxis.set_major_formatter(PercentFormatter(1., decimals=1))
    lines = [s for s in ('market', *sources) if s != 'market' or len(portfolios[s]) >= MIN_QQ_ROWS]
    _qq_reference(portfolio_ax, np.quantile(portfolios['rejection'], QUANTILE_LEVELS),
                  [(styles[s][3], styles[s][0], np.quantile(portfolios[s], QUANTILE_LEVELS), styles[s][1])
                   for s in lines], 'Portfolio return given the event: Q–Q', SOURCE_LABELS['rejection'], True)
    _qq_reference(direction_ax, np.quantile(drawn['rejection'] @ direction, QUANTILE_LEVELS),
                  [(styles[s][3], styles[s][0], np.quantile(drawn[s] @ direction, QUANTILE_LEVELS), styles[s][1])
                   for s in lines], 'Leading direction given the event: Q–Q', SOURCE_LABELS['rejection'], False)
    # Two legend rows: up to seven sources do not fit on one.
    handles = {}
    for ax in figure.axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    figure.legend(list(handles.values()), list(handles), loc='lower center', ncol=4, frameon=False, fontsize=9)
    if title:
        figure.suptitle(title, color=INK_SECONDARY, fontsize=10, x=.01, ha='left')
    figure.tight_layout(rect=(0, .13, 1, .94 if title else 1))
    path = Path(output) / 'conditional_distributions.png'
    figure.savefig(path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return path
