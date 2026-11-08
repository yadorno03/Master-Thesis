"""Figures and tables for the guided benchmark on the I3 DFM (scripts/i3_conditional.py).

    python scripts/i3_conditional.py report artifacts/i3_conditional/<run>
"""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIGURES = REPO / 'report/figures'
RESULTS = REPO / 'report/results/10_i3_guided_d500'
INK, MUTED, GRID = '#1f1f1e', '#6b6a65', '#e4e2dc'
# Fixed categorical slots of the dataviz palette; truth in neutral ink. Line style is a second encoding.
STYLE = {
    'truth': dict(color='#52514d', ls='-', label='Conditional truth'),
    'rejection': dict(color='#1baf7a', ls='-', label='DFM rejection (target)'),
    'refined_exact': dict(color='#2a78d6', ls='-', label='Refined CDG-ML + exact step'),
    'cdg_ml_exact': dict(color='#eb6834', ls='--', label='CDG-ML + exact step'),
    'unguided_exact': dict(color='#eda100', ls=':', label='No guidance + exact step'),
}
MODELS = ('refined_exact', 'cdg_ml_exact', 'unguided_exact')


def style_axes(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)


def label_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=10, loc='left')
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)


def population():
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / 'scripts'))
    from dfm_upgrades import panel_population_spec
    from i3_conditional import p2_population
    return p2_population(panel_population_spec())


def comparison_value(comparisons, left, right, metric, n=None):
    rows = comparisons[(comparisons.left == left) & (comparisons.right == right)]
    if n is not None:
        rows = rows[rows.n_used == n]
    return float(rows.iloc[0][metric])


def plot_tail(folder, pop):
    data = np.load(folder / 'conditional_samples.npz')
    comparisons = pd.read_csv(folder / 'comparisons.csv')
    run = json.loads((folder / 'run.json').read_text())
    tail, threshold, w = run['tail'], float(data['threshold']), data['weights']
    portfolios = {s: data[s] @ w for s in STYLE}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.4))
    fig.patch.set_facecolor('white')

    # (a) conditional portfolio CDF against the exact one
    ax = axes[0, 0]
    lo = min(np.quantile(portfolios['truth'], 0.002), np.quantile(portfolios['rejection'], 0.002))
    grid = np.linspace(lo, threshold, 400)
    ax.plot(grid, pop.cdf(grid, c=threshold), color=INK, lw=1, ls=(0, (4, 3)), label='Exact conditional CDF')
    for s in ('rejection', *MODELS):
        v = np.sort(portfolios[s])
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where='post', color=STYLE[s]['color'], ls=STYLE[s]['ls'],
                lw=1.8, label=STYLE[s]['label'])
    ax.set_xlim(lo, threshold + 0.03 * (threshold - lo))
    label_axes(ax, 'Equal-weight portfolio given the event: CDF', 'Portfolio return', 'Cumulative probability')

    # (b, c) portfolio QQ against the target (DFM rejection) and against the truth
    q = np.linspace(0.005, 0.995, 120)
    for ax, ref, title in ((axes[0, 1], 'rejection', 'QQ against DFM rejection (the target)'),
                           (axes[0, 2], 'truth', 'QQ against the conditional truth')):
        rq = np.quantile(portfolios[ref], q)
        others = [s for s in ('truth', 'rejection', *MODELS) if s != ref]
        for s in others:
            ax.plot(rq, np.quantile(portfolios[s], q), color=STYLE[s]['color'], ls=STYLE[s]['ls'], lw=1.8,
                    label=STYLE[s]['label'])
        span = [rq.min(), rq.max()]
        ax.plot(span, span, color=INK, lw=1, label='y = x')
        ax.axhline(threshold, color=MUTED, lw=0.8, ls='-.')
        label_axes(ax, title, f"{STYLE[ref]['label']} quantile", 'Source quantile')

    # (d, e) distances with Monte Carlo levels at the same sample size
    for ax, metric, title in ((axes[1, 0], 'sliced_w1', 'Joint distance: sliced W1 (lower is better)'),
                              (axes[1, 1], 'portfolio_ks', 'Portfolio KS (lower is better)')):
        refs = ('rejection', 'truth')
        width = 0.26
        for j, s in enumerate(MODELS):
            vals = [comparison_value(comparisons, s, r, metric) for r in refs]
            ax.bar(np.arange(2) + (j - 1) * width, vals, width * 0.9, color=STYLE[s]['color'],
                   label=STYLE[s]['label'], hatch='//' if s == 'unguided_exact' else None, edgecolor='white')
            for x, v in zip(np.arange(2) + (j - 1) * width, vals):
                ax.annotate(f'{v:.3f}', (x, v), xytext=(0, 2), textcoords='offset points', ha='center',
                            fontsize=7, color=INK)
        n_models = int(comparisons[(comparisons.left == MODELS[0]) & (comparisons.right == 'rejection')].n_used.iloc[0])
        n_truth = int(comparisons[(comparisons.left == MODELS[0]) & (comparisons.right == 'truth')].n_used.iloc[0])
        mc = [comparison_value(comparisons, 'rejection', 'rejection_reference', metric, n_models),
              comparison_value(comparisons, 'truth', 'truth_reference', metric, n_truth)]
        for i, v in enumerate(mc):
            ax.hlines(v, i - 0.45, i + 0.45, color=INK, lw=1.4, ls=(0, (4, 2)),
                      label='Monte Carlo noise (two independent draws)' if i == 0 else None)
        dfm_vs_truth = comparison_value(comparisons, 'rejection', 'truth', metric)
        ax.hlines(dfm_vs_truth, 1 - 0.45, 1 + 0.45, color=STYLE['rejection']['color'], lw=1.6,
                  label='DFM rejection vs truth')
        # DFM rejection has fewer rows than the guided sources: state its own noise level.
        n_rej = int(comparisons[(comparisons.left == 'rejection') & (comparisons.right == 'truth')].n_used.iloc[0])
        mc_rej = comparison_value(comparisons, 'truth', 'truth_reference', metric, n_rej)
        ax.annotate(f'DFM rejection vs truth, n={n_rej:,}\n(noise at that n: {mc_rej:.3f})', (1 - 0.45, dfm_vs_truth),
                    xytext=(0, 3), textcoords='offset points', fontsize=7, color=INK, va='bottom')
        ax.set_xticks([0, 1], [f'vs DFM rejection (n={n_models:,})', f'vs truth (n={n_truth:,})'])
        unguided_max = max(comparison_value(comparisons, 'unguided_exact', r, metric) for r in refs)
        guided_max = max(comparison_value(comparisons, s, r, metric) for s in MODELS[:2] for r in refs)
        top = max(guided_max, max(mc), dfm_vs_truth) * 1.6
        if unguided_max > top:  # keep the guided bars readable; the clipped bar is labelled
            ax.set_ylim(0, top)
            for i, r in enumerate(refs):
                v = comparison_value(comparisons, 'unguided_exact', r, metric)
                ax.annotate(f'{v:.2f} (clipped)', (i + width, top), xytext=(0, -10), textcoords='offset points',
                            ha='center', fontsize=7, color=INK)
        label_axes(ax, title, '', metric.replace('_', ' '))

    # (f) the two leading factor coordinates given the event
    ax = axes[1, 2]
    B = data['B']
    for s, size in (('rejection', 5), ('refined_exact', 5), ('cdg_ml_exact', 5)):
        z = data[s] @ B[:, :2]
        ax.scatter(z[:, 0], z[:, 1], s=size, alpha=0.35, color=STYLE[s]['color'], label=STYLE[s]['label'],
                   linewidths=0)
    label_axes(ax, 'Two leading factor coordinates given the event', 'Loading 1 (market)', 'Loading 2')
    ax.legend(fontsize=7, frameon=False, markerscale=3, loc='best')

    for ax in axes.flat:
        style_axes(ax)
    axes[0, 0].legend(fontsize=7, frameon=False, loc='upper left')
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f'P2, d = {run["d"]}, k = {run["k"]}, frozen I3 DFM; event: equal-weight portfolio below its '
                 f'{tail:.0%} quantile (C = {threshold:.3f}); {len(portfolios["refined_exact"]):,} guided rows, '
                 f'{len(portfolios["rejection"]):,} rejection rows', color=MUTED, fontsize=9, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f'10_i3_guided_tail{int(round(tail * 100))}.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def report(run_dir):
    run_dir = Path(run_dir)
    pop = population()
    RESULTS.mkdir(parents=True, exist_ok=True)
    for folder in sorted(run_dir.glob('tail*')):
        if not (folder / 'summary.csv').exists():
            print('skip (no summary):', folder)
            continue
        tail = json.loads((folder / 'run.json').read_text())['tail']
        tag = f'tail{int(round(tail * 100))}'
        path = plot_tail(folder, pop)
        summary = pd.read_csv(folder / 'summary.csv', index_col=0)
        h = pd.read_csv(folder / 'h_diagnostics.csv', index_col=0)
        timings = json.loads((folder / 'run.json').read_text())['timings']
        summary.to_csv(RESULTS / f'summary_{tag}.csv')
        h.to_csv(RESULTS / f'h_diagnostics_{tag}.csv')
        (RESULTS / f'timings_{tag}.json').write_text(json.dumps(timings, indent=1))
        print(f'\n=== event probability {tail:.0%}: figure {path}')
        print(summary.to_string(float_format=lambda v: f'{v:.4f}'))
        print(h.to_string())
        print({k: round(v) for k, v in timings.items()})
