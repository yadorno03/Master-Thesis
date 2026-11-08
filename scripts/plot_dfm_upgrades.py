"""Figures for report 09: base DFM vs I3 vs fitted Gaussian on P2 and P3 (N = 2,500, seed 42, DSM checkpoint).

Checkpoints come from the Modal volume (``modal volume get thesis-runs /dfm_upgrades/...``) and are
expected in ``artifacts/dfm_upgrades/checkpoints/{P2,P3}_N2500_{base,gaussres}_seed42.pt``.

    python scripts/plot_dfm_upgrades.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
from dfm_upgrades import (  # noqa: E402
    STEPS, T, T0, WIDTH, build_population, factor_metrics, panel_population_spec,
)
from DFM import UnconditionalDFM  # noqa: E402
from dfm_variants import (  # noqa: E402
    GaussianResidualScoreNetwork, fit_gaussian_factor_model, sample_gaussian_factor_model,
)
from sampling import generate_samples  # noqa: E402
from synthetic_benchmark.configuration import Streams  # noqa: E402

CKPT = REPO / 'artifacts/dfm_upgrades/checkpoints'
OUT = REPO / 'report/figures'
N, SEED, N_GEN, N_TRUTH = 2500, 42, 20_000, 400_000
# Default categorical slots 1-3 of the dataviz palette (validated adjacent-pair set); truth in neutral ink.
STYLE = {
    'I3': dict(color='#2a78d6', ls='-', marker='o', label='I3 (Gaussian part + MLP)'),
    'base': dict(color='#eb6834', ls='--', marker='s', label='Base DFM'),
    'gauss': dict(color='#1baf7a', ls=':', marker='^', label='Fitted Gaussian (I0)'),
}
INK, MUTED, GRID = '#1f1f1e', '#6b6a65', '#e4e2dc'


def load_dfm(arm, pop_name, d, k):
    model = UnconditionalDFM(d, k, sigma_max=2.0, T=T, t0=T0, hidden_dims=[WIDTH, WIDTH])
    if arm == 'gaussres':
        model.factor_score_network = GaussianResidualScoreNetwork(d, k, 2.0, (WIDTH, WIDTH))
    state = torch.load(CKPT / f'{pop_name}_N{N}_{arm}_seed{SEED}.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def generate(model, n, seed):
    out = []
    with torch.no_grad():
        for i, chunk in enumerate(np.array_split(np.arange(n), int(np.ceil(n / 2500)))):
            g = torch.Generator().manual_seed(seed + i)
            out.append(generate_samples(model, len(chunk), num_steps=STEPS, generator=g).double().numpy())
    return np.concatenate(out)


def samples_for(pop_spec):
    pop = build_population(pop_spec)
    d, k, name = pop.d, pop.k, pop_spec['name']
    x_train = pop.sample(N, Streams(SEED).np('train'))  # the exact training set of the run
    params = fit_gaussian_factor_model(x_train, k)
    gen = {'gauss': sample_gaussian_factor_model(params, N_GEN, np.random.default_rng(9100))}
    for key, arm in (('base', 'base'), ('I3', 'gaussres')):
        print(f'{name}: generating {N_GEN} rows from {arm}', flush=True)
        gen[key] = generate(load_dfm(arm, name, d, k), N_GEN, 5000)
    if hasattr(pop, 'reference_portfolios'):
        truth_ew = pop.reference_portfolios['equal']
    else:
        truth_ew = pop.sample(N_TRUTH, np.random.default_rng(9002)) @ pop.w
    return pop, gen, truth_ew


def style_axes(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)


def plot_row(axes, pop, gen, truth_ew, title):
    w = pop.w
    threshold = float(np.quantile(truth_ew, 0.05))
    order = ('I3', 'base', 'gauss')

    # (a) factor variance ratios, generated / true, along the true factor directions
    ax = axes[0]
    ratios = {key: np.asarray(factor_metrics(gen[key], pop)['factor_ratios']) for key in order}
    m = len(ratios['I3'])
    width = 0.26
    for j, key in enumerate(order):
        s = STYLE[key]
        ax.bar(np.arange(m) + (j - 1) * width, ratios[key], width * 0.9, color=s['color'], label=s['label'])
    ax.axhline(1.0, color=INK, lw=1.2)
    ax.set_xticks(np.arange(m), [str(i + 1) for i in range(m)])
    ax.set_xlabel('Factor (1 = market)', color=INK, fontsize=9)
    ax.set_ylabel('Variance generated / true', color=INK, fontsize=9)
    ax.set_ylim(0, max(1.3, max(r.max() for r in ratios.values()) * 1.08))
    ax.set_title(f'{title}\nFactor variances (target 1)', color=INK, fontsize=10, loc='left')

    # (b) QQ plot of the equal-weight portfolio against the truth, tails emphasized
    ax = axes[1]
    probs = np.concatenate([np.geomspace(1e-3, 0.05, 25), np.linspace(0.06, 0.94, 30), 1 - np.geomspace(0.05, 1e-3, 25)])
    tq = np.quantile(truth_ew, probs)
    for key in order:
        s = STYLE[key]
        ax.plot(tq, np.quantile(gen[key] @ w, probs), color=s['color'], ls=s['ls'], lw=2, label=s['label'])
    lim = [tq.min() * 1.05, tq.max() * 1.05]
    ax.plot(lim, lim, color=INK, lw=1.2, label='Truth (y = x)')
    for p in (0.01, 0.001):
        q = np.quantile(truth_ew, p)
        ax.axvline(q, color=MUTED, lw=0.8, ls='-.')
        ax.annotate(f'{p:.1%}', (q, lim[1]), xytext=(3, -12), textcoords='offset points', color=MUTED, fontsize=8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('True quantile, equal-weight portfolio', color=INK, fontsize=9)
    ax.set_ylabel('Generated quantile', color=INK, fontsize=9)
    ax.set_title('QQ plot, equal-weight portfolio\n(above the line on the left = losses too mild)',
                 color=INK, fontsize=10, loc='left')

    # (c) conditional distribution of the portfolio given the 5 % event (rejection)
    ax = axes[2]
    tail = truth_ew[truth_ew <= threshold]
    lo = np.quantile(tail, 0.001)
    bins = np.linspace(lo, threshold, 36)
    ax.hist(tail, bins=bins, density=True, color='#d9d7d0', label='Truth')
    ax.set_yscale('log')  # log density so the far tail (rare large losses) stays visible
    for key in order:
        s = STYLE[key]
        v = gen[key] @ w
        acc = v[v <= threshold]
        ax.hist(acc, bins=bins, density=True, histtype='step', color=s['color'], ls=s['ls'], lw=2,
                label=f"{s['label']}: {len(acc) / len(v):.1%} accepted")
    ax.set_xlabel('Equal-weight return given the 5 % event', color=INK, fontsize=9)
    ax.set_ylabel('Density (log scale)', color=INK, fontsize=9)
    ax.set_title(f'DFM rejection at the 5 % event\n(threshold {threshold:.3f})', color=INK, fontsize=10, loc='left')
    ax.legend(fontsize=7, frameon=False, loc='upper left')
    ax.set_ylim(bottom=1e-3)
    for ax in axes:
        style_axes(ax)
    return ratios


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p2 = panel_population_spec()
    specs = [(p2, 'P2: Gaussian factors calibrated on 500 real stocks'),
             (dict(p2, name='P3', nu=5.0), 'P3: same, Student-t factors (heavy tails)')]
    fig, grid = plt.subplots(2, 3, figsize=(15, 9.2))
    fig.patch.set_facecolor('white')
    for row, (spec, title) in zip(grid, specs):
        pop, gen, truth_ew = samples_for(spec)
        ratios = plot_row(row, pop, gen, truth_ew, title)
        w = pop.w
        for key in ('I3', 'base', 'gauss'):
            v = gen[key] @ w
            q1 = np.quantile(truth_ew, 0.01)
            print(f"{spec['name']} {key:5s} factors {np.round(ratios[key][:3], 2)} sd ratio {v.std() / truth_ew.std():.3f} "
                  f"P(below true 1%) {np.mean(v <= q1):.4f} kurtosis {stats.kurtosis(v):.2f} (true {stats.kurtosis(truth_ew):.2f})")
    handles, labels = grid[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=9)
    fig.suptitle('d = 500, N = 2,500 training days, seed 42, DSM checkpoint, 20,000 generated rows per model; '
                 'returns standardized per stock',
                 color=MUTED, fontsize=9, x=0.01, ha='left', y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    path = OUT / '09_i3_p2_p3.png'
    fig.savefig(path, dpi=150)
    print('saved', path)


if __name__ == '__main__':
    main()
