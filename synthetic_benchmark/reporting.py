"""Distribution metrics, figures, and the benchmark's written interpretation."""
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
    """Build the original report with its exact Markdown and whitespace."""
    text = f"""
1. **Unconditional recovery:** the first Gaussian training seed has covariance error {b.covariance_error:.3f}, portfolio KS {b.portfolio_ks:.3f} and sliced W1 {b.sliced_w1:.3f}. Matched true-versus-true values are {u0.covariance_error:.3f}, {u0.portfolio_ks:.3f}, {u0.sliced_w1:.3f}. For this execution, these gaps are {"substantially above" if b.covariance_error > 2*u0.covariance_error else "near"} the Monte Carlo covariance baseline. {"The unconditional joint law is not recovered accurately at this budget." if b.covariance_error > 2*u0.covariance_error else "The remaining metrics and seed variability must also be assessed before claiming recovery."}

Executed mode: **{config.mode}**, dimensions **{config.d} × {config.k}**, training seeds **{config.seeds}**, samples per comparison **{config.n_eval}**. One-seed smoke results do not estimate training variability.
"""
    return text


def plot_h_validation(h_curve, selected_epoch, output):
    """Save and display validation loss, marking the selected H epoch."""
    # Plot validation loss only; retain training losses in the saved CSV.
    ax = h_curve.set_index("epoch")[["validation"]].plot(
        title="H validation loss", ylabel="Time-averaged squared probability error",
    )
    ax.axvline(selected_epoch, color="gray", linestyle="--", label="Selected epoch")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output / "validation_loss.png", dpi=120)
    plt.show()


def conditional_sample_summary(guided_portfolio, filtered_guided_portfolio, true_conditional_portfolio, threshold, n_samples):
    """Summarize all guided rows, the event subset, and independent conditional truth."""
    conditional_sample_summary = pd.DataFrame([
        {"source": label, "samples": len(values),
         "retained_fraction": len(values) / n_samples,
         "event_rate": float(np.mean(values <= threshold)) if len(values) else np.nan,
         "portfolio_mean": float(values.mean()) if len(values) else np.nan,
         "portfolio_std": float(values.std(ddof=1)) if len(values) > 1 else np.nan}
        for label, values in (("H-guided DFM", guided_portfolio),
                              ("H-guided DFM (event filtered)", filtered_guided_portfolio),
                              ("True conditional", true_conditional_portfolio))
    ])
    return conditional_sample_summary


def plot_conditional_comparison(guided_samples, true_conditional_samples, filtered_guided_samples, population, threshold, case, output):
    """Save and display the original four panels, including the filtered subset."""
    # Compare original and event-filtered guided outputs with true conditional samples.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    guided_p = guided_samples @ population.w
    truth_p = true_conditional_samples @ population.w
    filtered_p = filtered_guided_samples @ population.w

    # Portfolio empirical CDFs and the exact conditional population CDF.
    ax = axes[0, 0]
    for values, label in ((truth_p, "True conditional"), (guided_p, "H-guided DFM"),
                          (filtered_p, "H-guided DFM (event filtered)")):
        if not len(values):
            ax.plot([], [], label=f"{label} (n=0; unavailable)")
            continue
        ordered = np.sort(values)
        ax.plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered), label=label)
    portfolio_grid = np.linspace(min(truth_p.min(), guided_p.min()), max(truth_p.max(), guided_p.max()), 500)
    ax.plot(portfolio_grid, population.cdf(portfolio_grid, c=threshold),
            "k--", label="Exact conditional CDF")
    ax.axvline(threshold, color="crimson", linestyle=":", label="Event threshold")
    ax.set(title="Conditional portfolio CDF", xlabel="Portfolio return", ylabel="Cumulative probability")
    ax.legend(fontsize=8)

    # QQ plot: agreement follows the diagonal, including in the tails.
    ax = axes[0, 1]
    quantiles = np.linspace(0.005, 0.995, 100)
    truth_quantiles = np.quantile(truth_p, quantiles)
    guided_quantiles = np.quantile(guided_p, quantiles)
    ax.scatter(truth_quantiles, guided_quantiles, s=12, label="H-guided DFM")
    if len(filtered_p):
        ax.scatter(truth_quantiles, np.quantile(filtered_p, quantiles), s=12,
                   label="H-guided DFM (event filtered)")
    else:
        ax.scatter([], [], label="H-guided DFM (event filtered; n=0; unavailable)")
    qq_limits = [min(truth_quantiles.min(), guided_quantiles.min()),
                 max(truth_quantiles.max(), guided_quantiles.max())]
    ax.plot(qq_limits, qq_limits, "k--", label="Perfect agreement")
    ax.axhline(threshold, color="crimson", linestyle=":", label="Event threshold")
    ax.set(title="Portfolio QQ: original and filtered", xlabel="True conditional quantile", ylabel="H-guided DFM quantile")
    ax.legend(fontsize=8)

    # One asset, identical histogram bins for both distributions.
    ax = axes[1, 0]
    asset_bins = np.histogram_bin_edges(
        np.concatenate([true_conditional_samples[:, 0], guided_samples[:, 0]]), bins=35,
    )
    ax.hist(true_conditional_samples[:, 0], bins=asset_bins, density=True,
            histtype="step", linewidth=1.7, label="True conditional")
    ax.hist(guided_samples[:, 0], bins=asset_bins, density=True,
            histtype="step", linewidth=1.7, label="H-guided DFM")
    if len(filtered_guided_samples):
        ax.hist(filtered_guided_samples[:, 0], bins=asset_bins, density=True,
                histtype="step", linewidth=1.7, label="H-guided DFM (event filtered)")
    else:
        ax.plot([], [], label="H-guided DFM (event filtered; n=0; unavailable)")
    ax.set(title="Asset 0 conditional return distribution", xlabel="Asset 0 return", ylabel="Density")
    ax.legend(fontsize=8)

    # Shared population directions expose differences in joint structure.
    ax = axes[1, 1]
    truth_projection = true_conditional_samples @ population.B[:, :2]
    guided_projection = guided_samples @ population.B[:, :2]
    ax.scatter(truth_projection[:, 0], truth_projection[:, 1], s=5, alpha=0.25, label="True conditional")
    ax.scatter(guided_projection[:, 0], guided_projection[:, 1], s=5, alpha=0.25, label="H-guided DFM")
    if len(filtered_guided_samples):
        filtered_projection = filtered_guided_samples @ population.B[:, :2]
        ax.scatter(filtered_projection[:, 0], filtered_projection[:, 1], s=5, alpha=0.4,
                   label="H-guided DFM (event filtered)")
    else:
        ax.scatter([], [], label="H-guided DFM (event filtered; n=0; unavailable)")
    ax.set(title="Two population loading projections", xlabel="Projection on loading 1", ylabel="Projection on loading 2")
    ax.legend(fontsize=8)

    guided_event_rate = np.mean(guided_p <= threshold)
    fig.suptitle(
        f"{case}: H-guided DFM vs true conditional distribution\n"
        f"Portfolio threshold = {threshold:.4f}; guided event satisfaction = {guided_event_rate:.1%}; "
        f"filtered n = {len(filtered_p):,}/{len(guided_p):,}"
    )
    fig.tight_layout()
    fig.savefig(output / "conditional_comparison.png", dpi=150)
    plt.show()
    print("Comparison includes all guided outputs and the subset satisfying the event; filtering does not guarantee the true conditional law.")



PORTFOLIO_CONVENTION = 'CPU NumPy float64 matrix-vector product; portfolio <= threshold, no tolerance'
MATCHING_RULE = 'Fixed seeded permutation per source; first n indices, without replacement, independent of values'


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


def conditional_comparisons(samples, weights, *, projection_seed, matching_seed, projections=64):
    """Sample-to-sample conditional diagnostics; never use unconditional targets/PITs.

    Each source gets one independent permutation. Prefixes give reproducible,
    nested matched-size subsets without changing the stored input arrays.
    Reference pairs must supply the *same* n as the associated comparison;
    insufficient references are explicitly unavailable, never silently downsized.
    Singleton empirical distances exist, but are flagged as weak evidence.
    """
    names = ('guided', 'filtered', 'rejection', 'truth', 'truth_reference', 'rejection_reference')
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

    def row(left, right, label, n, kind):
        available = n <= min(len(arrays[left]), len(arrays[right]))
        used = n if available else 0
        status = ('reference_shortfall' if not available else
                  'empty' if n == 0 else 'singleton' if n == 1 else 'ok')
        result = dict(comparison=label, row_type=kind, left=left, right=right,
                      n_left=len(arrays[left]), n_right=len(arrays[right]), target_n=n,
                      n_used=used, status=status, portfolio_w1=np.nan,
                      portfolio_ks=np.nan, sliced_w1=np.nan)
        if used:
            a, b = orders[left][:used], orders[right][:used]
            p, q = portfolios[left][a], portfolios[right][b]
            result.update(portfolio_w1=float(wasserstein_distance(p, q)),
                          portfolio_ks=float(ks_2samp(p, q).statistic),
                          sliced_w1=float(np.mean([
                              wasserstein_distance(projected[left][a, j], projected[right][b, j])
                              for j in range(projections)])))
        return result

    pairs = (('guided', 'rejection', 'Guided vs DFM rejection'),
             ('filtered', 'rejection', 'Event-filtered guided vs DFM rejection'),
             ('rejection', 'truth', 'DFM rejection vs true conditional'),
             ('guided', 'truth', 'Guided vs true conditional'),
             ('filtered', 'truth', 'Event-filtered guided vs true conditional'))
    rows = [row(a, b, label, min(len(arrays[a]), len(arrays[b])), 'comparison') for a, b, label in pairs]
    for n in sorted({r['n_used'] for r in rows}):
        rows.append(row('truth', 'truth_reference', 'True conditional vs independent true conditional', n, 'MC reference'))
        rows.append(row('rejection', 'rejection_reference', 'DFM rejection vs independent DFM rejection', n, 'MC reference'))
    return pd.DataFrame(rows), directions


def extend_conditional_summary(summary, samples, weights, threshold, diagnostics):
    """Separate event satisfaction, guided retention and proposal acceptance."""
    summary = summary.copy()
    summary['fraction_meaning'] = ['raw guided count / requested', 'guided filtering retention',
                                   'truth count / requested']
    summary['proposal_acceptance_fraction'] = np.nan
    summary['status'] = ['complete', 'empty' if not len(samples['filtered']) else 'filtered', 'complete']
    summary['shortfall'] = 0
    for name, label in (('rejection', 'DFM rejection'),
                        ('rejection_reference', 'Independent DFM rejection'),
                        ('truth_reference', 'Independent true conditional')):
        p = portfolio_values(samples[name], weights)
        diag = diagnostics.get(name)
        truth_shortfall = len(samples['truth']) - len(p) if diag is None else 0
        summary = pd.concat([summary, pd.DataFrame([dict(
            source=label, samples=len(p), retained_fraction=np.nan,
            fraction_meaning='not guided filtering',
            event_rate=float(np.mean(p <= threshold)) if len(p) else np.nan,
            portfolio_mean=float(p.mean()) if len(p) else np.nan,
            portfolio_std=float(p.std(ddof=1)) if len(p) > 1 else np.nan,
            proposal_acceptance_fraction=diag['observed_acceptance_fraction'] if diag else np.nan,
            status=diag['status'] if diag else 'complete' if not truth_shortfall else 'unavailable',
            shortfall=diag['shortfall'] if diag else truth_shortfall,
        )])], ignore_index=True)
    return summary


def plot_conditional_ecdf(samples, population, threshold, case, metrics, output, diagnostics):
    """Compact full-range step ECDF with matched-size descriptive W1 references."""
    fig, (ax, caption) = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={'width_ratios': [2.1, 1]})
    portfolios = {name: portfolio_values(samples[name], population.w)
                  for name in ('truth', 'rejection', 'guided', 'filtered')}
    values = np.concatenate(list(portfolios.values()))
    lo, hi = min(threshold, values.min()) if len(values) else threshold, max(threshold, values.max()) if len(values) else threshold
    pad = max((hi-lo)*.035, .001)
    for name, label, color in (('truth', 'Population conditional truth', 'tab:blue'),
                                ('rejection', 'DFM conditional (rejection)', 'tab:green'),
                                ('guided', 'Raw H-guided', 'tab:orange'),
                                ('filtered', 'Event-filtered H-guided', 'tab:purple')):
        p = portfolios[name]
        style = '--' if name == 'filtered' else '-'
        status = f"; {diagnostics[name]['status']}" if name in diagnostics else ''
        if len(p):
            ordered = np.sort(p)
            ax.step(np.r_[lo-pad, ordered, hi+pad], np.r_[0., np.arange(1, len(p)+1)/len(p), 1.],
                    where='post', color=color, linestyle=style, label=f'{label} (n={len(p):,}{status})')
        else:
            ax.plot([], [], color=color, linestyle=style, label=f'{label} (n=0{status})')
    grid = np.linspace(lo-pad, hi+pad, 600)
    ax.plot(grid, population.cdf(grid, c=threshold), 'k--', linewidth=1.2, label='Population conditional CDF')
    ax.axvline(threshold, color='crimson', linestyle=':', label='Event threshold C')
    ax.set(xlim=(lo-pad, hi+pad), ylim=(-.02, 1.02), xlabel='Portfolio return', ylabel='Cumulative probability',
           title=f'{case}: conditional portfolio')
    ax.legend(fontsize=8, loc='best')
    caption.axis('off')
    lines = ['Portfolio W1 (return units)']
    for left, right, label in (('rejection', 'truth', 'Rejection / truth'),
                               ('guided', 'rejection', 'H-guided / rejection'),
                               ('filtered', 'rejection', 'H-guided filtered / rejection'),
                               ('guided', 'truth', 'H-guided / truth'),
                               ('filtered', 'truth', 'H-guided filtered / truth')):
        r = metrics[(metrics.left == left) & (metrics.right == right)].iloc[0]
        val = f'{r.portfolio_w1:.4g}' if np.isfinite(r.portfolio_w1) else 'unavailable'
        lines.extend(['', f'{label}: {val} (n={r.n_used})'])
        for source, short in (('truth', 'Truth / truth'), ('rejection', 'Rejection / rejection')):
            ref = metrics[(metrics.row_type == 'MC reference') & (metrics.left == source) & (metrics.target_n == r.n_used)].iloc[0]
            v = f'{ref.portfolio_w1:.4g}' if np.isfinite(ref.portfolio_w1) else ref.status
            lines.append(f'  {short}: {v}')
    lines.extend(['', 'References describe Monte Carlo variability;', 'they are not significance thresholds.',
                  'Portfolio agreement does not establish', 'the full multivariate conditional law.'])
    for name, diag in diagnostics.items():
        if not diag['complete']:
            lines.append(f"{name}: shortfall {diag['shortfall']:,}")
    caption.text(0, .98, '\n'.join(lines), va='top', fontsize=8, linespacing=1.2)
    fig.tight_layout()
    fig.savefig(output / 'conditional_portfolio_ecdf.png', dpi=150)
    plt.show()
    return fig
