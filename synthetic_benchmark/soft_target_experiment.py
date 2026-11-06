"""Controlled target-only comparison of H probability targets on a frozen DFM.

All arms share the frozen DFM, event, base trajectory splits, H initialization,
normalization, optimizer, time-sampling distribution and training budget; only
the regression targets on late rows differ. Nothing here retrains the DFM or
writes into an existing artifact directory.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import kstest, wasserstein_distance
import torch

from h_model import _continuation_final_probability, evaluate_h_scores, train_h
from sampling import generate_samples
from .conditional import (
    _checkpoint_identity, _conditional_assessment, assess_conditional_samples, fit_and_evaluate_h,
    h_rng, initialize_h, load_frozen_dfm, load_trained_h, sample_dfm_rejection, sample_h_guided,
    sample_h_training_data, save_conditional_samples, save_h_training, filter_conditional_samples,
)
from .configuration import (
    ConditionalAssessmentConfig, validate_branch_seeds, validate_h_seeds, write_json,
)
from .reporting import conditional_sample_summary, extend_conditional_summary, portfolio_values
from .soft_targets import (
    array_digest, construction_identity, file_sha256, load_target_cache, save_target_cache,
    split_array_digests, target_construction,
)

# Existing Refined H, held fixed in every arm.
REFINED_OPTIONS = dict(normalize_inputs=True, loss='bce', late_time_fraction=0.4,
                       last_step_fraction=0.1, use_last_step_targets=True, times_per_trajectory=32)
EXPERIMENT_VERSION = 'h-soft-target-ablation-v1'


def default_arms(branch_count=8):
    """The four target arms; training weights stay the existing ones."""
    return {
        'existing_refined': dict(label='Existing Refined H', mode='off', window=None, branch_count=None),
        'reuse_0.1': dict(label='Reuse final, r=0.1', mode='reuse_final', window=0.1, branch_count=None),
        'reuse_0.5': dict(label='Reuse final, r=0.5', mode='reuse_final', window=0.5, branch_count=None),
        f'continuation_0.1_M{branch_count}': dict(
            label=f'Continuation, r=0.1, M={branch_count}', mode='continuation', window=0.1,
            branch_count=branch_count),
    }


def arm_training_options(arm, *, late_time_fraction=0.4, last_step_fraction=0.1):
    options = dict(REFINED_OPTIONS, late_time_fraction=late_time_fraction,
                   last_step_fraction=last_step_fraction)
    if arm['mode'] != 'off':
        options.update(soft_target_mode=arm['mode'], soft_target_window=arm['window'])
    return options


def continuation_setting(arms):
    """The single (window, branches) continuation collection the arms share, or None."""
    settings = {(a['window'], a['branch_count']) for a in arms.values() if a['mode'] == 'continuation'}
    if len(settings) > 1:
        raise ValueError('One run supports one continuation (window, M); run others separately')
    return next(iter(settings), None)


def peak_memory_mb():
    """Peak resident set size of this process so far (a high-water mark, not a phase delta)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == 'darwin' else peak / 2**10


def check_distinct_streams(named_seeds):
    """Every named stream must differ in its lower 32 bits (CPU generator aliasing)."""
    seen = {}
    for name, seed in named_seeds.items():
        key = int(seed) % 2**32
        if key in seen:
            raise ValueError(f'Random streams {seen[key]!r} and {name!r} reuse the same stream')
        seen[key] = name


def data_seed_sets(offset):
    """Base and branch seeds of one independent trajectory-data repetition."""
    base = 32001 + 100 * offset
    return (dict(train=base, validation=base + 1, test=base + 2, shuffle=base + 3),
            dict(train=33001 + 100 * offset, validation=33002 + 100 * offset, test=33003 + 100 * offset))


# ----------------------------------------------------------------------------- metrics

def sample_status(values):
    return 'empty' if len(values) == 0 else 'singleton' if len(values) == 1 else 'ok'


def conditional_moment_row(samples, population, threshold):
    """Mean/covariance error and covariance trace ratio against the exact conditional law."""
    row = dict(n=len(samples), status=sample_status(samples), mean_error=np.nan,
               covariance_error=np.nan, covariance_trace_ratio=np.nan)
    if len(samples) < 2:
        return row
    mean, covariance = population.moments(c=threshold)
    scale = np.sqrt(np.trace(population.moments()[1]))
    samples = np.asarray(samples, dtype=np.float64)
    sample_covariance = np.cov(samples, rowvar=False)
    row.update(mean_error=float(np.linalg.norm(samples.mean(0) - mean) / scale),
               covariance_error=float(np.linalg.norm(sample_covariance - covariance)
                                      / np.linalg.norm(covariance)),
               covariance_trace_ratio=float(np.trace(sample_covariance) / np.trace(covariance)))
    return row


def distribution_row(values, population, threshold, rejection, truth):
    """Exact-population KS, and empirical W1/KS to rejection and to truth portfolios."""
    row = dict(n=len(values), status=sample_status(values), ks_exact=np.nan)
    for name in ('rejection', 'truth'):
        row.update({f'w1_{name}': np.nan, f'ks_{name}': np.nan})
    if len(values) < 2:
        return row
    row['ks_exact'] = float(kstest(values, population.cdf, args=(threshold,)).statistic)
    for name, reference in (('rejection', rejection), ('truth', truth)):
        if len(reference) > 1:
            row[f'w1_{name}'] = float(wasserstein_distance(values, reference))
            row[f'ks_{name}'] = float(_two_sample_ks(values, reference))
    return row


def _two_sample_ks(a, b):
    from scipy.stats import ks_2samp
    return ks_2samp(a, b).statistic


def common_size_comparison(filtered, population, threshold, references, *, seed, repeats=20):
    """Equal-size, value-independent subsets so arms with different retention compare fairly.

    Subsets use seeded permutation prefixes, so selection never depends on the
    portfolio value. Reports available and used counts; nothing is upsampled.
    Arms below two rows (or with no reference) are explicitly 'insufficient'.
    """
    usable = {name: len(v) for name, v in filtered.items() if len(v) >= 2}
    n_common = min(usable.values()) if usable else 0
    streams = np.random.SeedSequence(seed).spawn(repeats)
    rows = []
    for name, values in filtered.items():
        row = dict(arm=name, n_available=len(values), n_used=n_common if len(values) >= 2 else 0,
                   repeats=repeats, status='ok' if len(values) >= 2 and n_common >= 2 else 'insufficient')
        for metric in ('ks_exact', 'w1_rejection', 'w1_truth'):
            row[f'{metric}_mean'] = row[f'{metric}_sd'] = np.nan
        if row['status'] == 'ok':
            draws = {metric: [] for metric in ('ks_exact', 'w1_rejection', 'w1_truth')}
            for stream in streams:
                rng = np.random.default_rng(stream)
                subset = np.asarray(values)[rng.permutation(len(values))[:n_common]]
                draws['ks_exact'].append(kstest(subset, population.cdf, args=(threshold,)).statistic)
                for metric, key in (('w1_rejection', 'rejection'), ('w1_truth', 'truth')):
                    reference = np.asarray(references[key])
                    if len(reference) >= n_common:
                        picked = reference[rng.permutation(len(reference))[:n_common]]
                        draws[metric].append(wasserstein_distance(subset, picked))
            for metric, results in draws.items():
                if results:
                    row[f'{metric}_mean'], row[f'{metric}_sd'] = float(np.mean(results)), float(np.std(results))
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------ late-state residual diagnostic

@torch.no_grad()
def build_residual_reference(model, *, steps, remaining, n_states, branches, path_seed, branch_seed,
                             branch_batch_size=2048):
    """Fixed held-out ambient states and a high-M continuation estimate of h_j^E(x).

    These states and branches are separate diagnostic streams: never used to
    construct training targets or to select models. The estimate carries Monte
    Carlo error, reported as a per-state standard error.
    """
    reference = model.reference
    device = next(model.network.parameters()).device
    paths = generate_samples(reference, n_states, num_steps=steps, return_trajectory=True,
                             generator=torch.Generator(device=device).manual_seed(path_seed))
    dt = model.horizon / steps
    times = torch.arange(steps, dtype=paths.dtype, device=device) * dt
    rows = sorted({steps - 1, *(int(torch.argmin((model.horizon - times - r).abs())) for r in remaining)})
    branch_rng = torch.Generator(device=device).manual_seed(branch_seed)
    states, targets, errors = {}, {}, {}
    for j in rows:
        x = paths[:, j].clone()
        states[j] = x
        if j == steps - 1:
            targets[j], errors[j] = model.exact_last_step_probability(x, dt)[:, 0], torch.zeros(n_states, device=device)
            continue
        order = torch.arange(n_states * branches, device=device)
        values = torch.cat([_continuation_final_probability(model, x[chunk // branches], j, steps,
                                                            generator=branch_rng)
                            for chunk in order.split(branch_batch_size)])[:, 0].reshape(n_states, branches)
        targets[j], errors[j] = values.mean(1), values.std(1, unbiased=True) / branches ** .5
    del paths
    return dict(rows=rows, times=times[rows].cpu(), remaining=(model.horizon - times[rows]).cpu(),
                states=states, targets=targets, standard_errors=errors, branches=branches,
                path_seed=path_seed, branch_seed=branch_seed, n_states=n_states)


@torch.no_grad()
def evaluate_residuals(model, reference_states):
    """H(t_j, G_{t_j} x) - h_j^E(x): network error plus any reduced-feature insufficiency."""
    rows = []
    for position, j in enumerate(reference_states['rows']):
        x = reference_states['states'][j]
        t = reference_states['times'][position].to(x.device, x.dtype)
        residual = (model(t, model.features(t, x))[:, 0] - reference_states['targets'][j]).double()
        noise = reference_states['standard_errors'][j].double().square().mean()
        mse = residual.square().mean()
        rows.append(dict(remaining=float(reference_states['remaining'][position]), row=j,
                         n_states=len(residual), reference_branches=reference_states['branches'],
                         mean_residual=float(residual.mean()), rmse=float(mse.sqrt()),
                         mean_reference_se=float(reference_states['standard_errors'][j].double().mean()),
                         rmse_noise_corrected=float((mse - noise).clamp_min(0).sqrt()),
                         mean_reference_probability=float(reference_states['targets'][j].double().mean())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ reference samples

def load_reference_samples(sample_dir, *, dfm_checkpoint, population, threshold, count, steps, settings):
    """Reuse saved truth/rejection draws only after verifying they describe this experiment."""
    sample_dir = Path(sample_dir).resolve()
    sampling = json.loads((sample_dir / 'sampling.json').read_text())
    identity = (sampling.get('checkpoint_identity') or {}).get('dfm') or {}
    if identity.get('sha256') != file_sha256(dfm_checkpoint):
        raise ValueError('Saved reference samples come from a different frozen DFM')
    if sampling['threshold'] != threshold or sampling['steps'] != steps or sampling['n'] != count:
        raise ValueError('Saved reference samples use a different event, grid or sample count')
    if sampling['assessment_config'] != asdict(settings):
        raise ValueError('Saved reference samples use different assessment settings')
    with np.load(sample_dir / 'conditional_samples.npz', allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    np.testing.assert_array_equal(arrays['weights'], population.w)
    if float(arrays['threshold']) != threshold:
        raise ValueError('Saved reference threshold does not match the event')
    samples = dict(truth=arrays['true_conditional_samples'], rejection=arrays['rejection_samples'],
                   rejection_reference=arrays['independent_rejection_samples'],
                   truth_reference=arrays['independent_true_conditional_samples'])
    return dict(samples=samples, rejection_diagnostics=sampling['rejection_diagnostics'],
                truth_reference_diagnostics=sampling['truth_reference_diagnostics'],
                projection_directions=arrays['projection_directions'], source=str(sample_dir))


def build_reference_samples(reference, population, threshold, count, *, steps, settings,
                            truth_seed, existing_seeds):
    """Fresh truth and DFM-rejection draws, shared by every arm and repetition."""
    truth = population.sample(count, np.random.default_rng(truth_seed), c=threshold)
    empty = np.empty((0, population.d))
    assessment = assess_conditional_samples(reference, population, threshold, empty, truth, empty,
                                            steps=steps, settings=settings, existing_seeds=existing_seeds)
    samples = {name: assessment['samples'][name] for name in
               ('truth', 'rejection', 'rejection_reference', 'truth_reference')}
    return dict(samples=samples, rejection_diagnostics=assessment['rejection_diagnostics'],
                truth_reference_diagnostics=assessment['truth_reference_diagnostics'],
                projection_directions=assessment['projection_directions'], source='fresh')


# --------------------------------------------------------------------------------- arm run

def _initial_weights_digest(model):
    return array_digest(torch.cat([p.detach().flatten() for p in model.network.parameters()]))


def _sample_arm(h_output, device, population, reference_samples, settings, *, count, steps,
                sample_batch_size, guided_seed, truth_seed, initialization_max_proposals):
    """Guided draws, filtering and assessment for one trained arm; failures stay explicit."""
    model, metadata, guidance, _ = load_trained_h(h_output, device)
    threshold = model.C
    diagnostics, status, error = {}, 'complete', None
    started = time.perf_counter()
    try:
        guided = sample_h_guided(model, guidance, n_samples=count, batch_size=sample_batch_size,
                                 steps=steps, seed=guided_seed,
                                 initialization_max_proposals=initialization_max_proposals,
                                 diagnostics=diagnostics)
    except RuntimeError as exc:                       # initialization budget exhausted
        status, error, guided = 'initialization_exhausted', str(exc), None
    except ValueError as exc:
        if 'nonfinite' not in str(exc).lower():
            raise
        status, error, guided = 'nonfinite_sampler', str(exc), None
    seconds = time.perf_counter() - started
    result = dict(status=status, error=error, sampling_seconds=seconds, diagnostics=diagnostics,
                  model=model, metadata=metadata, threshold=threshold, guided=guided)
    if guided is None:
        return result
    truth = reference_samples['samples']['truth']
    portfolio, truth_portfolio, mask, filtered, filtered_portfolio = filter_conditional_samples(
        guided, truth, population, threshold, count, model.reference.d)
    samples = dict(guided=guided, filtered=filtered, **reference_samples['samples'])
    assessment = _conditional_assessment(samples, population, settings,
                                         reference_samples['rejection_diagnostics'],
                                         reference_samples['truth_reference_diagnostics'])
    summary = extend_conditional_summary(
        conditional_sample_summary(portfolio, filtered_portfolio, truth_portfolio, threshold, count),
        samples, population.w, threshold, assessment['rejection_diagnostics'])
    sample_output = save_conditional_samples(
        Path(h_output), metadata, population, threshold, guided, truth, filtered, mask, summary,
        n_samples=count, steps=steps, batch_size=sample_batch_size, guided_seed=guided_seed,
        truth_seed=truth_seed, assessment=assessment, guided_sampling_diagnostics=diagnostics)
    result.update(samples=samples, mask=mask, assessment=assessment, sample_output=sample_output,
                  portfolio=portfolio, filtered_portfolio=filtered_portfolio)
    return result


def _arm_metric_row(name, arm, repetition, trained, sampled, population, reference_samples, scores,
                    costs):
    threshold = sampled['threshold']
    row = dict(arm=name, label=arm['label'], mode=arm['mode'], window=arm['window'],
               branch_count=arm['branch_count'], **repetition, status=sampled['status'],
               selected_epoch=trained['selection']['best_epoch'],
               optimization_seconds=trained['seconds'], sampling_seconds=sampled['sampling_seconds'],
               **{f'score_{k}': v for k, v in scores.items()}, **costs)
    terminal = trained['test_metrics']['terminal_diagnostics']
    for key in ('last_probability_rmse', 'gradient_relative_rmse', 'gradient_mean_cosine', 'gradient_n'):
        row[f'test_{key}'] = terminal.get(key, np.nan)
    rejection = reference_samples['rejection_diagnostics']['rejection']
    row['dfm_proposal_acceptance'] = rejection['observed_acceptance_fraction']
    if sampled['guided'] is None:
        return row
    portfolio, filtered = sampled['portfolio'], sampled['filtered_portfolio']
    count = len(portfolio)
    initialization = sampled['diagnostics'].get('initialization', {})
    row.update(n_requested=count, event_rate=float((portfolio <= threshold).mean()),
               n_filtered=len(filtered), filtered_fraction=len(filtered) / count,
               init_proposals=initialization.get('proposals'),
               init_acceptance_rate=initialization.get('acceptance_rate'))
    accept = row['dfm_proposal_acceptance']
    row['event_enrichment'] = row['event_rate'] / accept if accept else np.nan
    rejection_values = portfolio_values(reference_samples['samples']['rejection'], population.w)
    truth_values = portfolio_values(reference_samples['samples']['truth'], population.w)
    for label, values, samples in (('unfiltered', portfolio, sampled['samples']['guided']),
                                   ('filtered', filtered, sampled['samples']['filtered'])):
        for key, value in distribution_row(values, population, threshold, rejection_values,
                                           truth_values).items():
            row[f'{label}_{key}'] = value
        for key, value in conditional_moment_row(samples, population, threshold).items():
            if key not in ('n', 'status'):
                row[f'{label}_{key}'] = value
    metrics = sampled['assessment']['comparison_metrics']
    for label, comparison in (('unfiltered', 'Guided vs DFM rejection'),
                              ('filtered', 'Event-filtered guided vs DFM rejection')):
        found = metrics[metrics.comparison == comparison].iloc[0]
        row.update({f'{label}_sliced_w1_rejection': found.sliced_w1,
                    f'{label}_matched_n_rejection': found.n_used})
    return row


def run_arm(name, arm, *, h_data, construction, reference, population, tail, widths, init_seed,
            seeds, branch_seeds, branch_batch_size, settings_h, output_root, device, reference_samples, assessment, count, steps,
            sample_batch_size, guided_seed, truth_seed, initialization_max_proposals, repetition,
            collection_costs, source_checkpoint, array_digests):
    """Train, score and sample one arm from a fresh identical initialization."""
    options = arm_training_options(arm, late_time_fraction=settings_h['late_time_fraction'],
                                   last_step_fraction=settings_h['last_step_fraction'])
    _, model, _ = initialize_h(reference, population, tail, widths, init_seed, log_remaining_time=True)
    initial_digest = _initial_weights_digest(model)
    frozen_before = {k: v.detach().clone() for k, v in reference.state_dict().items()}
    arm_construction = target_construction(
        model, mode=arm['mode'], window=arm['window'], steps=steps,
        split_counts=construction['split_counts'], stream_seeds=seeds,
        path_batch_size=construction['path_batch_size'], dfm_checkpoint=source_checkpoint,
        branch_count=arm['branch_count'], branch_seeds=branch_seeds,
        branch_batch_size=branch_batch_size)
    # Identity of the target construction and of the exact arrays it produced.
    arm_construction['construction_identity'] = construction_identity(arm_construction)
    arm_construction['target_arrays'] = array_digests
    started = time.perf_counter()
    train_options = dict(options)
    selection, history, probability, test_metrics = fit_and_evaluate_h(
        model, reference, h_data, frozen_before, epochs=settings_h['epochs'],
        batch_size=settings_h['batch_size'], lr=settings_h['lr'], seeds=seeds, device=device,
        training_options=train_options,
        target_construction=None if arm['mode'] == 'off' else arm_construction)
    optimization_seconds = time.perf_counter() - started
    score_options = dict(late_time_fraction=options['late_time_fraction'],
                         last_step_fraction=options['last_step_fraction'], use_last_step_targets=True,
                         soft_target_mode=arm['mode'], soft_target_window=arm['window'])
    scores = {f'test_{k}': v for k, v in evaluate_h_scores(model, h_data['test'], **score_options).items()}
    scores.update({f'validation_{k}': v for k, v in
                   evaluate_h_scores(model, h_data['validation'], **score_options).items()})
    _, _, h_output = save_h_training(
        output_root, model, population, h_data, selection, history, probability, test_metrics,
        case='gaussian', seed=settings_h['dfm_seed'], tail=tail, n_train=settings_h['n_train'],
        n_validation=settings_h['n_validation'], n_test=settings_h['n_test'],
        epochs=settings_h['epochs'], lr=settings_h['lr'], batch_size=settings_h['batch_size'],
        path_batch_size=settings_h['path_batch_size'], steps=steps, stream_seeds=seeds,
        initialization_seed=init_seed, reference_checkpoint=source_checkpoint, started=started,
        target_construction=None if arm['mode'] == 'off' else arm_construction)
    trained = dict(selection=selection, seconds=optimization_seconds, test_metrics=test_metrics)
    sampled = _sample_arm(h_output, device, population, reference_samples, assessment, count=count,
                          steps=steps, sample_batch_size=sample_batch_size, guided_seed=guided_seed,
                          truth_seed=truth_seed, initialization_max_proposals=initialization_max_proposals)
    costs = dict(collection_seconds=collection_costs['collection_seconds'],
                 target_construction_seconds=collection_costs['branch_seconds'] if arm['mode'] == 'continuation' else 0.,
                 peak_memory_mb=peak_memory_mb(), initial_weights_sha256=initial_digest)
    row = _arm_metric_row(name, arm, repetition, trained, sampled, population, reference_samples,
                          scores, costs)
    return dict(row=row, h_output=h_output, sampled=sampled, model=sampled['model'],
                curve=pd.read_csv(h_output / 'h_curve.csv'), construction=arm_construction,
                initial_weights_sha256=initial_digest)


def target_summary(h_data):
    """How informative the probability targets are: spread, non-binariness and branch SE."""
    train = h_data['train']
    q = train.last_probabilities
    interior = lambda v: float(((v > .01) & (v < .99)).float().mean())
    summary = dict(binary_label_mean=float(train.labels.mean()),
                   last_step_probability=dict(mean=float(q.mean()), fraction_between_0p01_0p99=interior(q)))
    if train.soft_targets is not None:
        late = train.soft_targets[:, :-1]
        errors = train.soft_target_standard_errors
        summary['continuation'] = dict(
            rows=train.soft_target_indices.tolist(), n_intermediate_rows=late.shape[1],
            mean_over_intermediate_rows=float(late.mean()) if late.numel() else None,
            fraction_between_0p01_0p99=interior(late) if late.numel() else None,
            mean_standard_error=float(errors[:, :-1].mean()) if errors is not None and late.numel() else None,
            standard_error='from branch probabilities; None when M=1')
    return summary


# ----------------------------------------------------------------------------- aggregation

def aggregate_repetitions(table, columns):
    """Explicit separate variability: initialization seeds (fixed data) versus data offsets."""
    rows = []
    for arm, group in table.groupby('arm', sort=False):
        row = dict(arm=arm, n_runs=len(group))
        for column in columns:
            values = group[column].astype(float)
            row[f'{column}_mean'] = float(values.mean()) if values.notna().any() else np.nan
            within = [g[column].astype(float).std(ddof=1) for _, g in group.groupby('data_offset')
                      if len(g) > 1]
            row[f'{column}_sd_init_seed'] = float(np.mean(within)) if within else np.nan
            by_offset = group.groupby('data_offset')[column].mean()
            row[f'{column}_sd_data_offset'] = float(by_offset.std(ddof=1)) if len(by_offset) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


SUMMARY_COLUMNS = ('event_rate', 'filtered_fraction', 'event_enrichment', 'filtered_ks_exact',
                   'unfiltered_ks_exact', 'filtered_w1_rejection', 'filtered_w1_truth',
                   'filtered_sliced_w1_rejection', 'filtered_mean_error', 'filtered_covariance_error',
                   'filtered_covariance_trace_ratio', 'score_test_binary_brier', 'score_test_binary_bce',
                   'test_last_probability_rmse', 'test_gradient_relative_rmse', 'test_gradient_mean_cosine',
                   'optimization_seconds', 'sampling_seconds')


# ------------------------------------------------------------------------------ cost probe

def estimate_cost(reference, population, tail, widths, init_seed, *, arms, settings_h, count, steps,
                  assessment, device, probe_paths=128):
    """Scale small timed probes (after a warm-up call) to the requested budgets.

    Indicative only: probes are short, so expect tens of percent of error, and the
    estimate excludes DFM-rejection streams beyond their proposal caps.
    """
    from h_model import evaluate_h_objective
    _, model, guidance = initialize_h(reference, population, tail, widths, init_seed,
                                      log_remaining_time=True)
    seeds, branch_seeds = data_seed_sets(0)
    continuation = continuation_setting(arms)
    batch = min(256, probe_paths)

    def collect(**options):
        return model.sample_trajectories(probe_paths, steps, generator=h_rng('train', seeds, device),
                                         batch_size=batch, **options)

    def timed(function):
        function()                                     # warm-up: threads, caches, allocator
        started = time.perf_counter()
        value = function()
        return value, time.perf_counter() - started

    estimate = {}
    plain, seconds = timed(collect)
    total_paths = settings_h['n_train'] + settings_h['n_validation'] + settings_h['n_test']
    estimate['base_collection_seconds'] = seconds / probe_paths * total_paths
    if continuation:
        window, branches = continuation
        diagnostics = {}
        timed(lambda: collect(
            soft_target_mode='continuation', soft_target_window=window, branch_count=branches,
            branch_generator=h_rng('train', branch_seeds, device),
            branch_batch_size=settings_h['branch_batch_size'], diagnostics=diagnostics))
        # diagnostics accumulate over the warm-up and the timed call: normalise per call.
        estimate['continuation_target_seconds'] = diagnostics['branch_seconds'] / 2 / probe_paths * total_paths
        estimate['continuation_state_steps_per_path'] = diagnostics['branch_state_steps'] / 2 / probe_paths
        estimate['continuation_final_evaluations_per_path'] = diagnostics['branch_final_evaluations'] / 2 / probe_paths
    edges = torch.cat((plain.times, plain.times.new_tensor([model.horizon])))
    plain.weights = edges.diff() / model.horizon       # as sample_h_training_data does
    _, train_seconds = timed(lambda: train_h(
        model, plain, epochs=1, batch_size=settings_h['batch_size'], lr=settings_h['lr'],
        generator=h_rng('shuffle', seeds, device), **REFINED_OPTIONS))
    _, validation_seconds = timed(lambda: evaluate_h_objective(
        model, plain, batch_size=settings_h['batch_size'], loss='bce',
        late_time_fraction=settings_h['late_time_fraction'],
        last_step_fraction=settings_h['last_step_fraction'], use_last_step_targets=True))
    estimate['training_seconds_per_arm'] = settings_h['epochs'] * (
        train_seconds / probe_paths * settings_h['n_train']
        + validation_seconds / probe_paths * settings_h['n_validation'])
    start = torch.randn(16, population.d, generator=h_rng('train', seeds, device),
                        dtype=next(model.network.parameters()).dtype, device=device)
    _, guided_seconds = timed(lambda: generate_samples(
        reference, 16, num_steps=steps, initial_state=start, guidance_fn=guidance))
    estimate['guided_sampling_seconds_per_arm'] = guided_seconds / 16 * count
    _, proposal_seconds = timed(lambda: generate_samples(
        reference, 64, num_steps=steps, generator=h_rng('train', seeds, device)))
    caps = (assessment.rejection_max_proposals, assessment.rejection_reference_max_proposals)
    estimate['rejection_seconds_at_proposal_caps'] = proposal_seconds / 64 * sum(caps)
    estimate['per_arm_seconds'] = (estimate['training_seconds_per_arm']
                                   + estimate['guided_sampling_seconds_per_arm'])
    estimate['arms'] = len(arms)
    estimate['total_seconds_one_repetition_upper_bound'] = (
        estimate['base_collection_seconds'] + estimate.get('continuation_target_seconds', 0.)
        + estimate['rejection_seconds_at_proposal_caps'] + len(arms) * estimate['per_arm_seconds'])
    return estimate


# ------------------------------------------------------------------------------------ runner

def run_soft_target_experiment(
    source_run, output, *, arms=None, device='cpu', dfm_seed=42, n_train=10_000, n_validation=2_000,
    n_test=2_000, epochs=200, tail=0.05, widths=(64, 64), lr=1e-3, batch_size=64, path_batch_size=256,
    branch_batch_size=2048, late_time_fraction=0.4, last_step_fraction=0.1, init_seeds=(2027,),
    data_offsets=(0,), n_samples=3000, sample_batch_size=256, guided_seed=62001, truth_seed=62002,
    assessment_settings=None, initialization_max_proposals=None, reuse_reference_from=None,
    target_cache=None, residual_states=64, residual_branches=128, residual_remaining=(0.05, 0.1, 0.5, 1.0),
    residual_seeds=(63001, 63002), subset_repeats=20, source_h_run=None, estimate_only=False,
):
    """Run every arm on shared splits; write a fresh soft_target_comparison.json tree."""
    arms = default_arms() if arms is None else arms
    continuation = continuation_setting(arms)
    source_run, output = Path(source_run).resolve(), Path(output).resolve()
    if output == source_run or source_run in output.parents or output in source_run.parents:
        raise ValueError('Write the experiment to a fresh directory outside the source run')
    checkpoint_path = source_run / f'gaussian_seed{dfm_seed}_primary' / 'dfm.pt'
    checkpoint, dfm_config, population, _, reference = load_frozen_dfm(checkpoint_path, 'gaussian', device)
    steps = dfm_config['steps']
    assessment = assessment_settings or ConditionalAssessmentConfig(
        rejection_seed=62003, rejection_reference_seed=62004, truth_reference_seed=62005,
        projection_seed=62006, matching_seed=62007)
    settings_h = dict(n_train=n_train, n_validation=n_validation, n_test=n_test, epochs=epochs, lr=lr,
                      batch_size=batch_size, path_batch_size=path_batch_size,
                      branch_batch_size=branch_batch_size, late_time_fraction=late_time_fraction,
                      last_step_fraction=last_step_fraction, dfm_seed=dfm_seed)
    print(f'Frozen DFM: {checkpoint_path} (selected epoch {checkpoint["selected_epoch"]}); '
          f'd={population.d}, k={reference._score.k}, steps={steps}', flush=True)
    if estimate_only:
        estimate = estimate_cost(reference, population, tail, widths, init_seeds[0], arms=arms,
                                 settings_h=settings_h, count=n_samples, steps=steps,
                                 assessment=assessment, device=device)
        print(json.dumps(estimate, indent=2), flush=True)
        return estimate
    output.mkdir(parents=True, exist_ok=False)
    run_started = time.perf_counter()
    _, probe, _ = initialize_h(reference, population, tail, widths, init_seeds[0], log_remaining_time=True)
    threshold = probe.C
    saved_h = {}
    if source_h_run is not None:
        saved = json.loads((Path(source_h_run) / 'training.json').read_text())
        if Path(saved['reference_checkpoint']).resolve() != checkpoint_path:
            raise ValueError('The saved H run used a different frozen DFM')
        if saved['threshold'] != threshold:
            raise ValueError('The saved H run used a different event threshold')
        mine = dict(n_train=n_train, n_validation=n_validation, n_test=n_test, epochs=epochs, lr=lr,
                    batch_size=batch_size, path_batch_size=path_batch_size, steps=steps, tail=tail,
                    initialization_seed=init_seeds[0], stream_seeds=data_seed_sets(0)[0])
        saved_h = {k: dict(saved=saved.get(k), requested=v) for k, v in mine.items() if saved.get(k) != v}
        if saved_h:
            print('Settings differ from the saved H run:', saved_h, flush=True)
    all_seeds = {'guided': guided_seed, 'truth': truth_seed, 'residual_paths': residual_seeds[0],
                 'residual_branches': residual_seeds[1],
                 **{f'assessment_{name}': getattr(assessment, name) for name in
                    ('rejection_seed', 'rejection_reference_seed', 'truth_reference_seed',
                     'projection_seed', 'matching_seed')},
                 **{f'init_{s}': s for s in init_seeds}}
    for offset in data_offsets:
        base, branch = data_seed_sets(offset)
        all_seeds.update({f'base_{offset}_{k}': v for k, v in base.items()})
        all_seeds.update({f'branch_{offset}_{k}': v for k, v in branch.items()})
    check_distinct_streams(all_seeds)
    write_json(output / 'config.json', dict(
        version=EXPERIMENT_VERSION, source_run=str(source_run), dfm=_checkpoint_identity(checkpoint_path),
        arms=arms, settings=settings_h, tail=tail, widths=list(widths), init_seeds=list(init_seeds),
        data_offsets=list(data_offsets), steps=steps, threshold=threshold, n_samples=n_samples,
        sample_batch_size=sample_batch_size, guided_seed=guided_seed, truth_seed=truth_seed,
        assessment=asdict(assessment), initialization_max_proposals=initialization_max_proposals,
        residual=dict(n_states=residual_states, branches=residual_branches, remaining=list(residual_remaining),
                      seeds=list(residual_seeds)), subset_repeats=subset_repeats,
        source_h_run=None if source_h_run is None else str(source_h_run), differences_from_saved_h=saved_h,
        seed_streams=all_seeds, device=device, torch=torch.__version__))

    existing = tuple(all_seeds.values())
    reference_started = time.perf_counter()
    if reuse_reference_from is not None:
        reference_samples = load_reference_samples(
            reuse_reference_from, dfm_checkpoint=checkpoint_path, population=population,
            threshold=threshold, count=n_samples, steps=steps, settings=assessment)
        print('Reusing verified saved truth/rejection samples:', reference_samples['source'], flush=True)
    else:
        print('Generating fresh truth and DFM-rejection reference samples...', flush=True)
        # assess_conditional_samples checks assessment seeds against everything else.
        others = tuple(v for k, v in all_seeds.items() if not k.startswith('assessment_'))
        reference_samples = build_reference_samples(
            reference, population, threshold, n_samples, steps=steps, settings=assessment,
            truth_seed=truth_seed, existing_seeds=others)
    reference_seconds = time.perf_counter() - reference_started
    residual_reference = build_residual_reference(
        probe, steps=steps, remaining=residual_remaining, n_states=residual_states,
        branches=residual_branches, path_seed=residual_seeds[0], branch_seed=residual_seeds[1],
        branch_batch_size=branch_batch_size) if residual_states else None

    tables, residual_tables, repetitions = [], [], []
    curves, portfolio_store = {}, {}
    for offset in data_offsets:
        seeds, branch_seeds = data_seed_sets(offset)
        split_counts = (('train', n_train), ('validation', n_validation), ('test', n_test))
        collection_options = dict(soft_target_mode='off')
        if continuation:
            collection_options = dict(soft_target_mode='continuation', soft_target_window=continuation[0],
                                      branch_count=continuation[1], branch_seeds=branch_seeds,
                                      branch_batch_size=branch_batch_size)
        construction = target_construction(
            probe, mode=collection_options['soft_target_mode'], window=collection_options.get('soft_target_window'),
            steps=steps, split_counts=split_counts, stream_seeds=seeds, path_batch_size=path_batch_size,
            dfm_checkpoint=checkpoint_path, branch_count=collection_options.get('branch_count'),
            branch_seeds=collection_options.get('branch_seeds'), branch_batch_size=branch_batch_size)
        cache_dir = output / f'data_offset{offset}' / 'target_cache'
        collection = {}
        if target_cache is not None and (Path(target_cache) / 'identity.json').is_file():
            print('Loading verified target cache:', target_cache, flush=True)
            h_data = load_target_cache(target_cache, construction, probe)
            collection = {split: dict(collection_seconds=0., branch_seconds=0., cached=True)
                          for split in h_data}
        else:
            h_data = sample_h_training_data(
                probe, split_counts, steps=steps, seeds=seeds, path_batch_size=path_batch_size,
                device=device, diagnostics=collection, **collection_options)
            save_target_cache(cache_dir, construction, h_data)
        collection_costs = dict(
            collection_seconds=sum(c['collection_seconds'] for c in collection.values()),
            branch_seconds=sum(c.get('branch_seconds', 0.) for c in collection.values()))
        print(f'Collection at data offset {offset}: {collection_costs}; peak memory '
              f'{peak_memory_mb():,.0f} MB', flush=True)
        array_digests = split_array_digests(h_data)
        for init_seed in init_seeds:
            repetition = dict(data_offset=offset, init_seed=init_seed)
            run_rows, digests = {}, set()
            for name, arm in arms.items():
                print(f'=== Arm {name} (data offset {offset}, init seed {init_seed}) ===', flush=True)
                result = run_arm(
                    name, arm, h_data=h_data, construction=construction, reference=reference,
                    population=population, tail=tail, widths=widths, init_seed=init_seed, seeds=seeds,
                    branch_seeds=branch_seeds, branch_batch_size=branch_batch_size, settings_h=settings_h, output_root=output / f'data{offset}_init{init_seed}' / name,
                    device=device, reference_samples=reference_samples, assessment=assessment,
                    count=n_samples, steps=steps, sample_batch_size=sample_batch_size,
                    guided_seed=guided_seed, truth_seed=truth_seed,
                    initialization_max_proposals=initialization_max_proposals, repetition=repetition,
                    collection_costs=collection_costs, source_checkpoint=checkpoint_path,
                    array_digests=array_digests)
                digests.add(result['initial_weights_sha256'])
                tables.append(result['row'])
                curves[(offset, init_seed, name)] = result['curve']
                if residual_reference is not None:
                    residuals = evaluate_residuals(result['model'], residual_reference)
                    residuals.insert(0, 'arm', name)
                    residuals.insert(1, 'data_offset', offset)
                    residuals.insert(2, 'init_seed', init_seed)
                    residual_tables.append(residuals)
                run_rows[name] = dict(
                    h_checkpoint=str(result['h_output'] / 'h.pt'),
                    checkpoint_identity=_checkpoint_identity(result['h_output'] / 'h.pt'),
                    samples=str(result['sampled'].get('sample_output', '')),
                    sample_status=result['sampled']['status'], sample_error=result['sampled']['error'],
                    target_construction=result['construction'],
                    construction_identity=result['construction']['construction_identity'],
                    initial_weights_sha256=result['initial_weights_sha256'])
                if result['sampled']['guided'] is not None and (offset, init_seed) == (data_offsets[0], init_seeds[0]):
                    portfolio_store[name] = dict(unfiltered=result['sampled']['portfolio'],
                                                 filtered=result['sampled']['filtered_portfolio'])
            if len(digests) != 1:
                raise RuntimeError('Arms did not start from identical H weights')
            repetitions.append(dict(**repetition, arms=run_rows, collection=collection,
                                    target_summary=target_summary(h_data),
                                    collection_costs=collection_costs,
                                    target_cache=str(cache_dir), target_arrays=construction_identity(construction)))
    table = pd.DataFrame(tables)
    table.to_csv(output / 'arm_metrics.csv', index=False)
    summary = aggregate_repetitions(table, [c for c in SUMMARY_COLUMNS if c in table])
    summary.to_csv(output / 'arm_summary.csv', index=False)
    residual_table = pd.concat(residual_tables, ignore_index=True) if residual_tables else pd.DataFrame()
    residual_table.to_csv(output / 'late_state_residuals.csv', index=False)
    common = pd.DataFrame()
    if portfolio_store:
        rejection_values = portfolio_values(reference_samples['samples']['rejection'], population.w)
        truth_values = portfolio_values(reference_samples['samples']['truth'], population.w)
        common = common_size_comparison(
            {name: v['filtered'] for name, v in portfolio_store.items()}, population, threshold,
            dict(rejection=rejection_values, truth=truth_values), seed=assessment.matching_seed,
            repeats=subset_repeats)
    common.to_csv(output / 'common_size_comparison.csv', index=False)
    write_json(output / 'soft_target_comparison.json', dict(
        version=EXPERIMENT_VERSION, source_run=str(source_run), frozen_dfm=_checkpoint_identity(checkpoint_path),
        arms={name: arm for name, arm in arms.items()}, repetitions=repetitions,
        reference=dict(source=reference_samples['source'], seconds=reference_seconds,
                       rejection=reference_samples['rejection_diagnostics'],
                       truth_reference=reference_samples['truth_reference_diagnostics']),
        residual_diagnostic=None if residual_reference is None else dict(
            rows=residual_reference['rows'], remaining=[float(r) for r in residual_reference['remaining']],
            branches=residual_branches, n_states=residual_states, seeds=list(residual_seeds),
            note='Network fitting error plus reduced-feature insufficiency; separate from training targets'),
        metrics='arm_metrics.csv', summary='arm_summary.csv', common_size='common_size_comparison.csv',
        residuals='late_state_residuals.csv', peak_memory_mb=peak_memory_mb(),
        total_seconds=time.perf_counter() - run_started, shared_base_splits=True,
        variability_note='sd_init_seed: optimization only (fixed data); sd_data_offset: independent trajectory data'))
    print('Soft-target comparison saved to:', output, flush=True)
    return dict(output=output, table=table, summary=summary, residuals=residual_table, common=common,
                curves=curves, portfolios=portfolio_store, population=population, threshold=threshold,
                reference_samples=reference_samples)


def load_soft_target_results(run_dir):
    """Reload saved tables and, for plotting, the first repetition's arm portfolios."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / 'soft_target_comparison.json').read_text())
    config = json.loads((run_dir / 'config.json').read_text())
    table = pd.read_csv(run_dir / 'arm_metrics.csv')
    first = manifest['repetitions'][0]
    portfolios, curves = {}, {}
    from .conditional import _load_saved_h_results
    for name, arm in first['arms'].items():
        h_output = Path(arm['h_checkpoint']).parent
        curves[name] = pd.read_csv(h_output / 'h_curve.csv')
        if arm['samples']:
            results, _, _ = _load_saved_h_results(h_output, device='cpu', sample_output=arm['samples'])
            w = results['population'].w
            portfolios[name] = dict(unfiltered=portfolio_values(results['guided_samples'], w),
                                    filtered=portfolio_values(results['filtered_guided_samples'], w))
            reference = results
    out = dict(manifest=manifest, config=config, table=table, curves=curves, portfolios=portfolios,
               summary=pd.read_csv(run_dir / 'arm_summary.csv'),
               residuals=pd.read_csv(run_dir / 'late_state_residuals.csv'),
               common=pd.read_csv(run_dir / 'common_size_comparison.csv'))
    if portfolios:
        w = reference['population'].w
        out.update(population=reference['population'], threshold=reference['threshold'],
                   truth=portfolio_values(reference['true_conditional_samples'], w),
                   rejection=portfolio_values(reference['rejection_samples'], w),
                   rejection_diagnostics=reference['rejection_diagnostics'])
    return out
