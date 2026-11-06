"""Frozen-checkpoint H training, conditional sampling, and the separate Gaussian run."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import shutil
from numbers import Integral, Real
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from IPython.display import display, Image

from DFM import UnconditionalDFM
from reference import FrozenDFMReference
from conditioning import ConditioningConfig
from h_model import HNetwork, train_h, evaluate_h, evaluate_h_scores, diagnose_h_terminal, SOFT_TARGET_MODES
from guided_sampling import DFMHGuidance, sample_conditional_initial
from sampling import generate_samples
from synthetic_conditional_truth import (
    population_from_checkpoint, population_metadata, check_population_model, check_population_h,
)
from .configuration import write_json, ConditionalAssessmentConfig, validate_h_seeds, validate_branch_seeds
from .soft_targets import target_construction as build_target_construction
from .reporting import (
    plot_h_validation, conditional_sample_summary, plot_conditional_comparison,
    portfolio_values, conditional_comparisons, extend_conditional_summary,
    plot_conditional_ecdf, plot_h_refinement_comparison, PORTFOLIO_CONVENTION, MATCHING_RULE,
)


def sample_dfm_rejection(reference, weights, threshold, n_accepted, *, steps,
                         batch_size, seed, max_proposals):
    """Condition the supplied frozen numerical sampler by batched rejection.

    Proposals use generate_samples defaults: reference T/t0, parameter dtype and
    device, N(0,I) initialization, no guidance or trajectories. Accept using CPU
    NumPy float64 portfolio products and exact <=, also used for guided filtering.
    Accepted rows remain in generation order. All final-batch hits contribute to
    the observed fraction, even when only a prefix is retained. This fraction is
    descriptive under the stopping rule, not claimed to be unbiased.

    Nonfinite outputs stop the run explicitly: previous finite batches are saved
    for diagnosis but the assessment excludes this failed reference. Never treat
    dropping nonfinite proposals as a valid rejection law. A sampler exception
    from nonfinite integration reports the attempted batch separately because no
    terminal proposals from that batch are available to count.
    """
    for name, value in (('n_accepted', n_accepted), ('steps', steps),
                        ('batch_size', batch_size), ('max_proposals', max_proposals)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**63:
        raise ValueError('seed must be an integer in [0, 2**63)')
    if isinstance(threshold, bool) or not isinstance(threshold, Real) or not np.isfinite(threshold):
        raise ValueError('threshold must be finite')
    if np.iscomplexobj(weights):
        raise ValueError('weights must be real')
    weights = np.asarray(weights, dtype=np.float64)
    portfolio_values(np.empty((0, reference.d)), weights)
    if any(p.requires_grad for p in reference.parameters()):
        raise ValueError('reference parameters must be frozen')
    parameter = next(reference.parameters())
    rng = torch.Generator(device=parameter.device).manual_seed(int(seed))

    def synchronize():
        if parameter.device.type == 'cuda':
            torch.cuda.synchronize(parameter.device)
        elif parameter.device.type == 'mps':
            torch.mps.synchronize()

    synchronize()
    started = time.perf_counter()
    chunks, proposals, hits, retained, failed_batch = [], 0, 0, 0, 0
    status = None
    while retained < n_accepted and proposals < max_proposals:
        count = min(batch_size, max_proposals - proposals)
        try:
            batch = generate_samples(reference, count, num_steps=steps, generator=rng)
        except ValueError as exc:
            if 'nonfinite' not in str(exc).lower():
                raise
            status, failed_batch = 'nonfinite_sampler', count
            break
        batch = batch.detach().cpu().numpy()
        if batch.shape != (count, reference.d):
            raise ValueError('Sampler returned an unexpected terminal shape')
        proposals += count
        finite = np.isfinite(batch).all(axis=1)
        event = np.zeros(count, dtype=bool)
        try:
            event[finite] = portfolio_values(batch[finite], weights) <= threshold
        except ValueError as exc:
            if 'Nonfinite portfolio' not in str(exc):
                raise
            status = 'nonfinite_portfolio'
            break
        hits += int(event.sum())
        if not finite.all():
            status = 'nonfinite_output'
            break
        accepted = batch[event][:n_accepted-retained].copy()
        if len(accepted):
            chunks.append(accepted)
            retained += len(accepted)
    synchronize()
    elapsed = time.perf_counter() - started
    samples = np.concatenate(chunks) if chunks else np.empty((0, reference.d), dtype=torch.empty((), dtype=parameter.dtype).numpy().dtype)
    complete = retained == n_accepted and status is None
    status = status or ('complete' if complete else 'partial' if retained else 'empty')
    diagnostics = dict(
        requested_count=int(n_accepted), total_proposals=proposals, total_event_hits=hits,
        retained_count=retained, observed_acceptance_fraction=hits/proposals if proposals else None,
        sampling_seconds=elapsed, complete=complete, status=status, shortfall=int(n_accepted-retained),
        seed=int(seed), batch_size=int(batch_size), steps=int(steps), max_proposals=int(max_proposals),
        failed_batch_attempted=failed_batch, horizon=float(reference.T), stopping_time=float(reference.t0),
        initialization='N(0,I)', dtype=str(parameter.dtype), device=str(parameter.device),
        portfolio_convention=PORTFOLIO_CONVENTION,
        acceptance_description='Observed event hits / returned terminal proposals, including excess hits; stopping-rule descriptive fraction',
    )
    return samples, diagnostics


def assess_conditional_samples(reference, population, threshold, guided_samples,
                               true_conditional_samples, filtered_guided_samples, *,
                               steps, settings, existing_seeds=()):
    """Independent evaluation only; never consume H trajectory or existing streams."""
    seeds = (settings.rejection_seed, settings.rejection_reference_seed,
             settings.truth_reference_seed, settings.projection_seed, settings.matching_seed)
    # CPU Torch streams can coincide for seeds differing by a multiple of 2**32.
    if {int(seed) % 2**32 for seed in seeds} & {int(seed) % 2**32 for seed in existing_seeds}:
        raise ValueError('Assessment seeds must differ from existing sampling and training seeds')
    samples = dict(guided=guided_samples, truth=true_conditional_samples, filtered=filtered_guided_samples)
    diagnostics = {}
    for name, seed, cap in (
        ('rejection', settings.rejection_seed, settings.rejection_max_proposals),
        ('rejection_reference', settings.rejection_reference_seed, settings.rejection_reference_max_proposals),
    ):
        print(f'Generating independent {name} stream (cap={cap:,})...', flush=True)
        samples[name], diagnostics[name] = sample_dfm_rejection(
            reference, population.w, threshold, settings.rejection_n, steps=steps,
            batch_size=settings.rejection_batch_size, seed=seed, max_proposals=cap,
        )
        d = diagnostics[name]
        print(f"  {d['status']}: retained {d['retained_count']:,}/{d['requested_count']:,}; "
              f"shortfall={d['shortfall']:,}; hits={d['total_event_hits']:,}/{d['total_proposals']:,}; "
              f"observed proposal acceptance={d['observed_acceptance_fraction']}", flush=True)
    truth_started = time.perf_counter()
    truth_error = None
    try:
        samples['truth_reference'] = population.sample(
            len(true_conditional_samples), np.random.default_rng(settings.truth_reference_seed), c=threshold,
        )
        if not np.isfinite(samples['truth_reference']).all():
            raise FloatingPointError('Nonfinite independent conditional truth')
    except FloatingPointError as exc:
        truth_error = str(exc)
        samples['truth_reference'] = np.empty((0, population.d), dtype=np.float64)
        print(f'Independent conditional truth unavailable: {truth_error}', flush=True)
    truth_diagnostics = dict(
        seed=settings.truth_reference_seed, requested_count=len(true_conditional_samples),
        retained_count=len(samples['truth_reference']),
        shortfall=len(true_conditional_samples)-len(samples['truth_reference']),
        status='complete' if truth_error is None else 'unavailable_nonfinite',
        complete=truth_error is None, error=truth_error,
        sampling_seconds=time.perf_counter()-truth_started,
    )
    return _conditional_assessment(samples, population, settings, diagnostics, truth_diagnostics)


def _conditional_assessment(samples, population, settings, diagnostics, truth_diagnostics):
    """Evaluate supplied draws, allowing both H variants to share reference samples."""
    # Failed numerical runs are retained in artifacts but cannot define a reference.
    evaluation_samples = dict(samples)
    for name, d in diagnostics.items():
        if d['status'].startswith('nonfinite'):
            evaluation_samples[name] = samples[name][:0]
    metrics, directions = conditional_comparisons(
        evaluation_samples, population.w, projection_seed=settings.projection_seed,
        matching_seed=settings.matching_seed, projections=settings.projections,
    )
    for name, d in diagnostics.items():
        metrics[f'{name}_status'] = d['status']
    metrics['truth_reference_status'] = truth_diagnostics['status']
    # Available saved rows can differ from eligible reference rows after a failed
    # numerical stream. Make both counts visible instead of hiding that evidence.
    metrics['n_left_saved'] = metrics.left.map(lambda name: len(samples[name]))
    metrics['n_right_saved'] = metrics.right.map(lambda name: len(samples[name]))
    return dict(samples=samples, rejection_diagnostics=diagnostics, comparison_metrics=metrics,
                projection_directions=directions, assessment_config=asdict(settings),
                truth_reference_diagnostics=truth_diagnostics)


def load_frozen_dfm(checkpoint_path, case, device):
    """Restore the saved architecture and population, then freeze the selected DFM."""
    h_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    h_dfm_config = h_checkpoint["config"]
    h_population = population_from_checkpoint(h_dfm_config["d"], h_dfm_config["k"], case, h_checkpoint)
    h_base_model = UnconditionalDFM(
        h_dfm_config["d"], h_dfm_config["k"], sigma_max=2.0,
        T=h_dfm_config["T"], t0=h_dfm_config["t0"],
        hidden_dims=[h_dfm_config["width"], h_dfm_config["width"]],
    ).to(device)
    h_base_model.load_state_dict(h_checkpoint["state"])
    h_reference = FrozenDFMReference(h_base_model)

    check_population_model(h_population, h_reference)
    return h_checkpoint, h_dfm_config, h_population, h_base_model, h_reference


def reuse_frozen_unconditional_run(source, output, config):
    """Copy a reviewed DFM run into a new H experiment without training the DFM."""
    source, output = Path(source).resolve(), Path(output)
    for seed in config.seeds:
        name = f'gaussian_seed{seed}_primary'
        checkpoint, saved_config, _, _, _ = load_frozen_dfm(source/name/'dfm.pt', 'gaussian', config.device)
        expected = asdict(config)
        for key in ('d', 'k', 'T', 't0', 'steps', 'width', 'dfm_initialization', 'dfm_time_sampling',
                    'dfm_loading_lr_scale', 'dfm_loading_warmup_epochs', 'lr', 'dfm_epochs', 'n_train', 'n_val'):
            if saved_config.get(key) != expected[key]:
                raise ValueError(f'Frozen checkpoint configuration mismatch for {key}')
        shutil.copytree(source/name, output/name)
    metrics = pd.read_csv(source/'metrics.csv')
    metrics = metrics.loc[(metrics.case == 'gaussian') & metrics.seed.isin(config.seeds)]
    metrics.to_csv(output/'metrics.csv', index=False)
    write_json(output/'frozen_reference.json', dict(source=str(source), dfm_retrained=False,
        checkpoints={str(seed): hashlib.sha256((source/f'gaussian_seed{seed}_primary'/'dfm.pt').read_bytes()).hexdigest()
                     for seed in config.seeds}))
    print('Reusing frozen DFM and its saved unconditional assessment:', source, flush=True)
    return metrics, metrics.select_dtypes(include=np.number)


def initialize_h(reference, population, tail, widths, initialization_seed, *, log_remaining_time=False):
    """Initialize H inside fork_rng, preserving the RNG state after DFM loading."""
    h_event = ConditioningConfig(
        w=population.w,
        C=population.threshold(tail),
    )
    # Preserve the notebook's global RNG state while initializing H reproducibly.
    with torch.random.fork_rng():
        torch.manual_seed(initialization_seed)
        h_model = HNetwork(reference, h_event, hidden_dims=widths, log_remaining_time=log_remaining_time)
    check_population_h(population, h_model)
    h_guidance = DFMHGuidance(h_model)
    return h_event, h_model, h_guidance


def describe_h_model(h_model, checkpoint, dfm_config, case, seed):
    print(h_model.network)
    print(f"Frozen DFM: {case}, seed={seed}, selected epoch={checkpoint['selected_epoch']}")
    print(f"Event: equal-weight portfolio return <= {h_model.C:.6f}")
    print(f"H input: {dfm_config['k'] + 1} reduced coordinates + time"
          + (" + log remaining time" if h_model.log_remaining_time else "")
          + "; output: event probability")
    print("H is initialized only; train it before using h_guidance for sampling.")


def h_rng(name, seeds, device):
    """Construct a fresh named H generator; never share mutable stream state."""
    return torch.Generator(device=device).manual_seed(seeds[name])


# Collection settings are never train_h keyword arguments.
SAMPLER_ONLY_OPTIONS = ('branch_count', 'branch_seeds', 'branch_batch_size')


def split_h_options(options):
    """Separate train_h keywords from target-collection settings (branch_*)."""
    training = dict(options or {})
    sampler = {name: training.pop(name) for name in SAMPLER_ONLY_OPTIONS if name in training}
    return training, sampler


def sample_h_training_data(h_model, split_counts, *, steps, seeds, path_batch_size, device,
                           soft_target_mode='off', soft_target_window=None, branch_count=None,
                           branch_seeds=None, branch_batch_size=None, diagnostics=None):
    """Draw independent unguided trajectory splits and check both label classes.

    soft_target_mode='continuation' also constructs frozen branch-average
    probability targets per split from named branch streams (branch_seeds maps
    split names to seeds distinct from every base stream); the base paths do not
    depend on them. diagnostics, if given, receives per-split collection cost.
    """
    split_counts = tuple(split_counts)
    names = tuple(split for split, _ in split_counts)
    base_seeds = validate_h_seeds(seeds, names=names)
    if soft_target_mode not in SOFT_TARGET_MODES:
        raise ValueError(f'soft_target_mode must be one of {SOFT_TARGET_MODES}')
    continuation = soft_target_mode == 'continuation'
    if continuation:
        branch_seeds = validate_branch_seeds(branch_seeds, seeds, names=names)
    elif branch_count is not None or branch_seeds is not None or branch_batch_size is not None:
        raise ValueError("branch options require soft_target_mode='continuation'")
    h_data = {}
    for split, count in split_counts:
        print(f"Generating {count:,} {split} trajectories...", flush=True)
        split_diagnostics = {}
        started = time.perf_counter()
        branch_options = dict(soft_target_mode=soft_target_mode, soft_target_window=soft_target_window)
        if continuation:
            branch_options.update(branch_count=branch_count, branch_batch_size=branch_batch_size,
                                  branch_generator=h_rng(split, branch_seeds, device))
        h_data[split] = h_model.sample_trajectories(
            count, num_steps=steps, generator=h_rng(split, base_seeds, device),
            batch_size=path_batch_size, diagnostics=split_diagnostics, **branch_options,
        )
        split_diagnostics['collection_seconds'] = time.perf_counter() - started
        if diagnostics is not None:
            diagnostics[split] = split_diagnostics
        # Use the represented time intervals to avoid float32 rounding mismatches.
        time_edges = torch.cat((h_data[split].times, h_data[split].times.new_tensor([h_model.horizon])))
        h_data[split].weights = time_edges.diff() / h_model.horizon
        positives = int(h_data[split].labels.sum().item())
        print(f"  Terminal event: {positives:,}/{count:,} ({positives / count:.2%})", flush=True)
        if continuation:
            print(f"  Branch targets: {len(h_data[split].soft_target_indices)} rows x {branch_count} "
                  f"branches in {split_diagnostics.get('branch_seconds', 0.):.1f}s", flush=True)
        if positives == 0 or positives == count:
            raise RuntimeError(f"{split} has only one label class; increase its trajectory count before training H.")
    return h_data


def fit_and_evaluate_h(
    h_model, h_reference, h_data, frozen_before, *,
    epochs, batch_size, lr, seeds, device, training_options=None, target_construction=None,
):
    """Select H by validation loss, evaluate held-out loss, and verify the frozen DFM.

    training_options may also carry target-collection keys (branch_*); these are
    never forwarded to train_h. The optional target_construction record must
    agree with the soft-target mode, window and branch count requested here.
    """
    options, sampler_options = split_h_options(training_options)
    mode, window = options.get('soft_target_mode', 'off'), options.get('soft_target_window')
    if target_construction is not None:
        if (target_construction['mode'], target_construction['window']) != (mode, window):
            raise ValueError('Target construction does not match the requested soft-target mode/window')
        if ('branch_count' in sampler_options
                and sampler_options['branch_count'] != target_construction['branch_count']):
            raise ValueError('Target construction does not match the requested branch count')
    h_selection = {}
    print(f"Training H for {epochs} epochs; selecting by validation loss...", flush=True)
    h_train_history = train_h(
        h_model, h_data["train"], epochs=epochs, batch_size=batch_size, lr=lr,
        generator=h_rng("shuffle", seeds, device), validation_data=h_data["validation"],
        selection=h_selection,
        **options,
    )
    h_model.eval()
    h_training_probability = float(h_data["train"].labels.mean())
    h_test_metrics = evaluate_h(
        h_model, h_data["test"], constant_probability=h_training_probability,
        batch_size=batch_size,
    )
    h_selection['training_options'] = dict(options)
    if target_construction is not None:
        h_selection['target_construction'] = target_construction
    diagnostic_options = {key: options.get(key, 0.)
                          for key in ('late_time_fraction', 'last_step_fraction')}
    diagnostic_options.update(soft_target_mode=mode, soft_target_window=window)
    h_selection['validation_diagnostics'] = diagnose_h_terminal(h_model, h_data['validation'], **diagnostic_options)
    h_test_metrics['terminal_diagnostics'] = diagnose_h_terminal(h_model, h_data['test'], **diagnostic_options)
    if mode != 'off':
        score_options = dict(diagnostic_options, use_last_step_targets=True, batch_size=batch_size)
        h_selection['validation_scores'] = evaluate_h_scores(h_model, h_data['validation'], **score_options)
        h_test_metrics['configured_scores'] = evaluate_h_scores(h_model, h_data['test'], **score_options)
    for name, value in h_reference.state_dict().items():
        torch.testing.assert_close(value, frozen_before[name], rtol=0, atol=0)
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in h_reference.parameters())
    return h_selection, h_train_history, h_training_probability, h_test_metrics


def save_h_training(
    output_root, h_model, h_population, h_data, h_selection,
    h_train_history, h_training_probability, h_test_metrics, *,
    case, seed, tail, n_train, n_validation, n_test, epochs, lr, batch_size,
    path_batch_size, steps, stream_seeds, initialization_seed,
    reference_checkpoint, started, target_construction=None,
):
    """Save the selected checkpoint, trajectory settings, curves, and held-out metrics."""
    h_curve = pd.DataFrame({
        "epoch": np.arange(1, len(h_train_history) + 1),
        "train": h_train_history,
        "validation": h_selection["validation_history"],
    })
    h_training_config = dict(
        **population_metadata(h_population, case),
        case=case, seed=seed, tail=tail, threshold=h_model.C,
        n_train=n_train, n_validation=n_validation, n_test=n_test,
        epochs=epochs, lr=lr, batch_size=batch_size, path_batch_size=path_batch_size,
        steps=steps, stream_seeds=stream_seeds, initialization_seed=initialization_seed,
        selected_epoch=h_selection["best_epoch"],
        reference_checkpoint=str(reference_checkpoint),
        training_options=h_selection.get('training_options', {}),
        log_remaining_time=h_model.log_remaining_time,
    )
    if target_construction is not None:
        # Recorded only for soft-target modes; mode 'off' runs omit this key.
        h_training_config['target_construction'] = target_construction
    output = output_root / f"h_{case}_seed{seed}_tail{tail}" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    h_model.save_checkpoint(output / "h.pt", metadata=h_training_config)
    h_curve.to_csv(output / "h_curve.csv", index=False)
    write_json(output / "training.json", h_training_config)
    write_json(output / 'validation_diagnostics.json', h_selection['validation_diagnostics'])
    write_json(output / "test_metrics.json", dict(
        **h_test_metrics, training_event_rate=h_training_probability,
        test_event_rate=float(h_data["test"].labels.mean()),
        seconds=time.perf_counter() - started,
    ))
    return h_curve, h_training_config, output


def load_trained_h(output, device):
    """Reload the saved selected H, validating population provenance and portfolio weights."""
    if not (Path(output) / "h.pt").exists():
        raise RuntimeError("Run the H-training cell first to save its selected checkpoint.")
    h_sampling_model, h_sampling_metadata = HNetwork.load_checkpoint(
        output / "h.pt", device=device,
    )
    h_sampling_guidance = DFMHGuidance(h_sampling_model)
    conditional_population = population_from_checkpoint(
        h_sampling_model.reference.d, h_sampling_model.reference._score.k,
        h_sampling_metadata["case"], h_sampling_metadata,
    )
    check_population_h(conditional_population, h_sampling_model)
    np.testing.assert_allclose(
        np.asarray(h_sampling_model.config.w), conditional_population.w, rtol=0, atol=1e-12,
    )
    return h_sampling_model, h_sampling_metadata, h_sampling_guidance, conditional_population


def sample_h_guided(
    h_sampling_model, h_sampling_guidance, *,
    n_samples, batch_size, steps, seed, guidance_scale=1.0,
    initialization_max_proposals=None, diagnostics=None,
):
    """Start from the learned H tilt, then retain every guided terminal output.

    The initialization proposal cap applies once to the entire requested sample,
    not separately to each integration batch. Exhaustion raises without falling
    back to Gaussian starts. The optional diagnostics mapping records the actual
    initialization settings and proposal counts without changing the array return.
    """
    for name, value in (('n_samples', n_samples), ('batch_size', batch_size), ('steps', steps)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if (isinstance(guidance_scale, bool) or not isinstance(guidance_scale, Real)
            or not np.isfinite(guidance_scale) or guidance_scale < 0):
        raise ValueError('guidance_scale must be a finite nonnegative real number')
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**63:
        raise ValueError('seed must be an integer in [0, 2**63)')
    sampling_device = next(h_sampling_model.network.parameters()).device
    guided_rng = torch.Generator(device=sampling_device).manual_seed(int(seed))
    if initialization_max_proposals is None:
        initialization_max_proposals = max(100 * n_samples, batch_size)

    print(f"Generating {n_samples:,} H-guided DFM samples...", flush=True)
    starts, initialization = sample_conditional_initial(
        h_sampling_model, n_samples, batch_size=batch_size,
        max_proposals=initialization_max_proposals, generator=guided_rng,
    )
    guided_chunks = []
    for start in range(0, n_samples, batch_size):
        initial_state = starts[start:start + batch_size]
        batch = generate_samples(
            h_sampling_model.reference, len(initial_state), initial_state=initial_state,
            num_steps=steps, generator=guided_rng,
            guidance_fn=h_sampling_guidance, guidance_scale=guidance_scale,
        )
        guided_chunks.append(batch.detach().cpu().numpy())
    guided_samples = np.concatenate(guided_chunks, axis=0)
    if diagnostics is not None:
        diagnostics.update(
            initialization=dict(initialization, max_proposals=int(initialization_max_proposals),
                                batch_size=int(batch_size)),
            seed=int(seed), steps=int(steps), batch_size=int(batch_size),
            guidance_scale=float(guidance_scale), terminal_filtering=False,
        )
    return guided_samples


def filter_conditional_samples(
    guided_samples, true_conditional_samples, population, threshold, n_samples, dimension,
):
    """Validate both draws, then retain event rows without replacement sampling."""
    guided_portfolio = portfolio_values(guided_samples, population.w)
    true_conditional_portfolio = portfolio_values(true_conditional_samples, population.w)
    assert guided_samples.shape == true_conditional_samples.shape == (n_samples, dimension)
    assert np.isfinite(guided_samples).all() and np.isfinite(true_conditional_samples).all()
    assert np.all(true_conditional_portfolio <= threshold + 1e-10)

    # Keep the original guided draw and also discard rows outside the event.
    # No resampling: the filtered comparison has a smaller, explicit sample count.
    guided_event_mask = guided_portfolio <= threshold
    filtered_guided_samples = guided_samples[guided_event_mask]
    filtered_guided_portfolio = guided_portfolio[guided_event_mask]
    return (
        guided_portfolio, true_conditional_portfolio, guided_event_mask,
        filtered_guided_samples, filtered_guided_portfolio,
    )


def _checkpoint_identity(path):
    if path is None:
        return None
    path = Path(path)
    digest = None
    if path.is_file():
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    return dict(path=str(path), sha256=digest)


def _validate_checkpoint_identity(identity, path):
    """Check saved provenance before associating stored samples with a model."""
    if identity is None:
        return
    if identity.get('path') is not None and Path(identity['path']).resolve() != Path(path).resolve():
        raise ValueError(f'Saved checkpoint path does not match {path}')
    if identity.get('sha256') is not None:
        if _checkpoint_identity(path)['sha256'] != identity['sha256']:
            raise ValueError(f'Saved checkpoint hash does not match {path}')


def save_conditional_samples(
    h_output, h_sampling_metadata, conditional_population, conditional_threshold,
    guided_samples, true_conditional_samples, filtered_guided_samples,
    guided_event_mask, conditional_sample_summary, *,
    n_samples, steps, batch_size, guided_seed, truth_seed,
    assessment=None, guided_sampling_diagnostics=None,
):
    """Save both original and filtered draws with their exact masks and sampling metadata."""
    conditional_output = h_output / ("samples_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    conditional_output.mkdir(parents=True, exist_ok=False)
    extra_samples = {} if assessment is None else dict(
        rejection_samples=assessment['samples']['rejection'],
        independent_rejection_samples=assessment['samples']['rejection_reference'],
        independent_true_conditional_samples=assessment['samples']['truth_reference'],
        projection_directions=assessment['projection_directions'],
    )
    np.savez_compressed(
        conditional_output / "conditional_samples.npz",
        guided_samples=guided_samples, true_conditional_samples=true_conditional_samples,
        filtered_guided_samples=filtered_guided_samples, guided_event_mask=guided_event_mask,
        weights=conditional_population.w, threshold=conditional_threshold,
        **extra_samples,
    )
    conditional_sample_summary.to_csv(conditional_output / "sample_summary.csv", index=False)
    write_json(conditional_output / "sampling.json", dict(
        **population_metadata(conditional_population, h_sampling_metadata["case"]),
        h_checkpoint=str(h_output / "h.pt"), case=h_sampling_metadata["case"],
        n=n_samples, steps=steps, batch_size=batch_size,
        guided_seed=guided_seed, truth_seed=truth_seed,
        guided_sampling=guided_sampling_diagnostics,
        threshold=conditional_threshold, guidance_scale=1.0,
        filtered_n=int(guided_event_mask.sum()), rejected_n=int((~guided_event_mask).sum()),
        filter_rule="portfolio <= threshold; no resampling",
        event_weights=conditional_population.w.tolist(),
        reference_checkpoint=h_sampling_metadata.get('reference_checkpoint'),
        checkpoint_identity=dict(
            h=_checkpoint_identity(h_output / 'h.pt'),
            dfm=_checkpoint_identity(h_sampling_metadata.get('reference_checkpoint')),
        ),
        portfolio_convention=PORTFOLIO_CONVENTION,
        matching_rule=MATCHING_RULE,
        assessment_config=assessment['assessment_config'] if assessment else None,
        rejection_diagnostics=assessment['rejection_diagnostics'] if assessment else None,
        truth_reference_diagnostics=assessment['truth_reference_diagnostics'] if assessment else None,
    ))
    if assessment is not None:
        metrics = assessment['comparison_metrics']
        metrics.to_csv(conditional_output / 'conditional_metrics.csv', index=False)
        # pandas converts missing metrics to JSON null, compatible with strict JSON.
        write_json(conditional_output / 'conditional_metrics.json', json.loads(metrics.to_json(orient='records', double_precision=15)))
        write_json(conditional_output / 'rejection_diagnostics.json', assessment['rejection_diagnostics'])
    print("Samples saved to:", conditional_output)
    return conditional_output


def _load_saved_h_results(h_output, *, device, sample_output=None):
    """Restore an H run and its completed sample draw without generating samples."""
    h_output = Path(h_output).resolve()
    model, metadata, _, population = load_trained_h(h_output, device)
    training = json.loads((h_output / 'training.json').read_text())
    if training != metadata:
        raise ValueError('training.json does not match the saved H checkpoint metadata')
    if sample_output is None:
        completed = sorted((folder for folder in h_output.glob('samples_*')
                            if (folder / 'conditional_samples.npz').is_file()
                            and (folder / 'sampling.json').is_file()), reverse=True)
        if not completed:
            raise RuntimeError(f'No completed conditional samples found in {h_output}')
        sample_output = completed[0]
    sample_output = Path(sample_output).resolve()
    if sample_output.parent != h_output:
        raise ValueError('Saved conditional samples must belong to the selected H directory')
    sampling = json.loads((sample_output / 'sampling.json').read_text())
    if Path(sampling['h_checkpoint']).resolve() != h_output / 'h.pt':
        raise ValueError('Saved samples reference a different H checkpoint')
    identities = sampling.get('checkpoint_identity') or {}
    _validate_checkpoint_identity(identities.get('h'), h_output / 'h.pt')
    reference_path = metadata.get('reference_checkpoint')
    if sampling.get('reference_checkpoint') != reference_path:
        raise ValueError('Saved samples and H metadata reference different DFM checkpoints')
    if reference_path is not None:
        _validate_checkpoint_identity(identities.get('dfm'), reference_path)
    if sampling['steps'] != metadata['steps'] or sampling['threshold'] != model.C:
        raise ValueError('Saved sampling grid or event does not match the selected H')
    with np.load(sample_output / 'conditional_samples.npz', allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    np.testing.assert_array_equal(arrays['weights'], population.w)
    if float(arrays['threshold']) != model.C:
        raise ValueError('Saved sample threshold does not match the selected H')
    _, _, event_mask, filtered, _ = filter_conditional_samples(
        arrays['guided_samples'], arrays['true_conditional_samples'], population,
        model.C, sampling['n'], model.reference.d,
    )
    np.testing.assert_array_equal(arrays['guided_event_mask'], event_mask)
    np.testing.assert_array_equal(arrays['filtered_guided_samples'], filtered)
    samples = dict(
        guided=arrays['guided_samples'], filtered=filtered, truth=arrays['true_conditional_samples'],
        rejection=arrays['rejection_samples'], rejection_reference=arrays['independent_rejection_samples'],
        truth_reference=arrays['independent_true_conditional_samples'],
    )
    settings = ConditionalAssessmentConfig(**sampling['assessment_config'])
    rejection_diagnostics = sampling['rejection_diagnostics']
    truth_diagnostics = sampling['truth_reference_diagnostics']
    metrics_path = sample_output / 'conditional_metrics.csv'
    metrics = (pd.read_csv(metrics_path) if metrics_path.is_file() else
               _conditional_assessment(samples, population, settings,
                                       rejection_diagnostics, truth_diagnostics)['comparison_metrics'])
    results = dict(
        h_model=model, guided_samples=samples['guided'],
        guided_sampling_diagnostics=sampling.get('guided_sampling') or {},
        filtered_guided_samples=filtered, guided_event_mask=event_mask,
        true_conditional_samples=samples['truth'], rejection_samples=samples['rejection'],
        independent_rejection_samples=samples['rejection_reference'],
        independent_true_conditional_samples=samples['truth_reference'],
        rejection_diagnostics=rejection_diagnostics, truth_reference_diagnostics=truth_diagnostics,
        comparison_metrics=metrics, projection_directions=arrays['projection_directions'],
        assessment_config=asdict(settings), population=population, threshold=model.C,
        validation_curve=pd.read_csv(h_output / 'h_curve.csv')[["epoch", "validation"]],
        test_metrics=json.loads((h_output / 'test_metrics.json').read_text()),
        sample_summary=pd.read_csv(sample_output / 'sample_summary.csv'),
        h_output=h_output, sample_output=sample_output,
    )
    return results, training, sampling


def load_saved_h_results(h_output, *, device='cpu', sample_output=None):
    """Public loader: a saved H run with its newest completed conditional draw (no sampling)."""
    return _load_saved_h_results(h_output, device=device, sample_output=sample_output)[0]


def _raw_h_options():
    """CDG-ML (Guo et al., 2026): all five refinements off."""
    return dict(normalize_inputs=False, loss='brier', late_time_fraction=0.,
                last_step_fraction=0., use_last_step_targets=False, times_per_trajectory=None)


def _normalized_training_options(options):
    """Resolve options saved before soft targets existed to explicit 'off' defaults.

    Saved files are not rewritten; normalization applies only to the comparison,
    so checkpoints without soft-target keys remain valid.
    """
    if options is None:
        return None
    return dict(dict(soft_target_mode='off', soft_target_window=None), **options)


def _check_raw_h(raw_model, raw_metadata, refined_results, training):
    """A reusable baseline must describe exactly this frozen-DFM experiment."""
    if (raw_model.log_remaining_time or raw_model.normalization_times.numel()
            or _normalized_training_options(raw_metadata.get('training_options'))
            != _normalized_training_options(_raw_h_options())
            or raw_model.hidden_dims != refined_results['h_model'].hidden_dims):
        raise ValueError('Saved CDG-ML H does not disable the five refinements')
    for key in ('case', 'seed', 'tail', 'threshold', 'n_train', 'n_validation', 'n_test',
                'epochs', 'lr', 'batch_size', 'path_batch_size', 'steps', 'stream_seeds',
                'initialization_seed', 'reference_checkpoint'):
        if raw_metadata.get(key) != training.get(key):
            raise ValueError(f'Saved CDG-ML H configuration mismatch for {key}')
    for name, value in refined_results['h_model'].reference.state_dict().items():
        torch.testing.assert_close(raw_model.reference.state_dict()[name], value, rtol=0, atol=0)


def _display_h_comparison(raw_results, refined_results, output_root):
    """Show the H validation curves before the single performance figure."""
    for result, title in (
        (raw_results, 'CDG-ML (no refinements): validation loss'),
        (refined_results, 'Refined CDG-ML (5 refinements): validation loss'),
    ):
        h_output = Path(result['h_output'])
        training = json.loads((h_output / 'training.json').read_text())
        plot_h_validation(result['validation_curve'], training['selected_epoch'],
                          h_output, title=title, show=False)
        display(Image(filename=str(h_output / 'validation_loss.png')))
    path = plot_h_refinement_comparison(raw_results, refined_results, output_root, show=False)
    display(Image(filename=str(path)))
    return path


def _compare_raw_h(results, training, sampling, *, data=None, device='cpu'):
    """Train or restore the baseline while retaining the supplied refined draw."""
    h_output = Path(results['h_output']).resolve()
    output_root = h_output.parents[1]
    manifest_path = output_root / 'h_comparison.json'
    cached_raw = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        variants = manifest.get('variants', {})
        refined = variants.get('refined', {})
        if Path(refined.get('h_checkpoint', '')).resolve() == h_output / 'h.pt':
            _validate_checkpoint_identity(refined.get('checkpoint_identity'), h_output / 'h.pt')
            raw = variants['raw']
            raw_output = Path(raw['h_checkpoint']).resolve().parent
            if (raw_output / 'h.pt').is_file():
                _validate_checkpoint_identity(raw.get('checkpoint_identity'), raw_output / 'h.pt')
                raw_model, raw_metadata, raw_guidance, _ = load_trained_h(raw_output, device)
                _check_raw_h(raw_model, raw_metadata, results, training)
                cached_raw = (raw_model, raw_metadata, raw_guidance, raw_output)
                raw_sample_output = Path(raw['samples']).resolve()
                if (Path(refined['samples']).resolve() == Path(results['sample_output']).resolve()
                        and (raw_sample_output / 'conditional_samples.npz').is_file()
                        and (raw_sample_output / 'sampling.json').is_file()):
                    print('Reusing saved CDG-ML checkpoint and comparison samples...', flush=True)
                    raw_results, _, _ = _load_saved_h_results(
                        raw_output, device=device, sample_output=raw_sample_output,
                    )
                    for key in ('true_conditional_samples', 'rejection_samples',
                                'independent_rejection_samples', 'independent_true_conditional_samples',
                                'projection_directions'):
                        np.testing.assert_array_equal(raw_results[key], results[key])
                    results['comparison_path'] = _display_h_comparison(raw_results, results, output_root)
                    results['raw_h_results'] = raw_results
                    return results
    reference = results['h_model'].reference
    population = results['population']
    stream_seeds = validate_h_seeds(training['stream_seeds'])
    steps = training['steps']
    raw_options = _raw_h_options()
    if cached_raw is None:
        print('Training CDG-ML (no refinements); retaining the saved DFM and Refined CDG-ML...', flush=True)
        started = time.perf_counter()
        _, raw_model, _ = initialize_h(reference, population, training['tail'],
            results['h_model'].hidden_dims, training['initialization_seed'], log_remaining_time=False)
        raw_device = next(raw_model.network.parameters()).device
        if data is None:
            print('Regenerating the original H trajectory splits from their saved seeds...', flush=True)
            data = sample_h_training_data(raw_model,
                (("train", training['n_train']), ("validation", training['n_validation']),
                 ("test", training['n_test'])), steps=steps, seeds=stream_seeds,
                path_batch_size=training['path_batch_size'], device=raw_device)
        frozen_before = {name: value.detach().clone() for name, value in reference.state_dict().items()}
        selection, history, probability, test_metrics = fit_and_evaluate_h(
            raw_model, reference, data, frozen_before, epochs=training['epochs'],
            batch_size=training['batch_size'], lr=training['lr'], seeds=stream_seeds,
            device=raw_device, training_options=raw_options)
        curve, _, raw_output = save_h_training(
            output_root / 'raw_h', raw_model, population, data, selection, history, probability,
            test_metrics, case=training['case'], seed=training['seed'], tail=training['tail'],
            n_train=training['n_train'], n_validation=training['n_validation'], n_test=training['n_test'],
            epochs=training['epochs'], lr=training['lr'], batch_size=training['batch_size'],
            path_batch_size=training['path_batch_size'], steps=steps, stream_seeds=stream_seeds,
            initialization_seed=training['initialization_seed'],
            reference_checkpoint=training['reference_checkpoint'], started=started)
        raw_model, raw_metadata, raw_guidance, _ = load_trained_h(raw_output, device)
    else:
        print('Reusing saved CDG-ML checkpoint; generating its comparison samples...', flush=True)
        raw_model, raw_metadata, raw_guidance, raw_output = cached_raw
        curve = pd.read_csv(raw_output / 'h_curve.csv')
        test_metrics = json.loads((raw_output / 'test_metrics.json').read_text())
    count = len(results['guided_samples'])
    initialization = (sampling.get('guided_sampling') or {}).get('initialization') or {}
    raw_diagnostics = {}
    print('Sampling CDG-ML; reusing the saved refined, truth, and rejection draws...', flush=True)
    raw_samples = sample_h_guided(raw_model, raw_guidance, n_samples=count,
        batch_size=sampling['batch_size'], steps=steps, seed=sampling['guided_seed'],
        initialization_max_proposals=initialization.get('max_proposals'), diagnostics=raw_diagnostics)
    raw_portfolio, truth_portfolio, raw_mask, raw_filtered, filtered_portfolio = filter_conditional_samples(
        raw_samples, results['true_conditional_samples'], population, results['threshold'], count, reference.d)
    samples = dict(guided=raw_samples, filtered=raw_filtered, truth=results['true_conditional_samples'],
        rejection=results['rejection_samples'], rejection_reference=results['independent_rejection_samples'],
        truth_reference=results['independent_true_conditional_samples'])
    assessment = _conditional_assessment(samples, population,
        ConditionalAssessmentConfig(**results['assessment_config']), results['rejection_diagnostics'],
        results['truth_reference_diagnostics'])
    summary = conditional_sample_summary(raw_portfolio, filtered_portfolio, truth_portfolio,
                                         results['threshold'], count)
    summary = extend_conditional_summary(summary, samples, population.w, results['threshold'],
                                          assessment['rejection_diagnostics'])
    raw_sample_output = save_conditional_samples(raw_output, raw_metadata, population,
        results['threshold'], raw_samples, results['true_conditional_samples'], raw_filtered,
        raw_mask, summary, n_samples=count, steps=steps, batch_size=sampling['batch_size'],
        guided_seed=sampling['guided_seed'], truth_seed=sampling['truth_seed'],
        assessment=assessment, guided_sampling_diagnostics=raw_diagnostics)
    raw_results = dict(results, h_model=raw_model, guided_samples=raw_samples,
        guided_sampling_diagnostics=raw_diagnostics, filtered_guided_samples=raw_filtered,
        guided_event_mask=raw_mask, comparison_metrics=assessment['comparison_metrics'],
        validation_curve=curve[["epoch", "validation"]], test_metrics=test_metrics,
        sample_summary=summary, h_output=raw_output, sample_output=raw_sample_output)
    comparison_path = _display_h_comparison(raw_results, results, output_root)
    write_json(manifest_path, dict(variants={
        'raw': dict(h_checkpoint=str(raw_output / 'h.pt'), samples=str(raw_sample_output),
                    checkpoint_identity=_checkpoint_identity(raw_output / 'h.pt'),
                    training_options=raw_options, log_remaining_time=False),
        'refined': dict(h_checkpoint=str(h_output / 'h.pt'), samples=str(results['sample_output']),
                        checkpoint_identity=_checkpoint_identity(h_output / 'h.pt'),
                        training_options=training['training_options'],
                        log_remaining_time=results['h_model'].log_remaining_time)},
        reference_checkpoint=training['reference_checkpoint'], stream_seeds=stream_seeds,
        shared_trajectory_splits=True, shared_reference_samples=True,
        guided_seed=sampling['guided_seed'], truth_seed=sampling['truth_seed']))
    results['raw_h_results'] = raw_results
    results['comparison_path'] = comparison_path
    print('CDG-ML versus Refined CDG-ML comparison saved to:', comparison_path, flush=True)
    return results


def compare_saved_h_run(h_output, *, device='cpu'):
    """Add CDG-ML to a completed refined run, reusing all saved refined/reference draws.

    Only a missing baseline is trained. Its trajectory splits are
    regenerated with the saved frozen DFM, seeds, integration grid, and batching.
    A matching saved comparison is reused on later calls. Neither the DFM nor
    the Refined CDG-ML is retrained or resampled.
    """
    print('Loading saved Refined CDG-ML and its most recent completed conditional draw...', flush=True)
    results, training, sampling = _load_saved_h_results(h_output, device=device)
    return _compare_raw_h(results, training, sampling, device=device)


def run_gaussian_conditional_simulation(
    config, output_root, *, epochs=200, n_train=10_000,
    n_validation=2_000, n_test=2_000, n_samples=None,
    tail=0.05, widths=(64, 64), initialization_seed=2027,
    lr=1e-3, batch_size=64, path_batch_size=256,
    train_seeds=None, sample_batch_size=256, guided_seed=42001, truth_seed=42002,
    assessment_settings=None, initialization_max_proposals=None,
    h_training_options=None, log_remaining_time=False, validation_sampling_n=0,
    compare_raw_h=True,
):
    """Repeat all conditional stages for the separate Gaussian checkpoint and H.

    H-weighted starts and a bounded proposal budget define the main guided run.
    All model and sample variables are local to this Gaussian run.
    compare_raw_h also trains CDG-ML (H without any of the five refinements), sharing
    the frozen DFM, trajectory splits, and conditional reference samples.
    """
    if train_seeds is None:
        train_seeds = {"train": 32001, "validation": 32002, "test": 32003, "shuffle": 32004}
    train_seeds = validate_h_seeds(train_seeds)
    output_root = Path(output_root)
    case, seed = "gaussian", config.seeds[0]
    checkpoint_path = output_root / f"{case}_seed{seed}_primary" / "dfm.pt"
    checkpoint, dfm_config, population, base_model, reference = load_frozen_dfm(
        checkpoint_path, case, config.device,
    )
    print(f"Assets: {population.d}; population factors: {population.k}; DFM factors: {reference._score.k}")
    event, model, guidance = initialize_h(reference, population, tail, widths, initialization_seed,
                                         log_remaining_time=log_remaining_time)
    describe_h_model(model, checkpoint, dfm_config, case, seed)

    steps = dfm_config["steps"]
    device = next(model.network.parameters()).device
    started = time.perf_counter()
    frozen_before = {name: value.detach().clone() for name, value in reference.state_dict().items()}
    _, sampler_options = split_h_options(h_training_options)
    soft_mode = (h_training_options or {}).get('soft_target_mode', 'off')
    soft_window = (h_training_options or {}).get('soft_target_window')
    split_counts = (("train", n_train), ("validation", n_validation), ("test", n_test))
    data = sample_h_training_data(
        model, split_counts, steps=steps, seeds=train_seeds, path_batch_size=path_batch_size,
        device=device, soft_target_mode=soft_mode, soft_target_window=soft_window, **sampler_options,
    )
    construction = None if soft_mode == 'off' else build_target_construction(
        model, mode=soft_mode, window=soft_window, steps=steps, split_counts=split_counts,
        stream_seeds=train_seeds, path_batch_size=path_batch_size, dfm_checkpoint=checkpoint_path,
        **sampler_options)
    selection, history, training_probability, test_metrics = fit_and_evaluate_h(
        model, reference, data, frozen_before, epochs=epochs, batch_size=batch_size,
        lr=lr, seeds=train_seeds, device=device, training_options=h_training_options,
        target_construction=construction,
    )
    curve, training_config, h_output = save_h_training(
        output_root, model, population, data, selection, history, training_probability, test_metrics,
        case=case, seed=seed, tail=tail, n_train=n_train, n_validation=n_validation,
        n_test=n_test, epochs=epochs, lr=lr, batch_size=batch_size, path_batch_size=path_batch_size,
        steps=steps, stream_seeds=train_seeds, initialization_seed=initialization_seed,
        reference_checkpoint=checkpoint_path, started=started, target_construction=construction,
    )
    print(f"Selected H epoch: {selection['best_epoch']}")
    display(pd.Series({k: v for k, v in test_metrics.items() if k != 'terminal_diagnostics'}, name="Held-out test loss"))
    display(pd.DataFrame(selection['validation_diagnostics']['late_rows']))
    print('Held-out last-step diagnostic:', {k: v for k, v in test_metrics['terminal_diagnostics'].items()
                                            if k not in ('late_rows', 'calibration')})
    print("constant_loss predicts the training event frequency at every time and state.")
    print("Low probability loss alone does not establish accurate guidance gradients.")
    print("Saved H checkpoint and diagnostics:", h_output)

    sampling_model, sampling_metadata, sampling_guidance, population = load_trained_h(h_output, config.device)
    if validation_sampling_n:
        # Fixed, named validation streams are distinct from all final reporting
        # streams. These metrics never cause automatic retuning on the test draw.
        validation_seeds = (52001, 52002)
        existing = [*train_seeds.values(), guided_seed, truth_seed]
        if assessment_settings is not None:
            existing.extend(getattr(assessment_settings, name) for name in
                ('rejection_seed', 'rejection_reference_seed', 'truth_reference_seed', 'projection_seed', 'matching_seed'))
        if {s % 2**32 for s in validation_seeds} & {s % 2**32 for s in existing}:
            raise ValueError('Validation sampling seeds collide with training/final reporting streams')
        validation_guided = sample_h_guided(sampling_model, sampling_guidance,
            n_samples=validation_sampling_n, batch_size=sample_batch_size, steps=steps, seed=52001)
        validation_rejection, validation_rejection_info = sample_dfm_rejection(
            sampling_model.reference, population.w, sampling_model.C, validation_sampling_n,
            steps=steps, batch_size=sample_batch_size, seed=52002,
            max_proposals=max(100*validation_sampling_n, sample_batch_size))
        from scipy.stats import ks_2samp, wasserstein_distance
        vp = portfolio_values(validation_guided, population.w)
        rp = portfolio_values(validation_rejection, population.w)
        validation_metrics = dict(guided_seed=52001, rejection_seed=52002,
            n=validation_sampling_n, event_rate=float((vp <= sampling_model.C).mean()),
            rejection=validation_rejection_info)
        if len(rp) and not validation_rejection_info['status'].startswith('nonfinite'):
            for name, values in (('raw', vp), ('filtered', vp[vp <= sampling_model.C])):
                validation_metrics[name] = dict(n=len(values),
                    portfolio_ks=float(ks_2samp(values, rp).statistic) if len(values) else None,
                    portfolio_w1=float(wasserstein_distance(values, rp)) if len(values) else None)
        write_json(h_output / 'validation_sampling.json', validation_metrics)
        np.savez_compressed(h_output / 'validation_samples.npz', guided=validation_guided,
                            rejection=validation_rejection)
        print('Independent validation sampling:', validation_metrics, flush=True)
    count = config.n_eval if n_samples is None else n_samples
    steps = sampling_metadata["steps"]
    threshold = sampling_model.C
    guided_sampling_diagnostics = {}
    guided_samples = sample_h_guided(
        sampling_model, sampling_guidance, n_samples=count, batch_size=sample_batch_size,
        steps=steps, seed=guided_seed,
        initialization_max_proposals=initialization_max_proposals,
        diagnostics=guided_sampling_diagnostics,
    )
    print(f"Generating {count:,} true conditional samples...", flush=True)
    true_conditional_samples = population.sample(count, np.random.default_rng(truth_seed), c=threshold)
    guided_portfolio, truth_portfolio, event_mask, filtered_samples, filtered_portfolio = filter_conditional_samples(
        guided_samples, true_conditional_samples, population, threshold, count, sampling_model.reference.d,
    )
    summary = conditional_sample_summary(guided_portfolio, filtered_portfolio, truth_portfolio, threshold, count)
    if assessment_settings is None:
        assessment_settings = ConditionalAssessmentConfig(
            rejection_seed=42003, rejection_reference_seed=42004, truth_reference_seed=42005,
            projection_seed=42006, matching_seed=42007,
        )
    assessment = assess_conditional_samples(
        sampling_model.reference, population, threshold, guided_samples, true_conditional_samples,
        filtered_samples, steps=steps, settings=assessment_settings,
        existing_seeds=(guided_seed, truth_seed, initialization_seed, *train_seeds.values()),
    )
    summary = extend_conditional_summary(summary, assessment['samples'], population.w, threshold,
                                         assessment['rejection_diagnostics'])
    display(summary)
    display(assessment['comparison_metrics'])
    print(f"Condition: equal-weight portfolio return <= {threshold:.6f}")
    print(f"Event-filtered guided samples: {len(filtered_samples):,}/{count:,} retained.")
    if len(filtered_samples) == 0:
        print("No guided samples met the event; filtered statistics are unavailable and plots label the empty subset.")
    sample_output = save_conditional_samples(
        h_output, sampling_metadata, population, threshold, guided_samples, true_conditional_samples,
        filtered_samples, event_mask, summary, n_samples=count, steps=steps,
        batch_size=sample_batch_size, guided_seed=guided_seed, truth_seed=truth_seed,
        assessment=assessment, guided_sampling_diagnostics=guided_sampling_diagnostics,
    )
    results = {
        "h_model": sampling_model,
        "guided_samples": guided_samples,
        "guided_sampling_diagnostics": guided_sampling_diagnostics,
        "filtered_guided_samples": filtered_samples,
        "guided_event_mask": event_mask,
        "true_conditional_samples": true_conditional_samples,
        "rejection_samples": assessment['samples']['rejection'],
        "independent_rejection_samples": assessment['samples']['rejection_reference'],
        "independent_true_conditional_samples": assessment['samples']['truth_reference'],
        "rejection_diagnostics": assessment['rejection_diagnostics'],
        "truth_reference_diagnostics": assessment['truth_reference_diagnostics'],
        "comparison_metrics": assessment['comparison_metrics'],
        "projection_directions": assessment['projection_directions'],
        "assessment_config": assessment['assessment_config'],
        "population": population,
        "threshold": threshold,
        "validation_curve": curve[["epoch", "validation"]],
        "test_metrics": test_metrics,
        "sample_summary": summary,
        "h_output": h_output,
        "sample_output": sample_output,
    }
    # The Refined CDG-ML figures are always saved. With the CDG-ML
    # comparison, both validation curves and the single performance figure are
    # displayed together afterwards, independent of the matplotlib backend.
    plot_h_validation(curve, selection['best_epoch'], h_output, show=False)
    if not compare_raw_h:
        display(Image(filename=str(h_output / 'validation_loss.png')))
    plot_conditional_ecdf(
        assessment['samples'], population, threshold, sampling_metadata['case'],
        assessment['comparison_metrics'], sample_output, assessment['rejection_diagnostics'],
        show=not compare_raw_h,
    )
    plot_conditional_comparison(
        guided_samples, true_conditional_samples, filtered_samples, population, threshold,
        sampling_metadata['case'], sample_output, show=not compare_raw_h,
    )
    if compare_raw_h:
        sampling_config = dict(
            steps=steps, batch_size=sample_batch_size, guided_seed=guided_seed,
            truth_seed=truth_seed, guided_sampling=guided_sampling_diagnostics,
        )
        _compare_raw_h(results, training_config, sampling_config, data=data, device=config.device)
    return results
