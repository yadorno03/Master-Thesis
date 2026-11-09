"""Frozen-DFM conditioning: H training, guided sampling and the five-source benchmark.

The benchmark compares five sources of conditional returns for the event
w'R <= C (Chen et al., 2025, for the DFM; Guo et al., 2026, for H and its
Doob guidance):

* conditional ground truth: the exact truncated Gaussian population law;
* DFM rejection sampling: the frozen numerical DFM conditioned by rejection,
  the law that guidance targets;
* Refined CDG-ML + exact final step: H with the five refinements, guided Euler
  steps and the exact conditioned last step (guided_sampling.exact_final_step);
* CDG-ML + exact final step: the same sampler with H without refinements
  (Guo et al., 2026), which isolates the effect of the refinements;
* No guidance + exact final step: unguided DFM paths whose last step only is
  drawn exactly, which isolates the effect of the exact final step.

Both H models are trained on the same unguided trajectories of the frozen DFM
with the same recipe (40k trajectories, 50 epochs).
"""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import logging
import shutil
from numbers import Integral, Real
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from IPython.display import display, Image, Markdown

from DFM import UnconditionalDFM
from reference import FrozenDFMReference
from conditioning import ConditioningConfig
from h_model import HNetwork, train_h, evaluate_h, diagnose_h_terminal
from guided_sampling import DFMHGuidance, exact_final_step, sample_conditional_initial
from sampling import generate_samples
from synthetic_conditional_truth import (
    population_from_checkpoint, population_metadata, check_population_model, check_population_h,
)
from .configuration import Config, write_json, ConditionalAssessmentConfig, validate_h_seeds
from .reporting import (
    DFM_LABELS, plot_h_validation, portfolio_values, conditional_comparisons, conditional_moment_table,
    conditional_summary, plot_conditional_distributions, conditional_tables, shown_sources, table_notes,
    format_return, SOURCES, EVENT_TOLERANCE, PORTFOLIO_CONVENTION, MATCHING_RULE,
)
from .unconditional import dfm_folder_name

CDG_ML_OPTIONS = dict(normalize_inputs=False, loss='brier', late_time_fraction=0., last_step_fraction=0.,
                      use_last_step_targets=False, times_per_trajectory=None)
REFINED_OPTIONS = dict(normalize_inputs=True, loss='bce', late_time_fraction=0.4, last_step_fraction=0.1,
                       use_last_step_targets=True, times_per_trajectory=32)
# key: (label, training options, log remaining-time input, source name, final sampling step)
H_VARIANTS = {
    'refined': ('Refined CDG-ML', REFINED_OPTIONS, True, 'refined_exact', 'exact'),
    'cdg_ml': ('CDG-ML', CDG_ML_OPTIONS, False, 'cdg_ml_exact', 'exact'),
}
DEFAULT_TRAIN_SEEDS = {'train': 32001, 'validation': 32002, 'test': 32003, 'shuffle': 32004}
# Progress messages; the notebook shows results only (enable with logging.basicConfig(level=logging.INFO)).
logger = logging.getLogger(__name__)


def sample_dfm_rejection(reference, weights, threshold, n_accepted, *, steps,
                         batch_size, seed, max_proposals):
    """Condition the supplied frozen numerical sampler by batched rejection.

    Proposals use generate_samples defaults: reference T/t0, parameter dtype and
    device, N(0,I) initialization, no guidance or trajectories. They are mapped
    to returns (reference.standardization.to_returns) before the event w'r <= C
    is checked, so weights and threshold are in return units. Acceptance uses
    CPU NumPy float64 portfolio products and an exact <=. Accepted rows remain in
    generation order. All final-batch hits count in the observed fraction, even
    when only a prefix is retained, so the fraction is descriptive under the
    stopping rule rather than an unbiased estimate.

    Nonfinite outputs stop the run: earlier finite batches are kept for
    diagnosis, but the assessment excludes this reference, since dropping
    nonfinite proposals would change the rejection law. A sampler exception from
    nonfinite integration reports the attempted batch separately, because none of
    its terminal proposals can be counted.
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
        batch = reference.standardization.to_returns(batch)
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


def load_frozen_dfm(checkpoint_path, case, device):
    """Restore the saved architecture and population, then freeze the selected DFM."""
    h_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    h_dfm_config = h_checkpoint["config"]
    h_population = population_from_checkpoint(h_dfm_config["d"], h_dfm_config["k"], case, h_checkpoint)
    h_base_model = UnconditionalDFM(
        h_dfm_config["d"], h_dfm_config["k"], sigma_max=2.0,
        T=h_dfm_config["T"], t0=h_dfm_config["t0"],
        hidden_dims=[h_dfm_config["width"], h_dfm_config["width"]],
        score_network=h_dfm_config.get("dfm_network", "mlp"),
    ).to(device)
    state = dict(h_checkpoint["state"])
    # DFMs saved before return standardization model raw returns: identity map.
    for name, value in h_base_model.standardization.state_dict().items():
        state.setdefault(f"standardization.{name}", value)
    h_base_model.load_state_dict(state)
    h_reference = FrozenDFMReference(h_base_model)

    check_population_model(h_population, h_reference)
    return h_checkpoint, h_dfm_config, h_population, h_base_model, h_reference


def reuse_frozen_unconditional_run(source, output, config, networks=('mlp',)):
    """Copy reviewed DFM runs into a new H experiment without training the DFMs."""
    source, output = Path(source).resolve(), Path(output)
    checkpoints = {}
    for seed in config.seeds:
        for network in networks:
            name = dfm_folder_name('gaussian', seed, network)
            checkpoint, saved_config, _, _, _ = load_frozen_dfm(source/name/'dfm.pt', 'gaussian', config.device)
            expected = asdict(replace(config, dfm_network=network))
            for key in ('d', 'k', 'T', 't0', 'steps', 'width', 'dfm_initialization', 'dfm_time_sampling',
                        'dfm_standardization', 'dfm_loading_lr_scale', 'dfm_loading_warmup_epochs', 'lr',
                        'dfm_epochs', 'n_train', 'n_val', 'dfm_network'):
                if saved_config.get(key, Config.__dataclass_fields__[key].default) != expected[key]:
                    raise ValueError(f'Frozen checkpoint configuration mismatch for {key}')
            shutil.copytree(source/name, output/name)
            checkpoints[name] = hashlib.sha256((source/name/'dfm.pt').read_bytes()).hexdigest()
    metrics = pd.read_csv(source/'metrics.csv')
    metrics = metrics.loc[(metrics.case == 'gaussian') & metrics.seed.isin(config.seeds)]
    metrics.to_csv(output/'metrics.csv', index=False)
    write_json(output/'frozen_reference.json', dict(source=str(source), dfm_retrained=False, checkpoints=checkpoints))
    logger.info('Reusing frozen DFMs and their saved unconditional assessment: %s', source)
    return metrics, metrics.select_dtypes(include=np.number)


def initialize_h(reference, population, tail, widths, initialization_seed, *, log_remaining_time=False):
    """Initialize H inside fork_rng, preserving the RNG state after DFM loading.

    H uses the population's portfolio event mapped to the DFM's model coordinates.
    """
    w, C = reference.standardization.portfolio_event(population.w, population.threshold(tail))
    h_event = ConditioningConfig(w=w, C=C)
    # Preserve the notebook's global RNG state while initializing H reproducibly.
    with torch.random.fork_rng():
        torch.manual_seed(initialization_seed)
        h_model = HNetwork(reference, h_event, hidden_dims=widths, log_remaining_time=log_remaining_time)
    check_population_h(population, h_model)
    h_guidance = DFMHGuidance(h_model)
    return h_event, h_model, h_guidance


def h_rng(name, seeds, device):
    """Construct a fresh named H generator; never share mutable stream state."""
    return torch.Generator(device=device).manual_seed(seeds[name])


def sample_h_training_data(h_model, split_counts, *, steps, seeds, path_batch_size, device, diagnostics=None):
    """Draw independent unguided trajectory splits and check both label classes."""
    split_counts = tuple(split_counts)
    base_seeds = validate_h_seeds(seeds, names=tuple(split for split, _ in split_counts))
    h_data = {}
    for split, count in split_counts:
        logger.info("Generating %s %s trajectories", f"{count:,}", split)
        started = time.perf_counter()
        h_data[split] = h_model.sample_trajectories(count, num_steps=steps, generator=h_rng(split, base_seeds, device),
                                                    batch_size=path_batch_size)
        if diagnostics is not None:
            diagnostics[split] = dict(collection_seconds=time.perf_counter() - started)
        # Use the represented time intervals to avoid float32 rounding mismatches.
        time_edges = torch.cat((h_data[split].times, h_data[split].times.new_tensor([h_model.horizon])))
        h_data[split].weights = time_edges.diff() / h_model.horizon
        positives = int(h_data[split].labels.sum().item())
        logger.info("Terminal event: %s/%s (%.2f%%)", f"{positives:,}", f"{count:,}", 100 * positives / count)
        if positives == 0 or positives == count:
            raise RuntimeError(f"{split} has only one label class; increase its trajectory count before training H.")
    return h_data


def fit_and_evaluate_h(h_model, h_reference, h_data, frozen_before, *, epochs, batch_size, lr, seeds, device,
                       training_options=None):
    """Select H by validation loss, evaluate held-out loss, and verify the frozen DFM."""
    options = dict(training_options or {})
    h_selection = {}
    logger.info("Training H for %d epochs; selecting by validation loss", epochs)
    h_train_history = train_h(
        h_model, h_data["train"], epochs=epochs, batch_size=batch_size, lr=lr,
        generator=h_rng("shuffle", seeds, device), validation_data=h_data["validation"],
        selection=h_selection, **options,
    )
    h_model.eval()
    h_training_probability = float(h_data["train"].labels.mean())
    h_test_metrics = evaluate_h(h_model, h_data["test"], constant_probability=h_training_probability,
                                batch_size=batch_size)
    h_selection['training_options'] = options
    diagnostic_options = {key: options.get(key, 0.) for key in ('late_time_fraction', 'last_step_fraction')}
    h_selection['validation_diagnostics'] = diagnose_h_terminal(h_model, h_data['validation'], **diagnostic_options)
    h_test_metrics['terminal_diagnostics'] = diagnose_h_terminal(h_model, h_data['test'], **diagnostic_options)
    for name, value in h_reference.state_dict().items():
        torch.testing.assert_close(value, frozen_before[name], rtol=0, atol=0)
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in h_reference.parameters())
    return h_selection, h_train_history, h_training_probability, h_test_metrics


def save_h_training(
    output_root, h_model, h_population, h_data, h_selection,
    h_train_history, h_training_probability, h_test_metrics, *,
    case, seed, tail, n_train, n_validation, n_test, epochs, lr, batch_size,
    path_batch_size, steps, stream_seeds, initialization_seed, reference_checkpoint, started,
):
    """Save the selected checkpoint, trajectory settings, curves, and held-out metrics."""
    validation = list(h_selection["validation_history"])
    step_curve = None
    if len(validation) != len(h_train_history) and "validation_steps" in h_selection:
        # Step-interval validation: report the latest validation within each epoch.
        steps_per_epoch = h_selection["total_steps"] / len(h_train_history)
        validated_at = np.asarray(h_selection["validation_steps"])
        latest = np.searchsorted(validated_at, steps_per_epoch * np.arange(1, len(h_train_history) + 1),
                                 side="right") - 1
        step_curve = pd.DataFrame({"step": validated_at, "validation": validation})
        validation = [validation[i] if i >= 0 else np.nan for i in latest]
    h_curve = pd.DataFrame({"epoch": np.arange(1, len(h_train_history) + 1), "train": h_train_history,
                            "validation": validation})
    h_training_config = dict(
        **population_metadata(h_population, case),
        # The return threshold, and the threshold H uses in the DFM's model coordinates.
        case=case, seed=seed, tail=tail, threshold=float(h_population.threshold(tail)), model_threshold=h_model.C,
        n_train=n_train, n_validation=n_validation, n_test=n_test,
        epochs=epochs, lr=lr, batch_size=batch_size, path_batch_size=path_batch_size,
        steps=steps, stream_seeds=stream_seeds, initialization_seed=initialization_seed,
        selected_epoch=h_selection["best_epoch"], selected_step=h_selection.get("best_step"),
        total_steps=h_selection.get("total_steps"),
        reference_checkpoint=str(reference_checkpoint),
        training_options=h_selection.get('training_options', {}),
        log_remaining_time=h_model.log_remaining_time,
    )
    output = output_root / f"h_{case}_seed{seed}_tail{tail}" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    h_model.save_checkpoint(output / "h.pt", metadata=h_training_config)
    h_curve.to_csv(output / "h_curve.csv", index=False)
    if step_curve is not None:
        step_curve.to_csv(output / "h_validation_steps.csv", index=False)
    write_json(output / "training.json", h_training_config)
    write_json(output / 'validation_diagnostics.json', h_selection['validation_diagnostics'])
    write_json(output / "test_metrics.json", dict(
        **h_test_metrics, training_event_rate=h_training_probability,
        test_event_rate=float(h_data["test"].labels.mean()), seconds=time.perf_counter() - started,
    ))
    return h_curve, h_training_config, output


def load_trained_h(output, device):
    """Reload the saved selected H, validating population provenance and its portfolio event."""
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
    # H must condition on the population's event, in the model coordinates of its DFM.
    w, C = h_sampling_model.reference.standardization.portfolio_event(
        conditional_population.w, conditional_population.threshold(h_sampling_metadata["tail"]))
    np.testing.assert_allclose(np.asarray(h_sampling_model.config.w), w, rtol=0, atol=1e-12)
    np.testing.assert_allclose(h_sampling_model.C, C, rtol=0, atol=1e-12)
    return h_sampling_model, h_sampling_metadata, h_sampling_guidance, conditional_population


def sample_h_guided(
    h_sampling_model, h_sampling_guidance, *,
    n_samples, batch_size, steps, seed, guidance_scale=1.0,
    initialization_max_proposals=None, diagnostics=None, final_step='euler',
):
    """Start from the learned H tilt, then retain every guided terminal output.

    The initialization proposal cap applies once to the entire requested sample,
    not separately to each integration batch. Exhaustion raises without falling
    back to Gaussian starts. The optional diagnostics mapping records the actual
    initialization settings and proposal counts without changing the array return.
    final_step='exact' draws the last step from the exact finite-grid conditional
    transition (guided_sampling.exact_final_step); with the same seed, starts and
    all earlier increments equal those of the default guided Euler run.
    Sampling runs in the DFM's model coordinates; the returned rows are returns.
    """
    if final_step not in ('euler', 'exact'):
        raise ValueError("final_step must be 'euler' or 'exact'")
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

    logger.info("Generating %s H-guided DFM samples", f"{n_samples:,}")
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
            final_step=(None if final_step == 'euler' else
                        lambda x, g: exact_final_step(h_sampling_model, x, steps, g)),
        )
        guided_chunks.append(h_sampling_model.reference.standardization.to_returns(batch))
    guided_samples = np.concatenate(guided_chunks, axis=0)
    if diagnostics is not None:
        diagnostics.update(
            initialization=dict(initialization, max_proposals=int(initialization_max_proposals),
                                batch_size=int(batch_size)),
            seed=int(seed), steps=int(steps), batch_size=int(batch_size),
            guidance_scale=float(guidance_scale), terminal_filtering=False, final_step=final_step,
        )
    return guided_samples


def sample_unguided_exact(h_model, *, n_samples, batch_size, steps, seed, diagnostics=None):
    """Unguided DFM paths from N(0, I) whose last step only is drawn from the exact conditioned transition.

    No H enters: h_model only supplies the frozen DFM, the time grid and the
    event. Comparing this source with the guided ones shows what guidance
    contributes before the last step. The returned rows are returns.
    """
    for name, value in (('n_samples', n_samples), ('batch_size', batch_size), ('steps', steps)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**63:
        raise ValueError('seed must be an integer in [0, 2**63)')
    reference = h_model.reference
    generator = torch.Generator(device=next(reference.parameters()).device).manual_seed(int(seed))
    logger.info("Generating %s unguided DFM samples with the exact final step", f"{n_samples:,}")
    chunks = []
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            batch = generate_samples(reference, min(batch_size, n_samples - start), num_steps=steps,
                                     generator=generator,
                                     final_step=lambda x, g: exact_final_step(h_model, x, steps, g))
            chunks.append(reference.standardization.to_returns(batch))
    if diagnostics is not None:
        diagnostics.update(seed=int(seed), steps=int(steps), batch_size=int(batch_size), initialization='N(0,I)',
                           guidance='none', terminal_filtering=False, final_step='exact')
    return np.concatenate(chunks, axis=0)


def h_diagnostic_table(h_results):
    """Held-out loss and last-step diagnostics of both H models on the same test paths."""
    rows = {}
    for key, result in h_results.items():
        metrics, diagnostics = result['test_metrics'], result['test_metrics']['terminal_diagnostics']
        rows[result['label']] = {
            'Held-out Brier loss': metrics['h_loss'], 'Constant-probability Brier loss': metrics['constant_loss'],
            'First-layer Tanh saturation': diagnostics['first_layer_saturation_fraction'],
            'Last-step probability RMSE': diagnostics['last_probability_rmse'],
            'Last-step gradient relative RMSE': diagnostics['gradient_relative_rmse'],
            'Last-step gradient cosine': diagnostics['gradient_mean_cosine'],
            'Selected epoch': result['selection']['best_epoch'],
            'Selected optimizer step': result['selection'].get('best_step'),
        }
    return pd.DataFrame(rows)


H_COUNT_ROWS = ('Selected epoch', 'Selected optimizer step')


def readable_h_diagnostics(table):
    """Display version of h_diagnostic_table: counts as integers, other values with four decimals."""
    def text(row, value):
        if pd.isna(value):
            return '—'
        return f'{int(value):,}' if row in H_COUNT_ROWS else f'{value:.4f}'
    return pd.DataFrame({column: [text(row, value) for row, value in table[column].items()]
                         for column in table.columns}, index=table.index)


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_conditional_benchmark(
    config, output_root, *, tail=0.05, n_train=40_000, n_validation=2_000, n_test=2_000, epochs=50,
    validation_interval_steps=157, widths=(64, 64), initialization_seed=2027, lr=1e-3, batch_size=64,
    path_batch_size=256, train_seeds=None, n_samples=None, sample_batch_size=256, guided_seed=42001,
    truth_seed=42002, unguided_seed=42008, initialization_max_proposals=None, assessment_settings=None,
    show=True, plot_sources=None,
):
    """Train CDG-ML and Refined CDG-ML on shared trajectories, then compare the five sources.

    The frozen DFM is the one trained with config.dfm_network. Both H models share
    it, the trajectory splits, the initialization seed and the guided seed, and
    both are sampled with the exact final step, so they differ only by H. The
    no-guidance source draws the same exact last step after unguided paths. Every
    sampled row is kept (no event filtering). The DFM rejection streams and the
    independent truth draw use seeds distinct from every training and sampling stream.

    ``plot_sources`` selects the sources drawn in the figure and listed in the two
    readable tables, beside the exact law; the saved samples and CSV files keep
    all five. The tables' within-noise verdict is for the first shown H model.
    """
    train_seeds = validate_h_seeds(train_seeds or DEFAULT_TRAIN_SEEDS)
    plot_sources = SOURCES if plot_sources is None else tuple(plot_sources)
    shown = shown_sources(plot_sources)
    if 'rejection' not in shown or not any(s in shown for s in SOURCES[2:]):
        raise ValueError('Plot sources require DFM rejection and at least one sampled model')
    verdict = next(s for s in shown if s in SOURCES[2:])
    settings = assessment_settings or ConditionalAssessmentConfig(
        rejection_seed=42003, rejection_reference_seed=42004, truth_reference_seed=42005,
        projection_seed=42006, matching_seed=42007, noise_seed=42009)
    assessment_seeds = (settings.rejection_seed, settings.rejection_reference_seed, settings.truth_reference_seed,
                        settings.projection_seed, settings.matching_seed, settings.noise_seed)
    existing = (guided_seed, truth_seed, unguided_seed, initialization_seed, *train_seeds.values())
    # CPU Torch streams can coincide for seeds differing by a multiple of 2**32.
    if {int(s) % 2**32 for s in assessment_seeds} & {int(s) % 2**32 for s in existing}:
        raise ValueError('Assessment seeds must differ from existing sampling and training seeds')
    output_root = Path(output_root)
    case, seed = 'gaussian', config.seeds[0]
    checkpoint_path = output_root / dfm_folder_name(case, seed, config.dfm_network) / 'dfm.pt'
    checkpoint, dfm_config, population, _, reference = load_frozen_dfm(checkpoint_path, case, config.device)
    steps = int(dfm_config['steps'])
    run_dir = output_root / f'conditional_tail{tail}'
    run_dir.mkdir(parents=True, exist_ok=False)
    logger.info("Assets: %d; factors: %d; event probability %.0f%%; frozen %s, epoch %s", population.d,
                population.k, 100 * tail, DFM_LABELS[config.dfm_network], checkpoint['selected_epoch'])
    timings, h_results, models = {}, {}, {}
    for key, (label, _, log_remaining_time, _, _) in H_VARIANTS.items():
        _, models[key], _ = initialize_h(reference, population, tail, widths, initialization_seed,
                                         log_remaining_time=log_remaining_time)
    # The return threshold; H holds the same event in the DFM's model coordinates.
    threshold = float(population.threshold(tail))
    logger.info("Event: equal-weight portfolio return <= %.6f", threshold)
    device = next(models['refined'].network.parameters()).device
    started = time.perf_counter()
    splits = (('train', n_train), ('validation', n_validation), ('test', n_test))
    # Features G_t x do not depend on the network, so both variants share these paths.
    data = sample_h_training_data(models['refined'], splits, steps=steps, seeds=train_seeds,
                                  path_batch_size=path_batch_size, device=device)
    timings['trajectory_collection_seconds'] = time.perf_counter() - started
    frozen_before = {name: value.detach().clone() for name, value in reference.state_dict().items()}
    for key, (label, options, _, _, _) in H_VARIANTS.items():
        logger.info("%s: training on the shared trajectories", label)
        variant_started = time.perf_counter()
        selection, history, probability, test_metrics = fit_and_evaluate_h(
            models[key], reference, data, frozen_before, epochs=epochs, batch_size=batch_size, lr=lr,
            seeds=train_seeds, device=device,
            training_options=dict(options, validation_interval_steps=validation_interval_steps))
        timings[f'{key}_training_seconds'] = time.perf_counter() - variant_started
        curve, _, h_output = save_h_training(
            run_dir / key, models[key], population, data, selection, history, probability, test_metrics,
            case=case, seed=seed, tail=tail, n_train=n_train, n_validation=n_validation, n_test=n_test,
            epochs=epochs, lr=lr, batch_size=batch_size, path_batch_size=path_batch_size, steps=steps,
            stream_seeds=train_seeds, initialization_seed=initialization_seed,
            reference_checkpoint=checkpoint_path, started=variant_started)
        if H_VARIANTS[key][3] in plot_sources:
            plot_h_validation(curve, selection['best_epoch'], h_output, title=f'{label}: validation loss', show=False)
        model, _, guidance, population = load_trained_h(h_output, config.device)
        h_results[key] = dict(label=label, model=model, guidance=guidance, h_output=h_output, curve=curve,
                              selection=selection, test_metrics=test_metrics)
    del data

    count = config.n_eval if n_samples is None else int(n_samples)
    if initialization_max_proposals is None:
        initialization_max_proposals = max(int(np.ceil(20 * count / tail)), sample_batch_size)
    samples, diagnostics = {}, {}
    for key, (label, _, _, source, final_step) in H_VARIANTS.items():
        logger.info("%s: guided sampling (%s final step)", label, final_step)
        sampling_started = time.perf_counter()
        diagnostics[source] = {}
        samples[source] = sample_h_guided(
            h_results[key]['model'], h_results[key]['guidance'], n_samples=count, batch_size=sample_batch_size,
            steps=steps, seed=guided_seed, initialization_max_proposals=initialization_max_proposals,
            diagnostics=diagnostics[source], final_step=final_step)
        timings[f'{source}_sampling_seconds'] = time.perf_counter() - sampling_started
    sampling_started = time.perf_counter()
    diagnostics['unguided_exact'] = {}
    samples['unguided_exact'] = sample_unguided_exact(
        h_results['refined']['model'], n_samples=count, batch_size=sample_batch_size, steps=steps,
        seed=unguided_seed, diagnostics=diagnostics['unguided_exact'])
    timings['unguided_exact_sampling_seconds'] = time.perf_counter() - sampling_started
    logger.info("Generating %s + %s conditional ground-truth samples", f"{count:,}", f"{count:,}")
    samples['truth'] = population.sample(count, np.random.default_rng(truth_seed), c=threshold)
    samples['truth_reference'] = population.sample(count, np.random.default_rng(settings.truth_reference_seed),
                                                   c=threshold)
    for name, stream_seed, cap in (('rejection', settings.rejection_seed, settings.rejection_max_proposals),
                                   ('rejection_reference', settings.rejection_reference_seed,
                                    settings.rejection_reference_max_proposals)):
        logger.info("DFM rejection stream '%s' (cap %s proposals)", name, f"{cap:,}")
        samples[name], diagnostics[name] = sample_dfm_rejection(
            reference, population.w, threshold, settings.rejection_n, steps=steps,
            batch_size=settings.rejection_batch_size, seed=stream_seed, max_proposals=cap)
        info = diagnostics[name]
        logger.info("%s: %s/%s rows from %s proposals in %.0fs", info['status'], f"{info['retained_count']:,}",
                    f"{info['requested_count']:,}", f"{info['total_proposals']:,}", info['sampling_seconds'])
        if info['status'] not in ('complete', 'partial'):
            raise RuntimeError(f"DFM rejection stream {name} failed: {info['status']}")
        timings[f'{name}_seconds'] = info['sampling_seconds']

    comparisons, directions = conditional_comparisons(
        samples, population.w, projection_seed=settings.projection_seed, matching_seed=settings.matching_seed,
        projections=settings.projections)
    moments = conditional_moment_table({name: samples[name] for name in
                                        (*SOURCES, 'truth_reference', 'rejection_reference')}, population, threshold)
    # Three standard deviations of the last Euler step along the return portfolio: sqrt(dt) * |w|,
    # with w in model coordinates (scale * w), where the step adds sqrt(dt) noise to every coordinate.
    near_band = float(3 * np.sqrt(h_results['refined']['model'].horizon / steps)
                      * np.linalg.norm(h_results['refined']['model'].config.w))
    summary = conditional_summary(samples, comparisons, moments, threshold, population.w, near_band=near_band)
    h_table = h_diagnostic_table(h_results)
    values, distances = conditional_tables(
        samples, population, threshold, sources=shown, verdict=verdict, directions=directions,
        matching_seed=settings.matching_seed, noise_seed=settings.noise_seed, replicates=settings.noise_replicates,
        event_frequency=diagnostics['rejection']['observed_acceptance_fraction'])
    np.savez_compressed(run_dir / 'conditional_samples.npz', **samples, weights=population.w, threshold=threshold,
                        projection_directions=directions)
    comparisons.to_csv(run_dir / 'comparisons.csv', index=False)
    moments.to_csv(run_dir / 'moments.csv', index=False)
    summary.to_csv(run_dir / 'summary.csv')
    h_table.to_csv(run_dir / 'h_diagnostics.csv')
    values.to_csv(run_dir / 'table_values.csv')
    distances.to_csv(run_dir / 'table_distances.csv')
    figure = plot_conditional_distributions(
        samples, population, threshold, run_dir, sources=shown, show=False,
        title=f'{tail:.0%} event: equal-weight portfolio return below '
              f'C = {format_return(threshold)}; '
              f'{DFM_LABELS[config.dfm_network]}, d={population.d}, k={population.k}')
    write_json(run_dir / 'run.json', _clean(dict(
        **population_metadata(population, case), dfm_network=config.dfm_network,
        dfm_standardization=dfm_config.get('dfm_standardization', 'none'), tail=tail, threshold=threshold,
        steps=steps, near_band=near_band, event_tolerance=EVENT_TOLERANCE, plot_sources=list(plot_sources),
        dfm_checkpoint=str(checkpoint_path),
        dfm_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        h_recipe=dict(n_train=n_train, n_validation=n_validation, n_test=n_test, epochs=epochs,
                      validation_interval_steps=validation_interval_steps, widths=list(widths), lr=lr,
                      batch_size=batch_size, initialization_seed=initialization_seed, train_seeds=train_seeds),
        h_outputs={key: str(result['h_output']) for key, result in h_results.items()},
        h_training_options={key: dict(H_VARIANTS[key][1]) for key in H_VARIANTS},
        n_samples=count, guided_seed=guided_seed, truth_seed=truth_seed, unguided_seed=unguided_seed,
        initialization_max_proposals=initialization_max_proposals, assessment_config=asdict(settings),
        sampling_diagnostics=diagnostics, timings=timings, torch_threads=torch.get_num_threads(),
        portfolio_convention=PORTFOLIO_CONVENTION, matching_rule=MATCHING_RULE)))
    if show:
        for key, result in h_results.items():
            if H_VARIANTS[key][3] in plot_sources:
                display(Image(filename=str(result['h_output'] / 'validation_loss.png')))
        display(readable_h_diagnostics(h_table))
        display(Image(filename=str(figure)))
        values_note, distances_note = table_notes(min(len(samples[s]) for s in shown), tail)
        display(Markdown(values_note), values, Markdown(distances_note), distances)
    logger.info('Conditional benchmark saved to: %s', run_dir)
    return dict(tail=tail, samples=samples, comparisons=comparisons, moments=moments, summary=summary,
                values=values, distances=distances, h_diagnostics=h_table, h_results=h_results,
                population=population, threshold=threshold, timings=timings, sampling_diagnostics=diagnostics,
                run_dir=run_dir, figure=figure)
