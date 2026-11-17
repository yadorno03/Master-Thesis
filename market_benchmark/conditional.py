"""Conditioning the frozen market DFM on a portfolio loss: H models, guided samplers and their comparison.

The event is w'R <= C for the equal-weight portfolio, with C the empirical
quantile of the training days. The H models, the samplers and the tables are
those of the synthetic benchmark (synthetic_benchmark.conditional): CDG-ML,
Refined CDG-ML and Refined CDG-ML + SMC. Only the references change: the
hold-out market days in the event replace the exact law, and DFM rejection,
the law guidance targets, measures the conditioning error.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import logging
import time

import numpy as np
import pandas as pd
import torch
from IPython.display import display, Image, Markdown

from guided_sampling import DFMHGuidance
from synthetic_benchmark.configuration import ConditionalAssessmentConfig, validate_h_seeds, write_json
from synthetic_benchmark.conditional import (
    DEFAULT_TRAIN_SEEDS, H_VARIANTS, SMC_VARIANTS, VARIANTS, _clean, build_h, fit_and_evaluate_h,
    h_diagnostic_table, h_training_curves, readable_h_diagnostics, sample_conditional_models, sample_dfm_rejection,
    sample_h_training_data,
)
from synthetic_benchmark.reporting import (
    DFM_LABELS, MATCHING_RULE, MODELS, PORTFOLIO_CONVENTION, conditional_comparisons, format_return,
    plot_h_validation, portfolio_values, readable_smc, smc_table,
)
from synthetic_benchmark.unconditional import dfm_folder_name
from .reporting import (
    leading_direction, market_conditional_tables, market_rejection_note, market_rejection_table,
    market_table_notes, noise_rows, plot_market_conditional,
)
from .unconditional import CASE, load_market_dfm

logger = logging.getLogger(__name__)
# The assessment seeds of both notebooks.
ASSESSMENT_SEEDS = dict(rejection_seed=42003, rejection_reference_seed=42004, truth_reference_seed=42005,
                        projection_seed=42006, matching_seed=42007, noise_seed=42009)


def event_threshold(split, tail):
    """The tail quantile of the equal-weight portfolio return over the training days."""
    if not 0 < tail < 1:
        raise ValueError('tail must be a probability in (0, 1)')
    return float(np.quantile(portfolio_values(split.train, split.weights), tail))


def conditional_folder(tail, label=None):
    """Folder name of a conditional run; ``label`` (the frozen network) keeps runs on several DFMs apart."""
    return f'conditional_{label}_tail{tail}' if label else f'conditional_tail{tail}'


def event_days(split, threshold):
    """Market days in the event: (hold-out days, training and validation days, share of hold-out days)."""
    hold_out = portfolio_values(split.evaluation, split.weights) <= threshold
    insample = portfolio_values(split.insample, split.weights) <= threshold
    return split.evaluation[hold_out], split.insample[insample], float(hold_out.mean())


def read_table(path):
    """A readable table saved as text, read back unchanged."""
    return pd.read_csv(path, index_col=0, dtype=str, keep_default_na=False)


def run_market_conditional(
    config, split, output_root, *, tail=0.05, plot_sources=None, n_train=40_000, n_validation=2_000,
    n_test=2_000, epochs=50, validation_interval_steps=157, widths=(64, 64), initialization_seed=2027, lr=1e-3,
    batch_size=64, path_batch_size=256, train_seeds=None, n_samples=None, sample_batch_size=256,
    guided_seed=42001, smc_seed=42011, smc_ess_threshold=.5, initialization_max_proposals=None,
    assessment_settings=None, label=None,
):
    """Train the H models on shared trajectories of the frozen market DFM, sample every model and compare.

    The frozen DFM is the one of config.dfm_network saved under ``output_root``
    (run_market_unconditional). The models and the recipe are those of
    synthetic_benchmark.conditional.run_conditional_benchmark; ``plot_sources``
    selects the models drawn in the figure (all by default). Every sampled row is
    kept. Results are saved under conditional_folder(tail, label).
    """
    train_seeds = validate_h_seeds(train_seeds or DEFAULT_TRAIN_SEEDS)
    plot_sources = MODELS if plot_sources is None else tuple(plot_sources)
    if not plot_sources or not set(plot_sources).issubset(MODELS):
        raise ValueError(f'plot_sources must be models among {MODELS}')
    settings = assessment_settings or ConditionalAssessmentConfig(**ASSESSMENT_SEEDS)
    used = (guided_seed, smc_seed, initialization_seed, *train_seeds.values(), settings.rejection_seed,
            settings.rejection_reference_seed, settings.projection_seed, settings.matching_seed, settings.noise_seed)
    # CPU Torch streams can coincide for seeds differing by a multiple of 2**32.
    if len({int(seed) % 2**32 for seed in used}) != len(used):
        raise ValueError('Training, sampling and assessment seeds must all differ')
    output_root = Path(output_root)
    seed = config.seeds[0]
    checkpoint_path = output_root / dfm_folder_name(CASE, seed, config.dfm_network) / 'dfm.pt'
    checkpoint, reference = load_market_dfm(checkpoint_path.parent, config.device)
    if checkpoint['panel_sha256'] != split.info['panel_sha256'] or reference.d != split.train.shape[1]:
        raise ValueError('The frozen DFM was trained on a different panel')
    steps = int(checkpoint['config']['steps'])
    weights, threshold = split.weights, event_threshold(split, tail)
    run_dir = output_root / conditional_folder(tail, label)
    run_dir.mkdir(parents=True, exist_ok=False)
    logger.info("Assets: %d; factors: %d; event probability %.0f%%; frozen %s, epoch %s", reference.d,
                checkpoint['config']['k'], 100 * tail, DFM_LABELS[config.dfm_network], checkpoint['selected_epoch'])
    logger.info("Event: equal-weight portfolio return <= %.6f", threshold)
    timings, h_results, models = {}, {}, {}
    for key in VARIANTS:
        _, models[key] = build_h(reference, weights, threshold, widths, initialization_seed,
                                 log_remaining_time=H_VARIANTS[key][2])
    device = next(models[VARIANTS[0]].network.parameters()).device
    started = time.perf_counter()
    # Features G_t x do not depend on the network, so both H models share these paths.
    data = sample_h_training_data(models[VARIANTS[0]], (('train', n_train), ('validation', n_validation),
                                                        ('test', n_test)), steps=steps, seeds=train_seeds,
                                  path_batch_size=path_batch_size, device=device)
    timings['trajectory_collection_seconds'] = time.perf_counter() - started
    frozen_before = {name: value.detach().clone() for name, value in reference.state_dict().items()}
    for key in VARIANTS:
        h_label, options = H_VARIANTS[key][:2]
        logger.info("%s: training on the shared trajectories", h_label)
        variant_started = time.perf_counter()
        selection, history, probability, test_metrics = fit_and_evaluate_h(
            models[key], reference, data, frozen_before, epochs=epochs, batch_size=batch_size, lr=lr,
            seeds=train_seeds, device=device,
            training_options=dict(options, validation_interval_steps=validation_interval_steps))
        timings[f'{key}_training_seconds'] = time.perf_counter() - variant_started
        h_output = run_dir / key
        h_output.mkdir()
        curve, step_curve = h_training_curves(selection, history)
        models[key].save_checkpoint(h_output / 'h.pt', metadata=dict(
            case=CASE, tail=tail, threshold=threshold, model_threshold=models[key].C,
            selected_epoch=selection['best_epoch'], reference_checkpoint=str(checkpoint_path)))
        curve.to_csv(h_output / 'h_curve.csv', index=False)
        if step_curve is not None:
            step_curve.to_csv(h_output / 'h_validation_steps.csv', index=False)
        write_json(h_output / 'test_metrics.json', _clean(dict(
            **test_metrics, training_event_rate=probability, test_event_rate=float(data['test'].labels.mean()),
            seconds=time.perf_counter() - variant_started)))
        plot_h_validation(curve, selection['best_epoch'], h_output, title=f'{h_label}: validation loss', show=False)
        h_results[key] = dict(label=h_label, model=models[key], guidance=DFMHGuidance(models[key]),
                              h_output=h_output, curve=curve, selection=selection, test_metrics=test_metrics)
    del data

    count = config.n_eval if n_samples is None else int(n_samples)
    if initialization_max_proposals is None:
        initialization_max_proposals = max(int(np.ceil(20 * count / tail)), sample_batch_size)
    samples, diagnostics, sampling_seconds = sample_conditional_models(
        h_results, count=count, steps=steps, sample_batch_size=sample_batch_size, guided_seed=guided_seed,
        smc_seed=smc_seed, smc_ess_threshold=smc_ess_threshold,
        initialization_max_proposals=initialization_max_proposals)
    timings.update({f'{source}_sampling_seconds': seconds for source, seconds in sampling_seconds.items()})
    for name, stream_seed, cap in (('rejection', settings.rejection_seed, settings.rejection_max_proposals),
                                   ('rejection_reference', settings.rejection_reference_seed,
                                    settings.rejection_reference_max_proposals)):
        logger.info("DFM rejection stream '%s' (cap %s proposals)", name, f"{cap:,}")
        samples[name], diagnostics[name] = sample_dfm_rejection(
            reference, weights, threshold, settings.rejection_n, steps=steps,
            batch_size=settings.rejection_batch_size, seed=stream_seed, max_proposals=cap)
        info = diagnostics[name]
        logger.info("%s: %s/%s rows from %s proposals in %.0fs", info['status'], f"{info['retained_count']:,}",
                    f"{info['requested_count']:,}", f"{info['total_proposals']:,}", info['sampling_seconds'])
        if info['status'] not in ('complete', 'partial'):
            raise RuntimeError(f"DFM rejection stream {name} failed: {info['status']}")
        timings[f'{name}_seconds'] = info['sampling_seconds']
    # Market days in the event: hold-out days, and training and validation days as their reference.
    samples['market'], samples['market_reference'], market_frequency = event_days(split, threshold)

    comparisons, directions = conditional_comparisons(
        dict(samples, truth=samples['market'], truth_reference=samples['market_reference']), weights,
        projection_seed=settings.projection_seed, matching_seed=settings.matching_seed,
        projections=settings.projections)
    comparisons[['left', 'right']] = comparisons[['left', 'right']].replace(
        {'truth': 'market', 'truth_reference': 'market_reference'})
    dfm_frequency = diagnostics['rejection']['observed_acceptance_fraction']
    values, distances = market_conditional_tables(
        samples, weights, threshold, pool=np.concatenate((samples['market'], samples['market_reference'])),
        directions=directions, matching_seed=settings.matching_seed, noise_seed=settings.noise_seed,
        replicates=settings.noise_replicates, market_frequency=market_frequency, dfm_frequency=dfm_frequency)
    smc = smc_table({f'{key}_smc': diagnostics[f'{key}_smc'] for key in SMC_VARIANTS}, event_frequency=dfm_frequency)
    h_table = h_diagnostic_table(h_results)
    np.savez_compressed(run_dir / 'conditional_samples.npz', **samples, weights=weights, threshold=threshold,
                        projection_directions=directions)
    comparisons.to_csv(run_dir / 'comparisons.csv', index=False)
    values.to_csv(run_dir / 'values.csv')
    distances.to_csv(run_dir / 'distances.csv')
    smc.to_csv(run_dir / 'smc.csv')
    h_table.to_csv(run_dir / 'h_diagnostics.csv')
    figure = plot_market_conditional(
        samples, weights, threshold, leading_direction(split.train, weights), run_dir, sources=plot_sources,
        show=False, title=f'{tail:.0%} event: equal-weight portfolio return below C = {format_return(threshold)}; '
                          f"{DFM_LABELS[config.dfm_network]}, d={reference.d}, k={checkpoint['config']['k']}")
    write_json(run_dir / 'run.json', _clean(dict(
        case=CASE, split=split.info, dfm_network=config.dfm_network, tail=tail, threshold=threshold, steps=steps,
        models=list(MODELS), plot_sources=list(plot_sources), dfm_checkpoint=str(checkpoint_path),
        dfm_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        h_recipe=dict(n_train=n_train, n_validation=n_validation, n_test=n_test, epochs=epochs,
                      validation_interval_steps=validation_interval_steps, widths=list(widths), lr=lr,
                      batch_size=batch_size, initialization_seed=initialization_seed, train_seeds=train_seeds),
        h_training_options={key: dict(H_VARIANTS[key][1]) for key in VARIANTS},
        smc_seed=smc_seed, smc_ess_threshold=smc_ess_threshold, n_samples=count, guided_seed=guided_seed,
        initialization_max_proposals=initialization_max_proposals, assessment_config=asdict(settings),
        market_event_days=len(samples['market']), market_reference_event_days=len(samples['market_reference']),
        sampling_diagnostics=diagnostics, timings=timings, torch_threads=torch.get_num_threads(),
        portfolio_convention=PORTFOLIO_CONVENTION, matching_rule=MATCHING_RULE)))
    logger.info('Conditional market run saved to: %s', run_dir)
    return dict(tail=tail, samples=samples, comparisons=comparisons, values=values, distances=distances, smc=smc,
                h_diagnostics=h_table, h_results=h_results, threshold=threshold, timings=timings,
                sampling_diagnostics=diagnostics, run_dir=run_dir, figure=figure)


def show_market_conditional(output_root, tail, label=None):
    """Display the saved figure and tables of run_market_conditional for one event."""
    run_dir = Path(output_root) / conditional_folder(tail, label)
    run = json.loads((run_dir / 'run.json').read_text())
    for key in VARIANTS:
        if H_VARIANTS[key][3] in run['plot_sources']:
            display(Image(filename=str(run_dir / key / 'validation_loss.png')))
    display(readable_h_diagnostics(pd.read_csv(run_dir / 'h_diagnostics.csv', index_col=0)))
    display(Image(filename=str(run_dir / 'conditional_distributions.png')))
    values, distances = read_table(run_dir / 'values.csv'), read_table(run_dir / 'distances.csv')
    values_note, distances_note = market_table_notes(noise_rows(distances), tail)
    display(Markdown(values_note), values, Markdown(distances_note), distances)
    note, table = readable_smc(pd.read_csv(run_dir / 'smc.csv', index_col=0))
    display(Markdown(note), table)
    return run


def run_market_rejection_comparison(config, split, output_root, *, tail, networks, label=None,
                                    assessment_settings=None):
    """DFM rejection of each frozen DFM of ``networks`` beside the hold-out market days in the event.

    The DFM of config.dfm_network reuses the rejection sample and the projections
    of its conditional run, conditional_folder(tail, label); the other DFMs saved
    under ``output_root`` are sampled here, with the same settings. The table
    (market_rejection_table) is saved as rejection_comparison_tail<tail>.csv.
    """
    settings = assessment_settings or ConditionalAssessmentConfig(**ASSESSMENT_SEEDS)
    output_root = Path(output_root)
    weights, threshold = split.weights, event_threshold(split, tail)
    run_dir = output_root / conditional_folder(tail, label)
    run = json.loads((run_dir / 'run.json').read_text())
    if run['dfm_network'] != config.dfm_network:
        raise ValueError(f"The conditional run at {tail:.0%} conditions {run['dfm_network']}")
    with np.load(run_dir / 'conditional_samples.npz') as saved:
        rejections, directions = {config.dfm_network: saved['rejection']}, saved['projection_directions']
    frequencies = {config.dfm_network: run['sampling_diagnostics']['rejection']['observed_acceptance_fraction']}
    for network in networks:
        if network in rejections:
            continue
        checkpoint, reference = load_market_dfm(output_root / dfm_folder_name(CASE, config.seeds[0], network),
                                                config.device)
        if checkpoint['panel_sha256'] != split.info['panel_sha256']:
            raise ValueError('The frozen DFM was trained on a different panel')
        logger.info("DFM rejection of %s (cap %s proposals)", DFM_LABELS[network],
                    f"{settings.rejection_max_proposals:,}")
        rejections[network], info = sample_dfm_rejection(
            reference, weights, threshold, settings.rejection_n, steps=int(checkpoint['config']['steps']),
            batch_size=settings.rejection_batch_size, seed=settings.rejection_seed,
            max_proposals=settings.rejection_max_proposals)
        if info['status'] not in ('complete', 'partial'):
            raise RuntimeError(f"DFM rejection of {network} failed: {info['status']}")
        frequencies[network] = info['observed_acceptance_fraction']
    market, market_reference, market_frequency = event_days(split, threshold)
    table = market_rejection_table(
        {network: rejections[network] for network in networks}, frequencies, market,
        np.concatenate((market, market_reference)), weights, market_frequency=market_frequency,
        directions=directions, matching_seed=settings.matching_seed, noise_seed=settings.noise_seed,
        replicates=settings.noise_replicates)
    table.to_csv(output_root / f'rejection_comparison_tail{tail}.csv')
    return table


def show_market_rejection_comparison(output_root, tail):
    """Display the saved table of run_market_rejection_comparison."""
    table = read_table(Path(output_root) / f'rejection_comparison_tail{tail}.csv')
    display(Markdown(market_rejection_note(table, tail)), table)
    return table
