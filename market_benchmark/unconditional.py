"""Unconditional DFMs on the market panel: training, sampling and comparison with the hold-out days.

The DFMs and their recipe are those of the synthetic notebook
(synthetic_benchmark.configuration.notebook_config and network_config).
"""
from dataclasses import asdict
from pathlib import Path
import json
import logging
import time

import numpy as np
import pandas as pd
import torch
from IPython.display import display, Image, Markdown

from DFM import UnconditionalDFM
from reference import FrozenDFMReference
from sampling import generate_samples
from synthetic_benchmark.configuration import DFM_NETWORKS, NETWORK_SETTINGS, Streams, network_config, write_json
from synthetic_benchmark.reporting import DFM_LABELS, plot_dfm_validation
from synthetic_benchmark.unconditional import dfm_folder_name, fit_dfm, initialize_unconditional_run
from .reporting import (
    leading_direction, market_dfm_tables, market_table_notes, noise_rows, plot_market_dfm_distributions,
)

CASE = 'market'
logger = logging.getLogger(__name__)


def load_market_dfm(folder, device='cpu'):
    """The saved DFM of a market run, frozen: (checkpoint, reference)."""
    checkpoint = torch.load(Path(folder) / 'dfm.pt', map_location=device, weights_only=True)
    saved = checkpoint['config']
    model = UnconditionalDFM(saved['d'], saved['k'], sigma_max=2.0, T=saved['T'], t0=saved['t0'],
                             hidden_dims=[saved['width'], saved['width']],
                             score_network=saved['dfm_network'], regimes=saved.get('dfm_regimes'),
                             refit_knots=saved.get('dfm_refit_knots', 0)).to(device)
    model.load_state_dict(checkpoint['state'])
    return checkpoint, FrozenDFMReference(model)


def run_market_unconditional(config, split, output, *, networks=DFM_NETWORKS, network_settings=None,
                             verdict='regime_mlp'):
    """Train each DFM on the training days, sample it and compare it with the hold-out days.

    Each network trains with network_config(config, network, network_settings), its
    settings saved in network_settings.json. The networks share the days and the random streams
    of config.seeds[0]. The validation days select the epoch; the evaluation days enter the
    tables only (market_dfm_report, with the within-noise verdict for ``verdict``).
    Everything is saved under ``output``, which must not exist.
    """
    settings = NETWORK_SETTINGS if network_settings is None else network_settings
    output = Path(output)
    if split.train.shape != (config.n_train, config.d) or len(split.validation) != config.n_val:
        raise ValueError('The split does not match the configured dimensions')
    initialize_unconditional_run(output, config)
    write_json(output / 'split.json', split.info)
    write_json(output / 'network_settings.json', {network: settings.get(network, {}) for network in networks})
    started = time.perf_counter()
    seed, timings = config.seeds[0], {}
    for network in networks:
        cfg = network_config(config, network, settings)
        logger.info("Training %s on %s market days, d=%d, k=%d", DFM_LABELS[network], f"{cfg.n_train:,}", cfg.d,
                    cfg.k)
        folder = output / dfm_folder_name(CASE, seed, network)
        folder.mkdir()
        streams = Streams(seed)
        write_json(folder / 'streams.json', streams.seeds)
        fit_started = time.perf_counter()
        model, history, selected = fit_dfm(split.train, split.validation, cfg, streams)
        fit_seconds = time.perf_counter() - fit_started
        pd.DataFrame(history).to_csv(folder / 'dfm_curve.csv', index=False)
        torch.save(dict(state=model.state_dict(), config=asdict(cfg), selected_epoch=selected, case=CASE,
                        panel_sha256=split.info['panel_sha256']), folder / 'dfm.pt')
        reference = FrozenDFMReference(model)
        generation_started = time.perf_counter()
        # The DFM generates standardized returns; map them back to returns.
        generated = reference.standardization.to_returns(generate_samples(
            reference, cfg.n_eval, num_steps=cfg.steps, generator=streams.torch('generated', cfg.device)))
        timings[network] = dict(fit_seconds=fit_seconds, generation_seconds=time.perf_counter() - generation_started,
                                selected_epoch=int(selected))
        np.savez_compressed(folder / 'unconditional_samples.npz', generated=generated)
        score = model.factor_score_network
        if hasattr(score, 'regime_scales'):
            factor, residual, table = (value.detach().cpu().tolist() for value in score.regime_scales())
            write_json(folder / 'regimes.json', dict(factor_scales=factor, residual_scales=residual,
                                                     probabilities=table))
    values, distances = market_dfm_report(output, split, networks=networks, verdict=verdict)
    write_json(output / 'runtime.json', dict(seconds=time.perf_counter() - started, networks=timings))
    logger.info('Unconditional market run saved to: %s', output)
    return dict(values=values, distances=distances, timings=timings, output=output)


def market_dfm_report(output, split, *, networks=DFM_NETWORKS, verdict='regime_mlp', matching_seed=42010,
                      noise_seed=42012):
    """Tables and figure of the saved samples of each DFM against the hold-out days (market_dfm_tables).

    Reads the samples saved by run_market_unconditional under ``output`` and writes
    dfm_values.csv, dfm_distances.csv and dfm_distributions.png there.
    """
    output = Path(output)
    seed = json.loads((output / 'config.json').read_text())['seeds'][0]
    generated = {}
    for network in networks:
        with np.load(output / dfm_folder_name(CASE, seed, network) / 'unconditional_samples.npz') as saved:
            generated[network] = saved['generated']
    values, distances = market_dfm_tables(generated, split.evaluation, split.insample, split.weights,
                                          train=split.train, verdict=verdict, matching_seed=matching_seed,
                                          noise_seed=noise_seed)
    values.to_csv(output / 'dfm_values.csv')
    distances.to_csv(output / 'dfm_distances.csv')
    plot_market_dfm_distributions(generated, split.evaluation, split.weights,
                                  leading_direction(split.train, split.weights), output / 'dfm_distributions.png',
                                  show=False)
    return values, distances


def show_market_unconditional(output, *, networks=DFM_NETWORKS):
    """Display the saved figure, validation curves and tables of run_market_unconditional."""
    output = Path(output)
    config = json.loads((output / 'config.json').read_text())
    display(Image(filename=str(output / 'dfm_distributions.png')))
    for network in networks:
        plot_dfm_validation(output / dfm_folder_name(CASE, config['seeds'][0], network), DFM_LABELS[network])
    values, distances = (pd.read_csv(output / name, index_col=0, dtype=str, keep_default_na=False)
                         for name in ('dfm_values.csv', 'dfm_distances.csv'))
    values_note, distances_note = market_table_notes(noise_rows(distances))
    display(Markdown(values_note), values, Markdown(distances_note), distances)
    return values, distances
