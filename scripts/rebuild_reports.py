"""Redraw the figures and readable tables of a saved notebook run from its stored samples.

No DFM or H is trained and nothing is sampled, so this takes about a minute:

    python3 scripts/rebuild_reports.py artifacts/synthetic_unconditional_benchmark/<run>

Writes dfm_distributions.png, dfm_values.csv and dfm_distances.csv in the run folder,
and conditional_distributions.png, table_values.csv and table_distances.csv in each
conditional_tail... folder. conditioning_error.csv, in the run folder, is read from the
saved comparisons.csv of the events. The seeds match the notebook's, so a new run of the
notebook writes the same files.
"""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from synthetic_conditional_truth import population_from_checkpoint  # noqa: E402
from synthetic_benchmark.reporting import (  # noqa: E402
    DFM_LABELS, conditional_tables, conditioning_error_table, dfm_tables, format_return,
    plot_conditional_distributions, plot_dfm_distributions, readable_conditioning_error, table_notes,
)

CASE = 'gaussian'
DFM_NOISE_SEED = 42010  # As in the notebook's unconditional tables.
CONDITIONAL_NOISE_SEED = 42009  # The notebook's default, for runs saved before noise seeds were recorded.


def show(notes, tables):
    for note, table in zip(notes, tables):
        print(note.replace('**', ''))
        print(table.to_string(), '\n')


def rebuild(run):
    config = json.loads((run / 'config.json').read_text())
    seed = config['seeds'][0]
    events = sorted(run.glob('conditional_tail*/run.json'))
    dfm_runs = {network: run / f'{CASE}_seed{seed}_{network}' for network in DFM_LABELS}
    dfm_runs = {network: folder for network, folder in dfm_runs.items() if folder.is_dir()}
    if not dfm_runs:
        raise SystemExit(f'No saved DFM samples under {run}')
    metadata = json.loads(events[0].read_text()) if events else None
    if metadata is None:
        raise SystemExit('The population is read from a conditional run.json; none was found')
    population = population_from_checkpoint(metadata['population_d'], metadata['population_k'], CASE, metadata)

    generated = {}
    for network, folder in dfm_runs.items():
        with np.load(folder / 'unconditional_samples.npz') as saved:
            generated[network] = saved['generated']
            truth = saved['truth']
    plot_dfm_distributions(generated, population, run / 'dfm_distributions.png', show=False)
    verdict = 'gaussian_mlp' if 'gaussian_mlp' in generated else next(iter(generated))
    tables = dfm_tables(generated, truth, population, verdict=verdict, noise_seed=DFM_NOISE_SEED)
    for name, table in zip(('dfm_values.csv', 'dfm_distances.csv'), tables):
        table.to_csv(run / name)
    show(table_notes(len(truth)), tables)

    comparisons = {}
    for path in events:
        folder, settings = path.parent, json.loads(path.read_text())
        comparisons[float(settings['tail'])] = pd.read_csv(folder / 'comparisons.csv')
        with np.load(folder / 'conditional_samples.npz') as saved:
            samples = {key: saved[key] for key in saved.files}
        threshold, tail = float(samples['threshold']), float(settings['tail'])
        sources = tuple(s for s in settings.get('plot_sources', ('rejection', 'refined_exact')) if s != 'truth')
        verdict = next(s for s in sources if s != 'rejection')
        plot_conditional_distributions(
            samples, population, threshold, folder, sources=sources, show=False,
            title=f'{tail:.0%} event: equal-weight portfolio return below '
                  f'C = {format_return(threshold)}; '
                  f'{DFM_LABELS[settings["dfm_network"]]}, d={population.d}, k={population.k}')
        assessment = settings['assessment_config']
        tables = conditional_tables(
            samples, population, threshold, sources=sources, verdict=verdict,
            directions=samples['projection_directions'], matching_seed=assessment['matching_seed'],
            noise_seed=assessment.get('noise_seed', CONDITIONAL_NOISE_SEED),
            replicates=assessment.get('noise_replicates', 100),
            event_frequency=settings['sampling_diagnostics']['rejection']['observed_acceptance_fraction'])
        for name, table in zip(('table_values.csv', 'table_distances.csv'), tables):
            table.to_csv(folder / name)
        show(table_notes(min(len(samples[s]) for s in sources), tail), tables)
    # The rarer event last, as in the notebook.
    table = conditioning_error_table(dict(sorted(comparisons.items(), reverse=True)))
    table.to_csv(run / 'conditioning_error.csv')
    note, readable = readable_conditioning_error(table)
    show([note], [readable])
    print('Figures and tables written under', run)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    pd.set_option('display.width', 200)
    rebuild(Path(sys.argv[1]))
