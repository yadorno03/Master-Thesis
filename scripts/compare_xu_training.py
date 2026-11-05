"""Compare DFM training setups using observations and validation data only.

Example: python3 scripts/compare_xu_training.py --output artifacts/xu_training_comparison
Population truth is used to generate independent datasets, never by fit_dfm or
candidate selection. Post-selection truth diagnostics belong to a separate run.
"""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import torch

from synthetic_conditional_truth import make_population, population_metadata, population_details
from synthetic_benchmark.configuration import Config, Streams, write_json
from synthetic_benchmark.unconditional import fit_dfm
from reference import FrozenDFMReference
from sampling import generate_samples


SETUPS = {
    'baseline': {},
    'log_importance': {'dfm_time_sampling': 'log_importance'},
    'log_balanced': {'dfm_time_sampling': 'log_balanced', 'lr': .001},
    'pca_uniform': {'dfm_initialization': 'pca', 'lr': .001,
                    'dfm_loading_lr_scale': .1, 'dfm_loading_warmup_epochs': 100},
    'pca_balanced': {'dfm_initialization': 'pca', 'dfm_time_sampling': 'log_balanced',
                    'lr': .001, 'dfm_loading_lr_scale': .1, 'dfm_loading_warmup_epochs': 100},
    'pca_importance': {'dfm_initialization': 'pca', 'dfm_time_sampling': 'log_importance',
                      'lr': .001, 'dfm_loading_lr_scale': .1, 'dfm_loading_warmup_epochs': 100},
}


def run(output, names, seeds, epochs):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    base = Config(dfm_epochs=epochs)
    population = make_population(base.d, k=base.k)
    write_json(output / 'population.json', population_details(population, 'gaussian'))
    rows = []
    for seed in seeds:
        streams = Streams(seed)
        train = population.sample(base.n_train, streams.np('train'))
        validation = population.sample(base.n_val, streams.np('validation'))
        validation_covariance = np.cov(validation, rowvar=False)
        for name in names:
            cfg = replace(base, **SETUPS[name])
            folder = output / f'{name}_seed{seed}'
            folder.mkdir()
            started = time.perf_counter()
            model, history, selected = fit_dfm(train, validation, cfg, streams)
            pd.DataFrame(history).to_csv(folder / 'dfm_curve.csv', index=False)
            torch.save(dict(state=model.state_dict(), config=asdict(cfg), selected_epoch=selected,
                            **population_metadata(population, 'gaussian')), folder / 'dfm.pt')
            # Common random numbers across candidates; separate from fit streams.
            generated = generate_samples(FrozenDFMReference(model), base.n_eval,
                num_steps=base.steps, generator=torch.Generator().manual_seed(90101)).numpy()
            covariance = np.cov(generated, rowvar=False)
            result = dict(setup=name, seed=seed, selected_epoch=selected,
                validation_loss=history[selected - 1]['validation'],
                validation_covariance_error=float(np.linalg.norm(covariance - validation_covariance)
                                                 / np.linalg.norm(validation_covariance)),
                validation_mean_error=float(np.linalg.norm(generated.mean(0) - validation.mean(0))
                                            / np.sqrt(np.trace(validation_covariance))),
                validation_marginal_ks=float(np.mean([ks_2samp(generated[:, i], validation[:, i]).statistic
                                                     for i in range(base.d)])),
                seconds=time.perf_counter() - started)
            write_json(folder / 'result.json', result)
            np.savez_compressed(folder / 'validation_samples.npz', generated=generated)
            rows.append(result)
            pd.DataFrame(rows).to_csv(output / 'comparison.csv', index=False)
            print(result, flush=True)
    summary = pd.DataFrame(rows).groupby('setup').validation_covariance_error.mean().sort_values()
    write_json(output / 'selection.json', dict(
        criterion='Mean generated-versus-validation relative covariance error across training seeds',
        selected_setup=str(summary.index[0]), values=summary.to_dict(),
        note='Selection uses no analytical population moments. Validation DSM objectives differ by setup.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--setups', nargs='+', choices=SETUPS, default=list(SETUPS))
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--epochs', type=int, default=1000)
    args = parser.parse_args()
    run(args.output, args.setups, args.seeds, args.epochs)
