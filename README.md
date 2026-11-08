# Master Thesis: Diffusion Factor Models

Generate joint return scenarios with a diffusion factor model (DFM), then condition
them on a portfolio stress event using learned H guidance and an exact final
sampling step. The synthetic Gaussian factor population of Chen et al. provides
known conditional truth for checking both the generator and the conditioning method.

## Where to start

- [Benchmark notebook](synthetic_dfm_conditional_benchmark.ipynb): run the experiment,
  follow the formulas alongside the code, and inspect its generated figures and tables.
- [Methods](METHODS.md): detailed model definitions, retained refinements, derivations,
  exact-step conditioning, and approximation limits.
- [Synthetic dependencies](requirements-synthetic.txt): pinned Python 3.12 environment.

## Current experiment

The notebook's standard configuration uses **500 assets, 8 factors, and 2,500 / 750
training / validation observations** drawn from the Chen et al. Gaussian factor
population. It compares two compact score networks on the same observations:

- **DFM (MLP):** the factor-aware score of Chen et al. with an MLP.
- **DFM (Gaussian + MLP):** the same outer score with an explicit Gaussian factor
  component and a learned MLP correction.

Both use PCA initialization, staged loading updates, and log-time,
variance-weighted score matching. The notebook freezes **DFM (Gaussian + MLP)** for
conditioning at the 5% and 1% equal-weight portfolio thresholds.

CDG-ML and Refined CDG-ML train on the same 40,000 reference trajectories for 50
epochs. The benchmark compares five sources: conditional population truth, DFM
rejection, refined guidance with an exact final step, baseline guidance with an
exact final step, and no guidance with an exact final step. Rejection is the
conditional numerical DFM law that guidance targets. Distances include independent
Monte Carlo reference levels at matched sample sizes.
Figures show distributions only, against the exact law: CDF and Q–Q plots of the
portfolio and Q–Q plots along the largest factor direction. They draw DFM rejection
and Refined CDG-ML. Two short tables per event put the expected shortfall and
volatility next to their exact values, then three distances next to a 95% noise
level with a yes/no verdict. All five sources remain in the saved samples and CSV files.

This tests conditional generation on a Gaussian benchmark; it does not establish
real-return tail accuracy. Attribution and methodological limits are in [METHODS.md](METHODS.md).

## Run locally

From the repository root, using Python 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-synthetic.txt
jupyter notebook synthetic_dfm_conditional_benchmark.ipynb
```

The default Chen population needs no market panel. The alternative
`CONFIG_OVERRIDES["population"] = "market"` requires a locally cleaned training panel:

```text
artifacts/market_data_cleaning/20261029T104817954744Z/train_complete.npz
```

Set `CONFIG_OVERRIDES["market_panel"]` to its location. This input is not included
in Git; its `returns_decimal_float64` array contains the training returns with
days as rows and stocks as columns. The repository does not include the panel's
download and cleaning workflow.

Set `MODE = "smoke"` for 12 assets and reduced budgets. Smoke mode uses the
same configured population.
DFM settings are in `CONFIG_OVERRIDES`; H training and sampling settings are in
`H_RECIPE`. Restart the kernel and run cells in order after changing configuration.

## Codebase

| Location | Responsibility |
| --- | --- |
| [NeuralNets.py](NeuralNets.py), [DFM.py](DFM.py) | Score architectures, OU corruption, and generic DFM training |
| [sampling.py](sampling.py), [reference.py](reference.py) | Reverse integration and the frozen score reference |
| [conditioning.py](conditioning.py), [h_model.py](h_model.py), [guided_sampling.py](guided_sampling.py) | Portfolio event, reduced coordinates, H training, guidance, and exact final transition |
| [synthetic_conditional_truth.py](synthetic_conditional_truth.py) | Gaussian populations, calibration, exact conditional draws, and moments |
| [synthetic_benchmark/](synthetic_benchmark/) | Configuration, fitting, checkpoint selection, independent references, metrics, and figures |
| [return_preprocessing.py](return_preprocessing.py), [reference_checkpoint.py](reference_checkpoint.py), [h_setup.py](h_setup.py) | Return-coordinate transforms and the separate saved-equity-checkpoint workflow |
| [scripts/](scripts/) | Report rebuilding from saved runs, remote benchmarks, and earlier architecture/guidance experiments |
| [tests/](tests/) | Mathematical identities, sampling behavior, training, checkpoints, and benchmark checks |

## Outputs

Each notebook execution writes a timestamped directory under
`artifacts/synthetic_unconditional_benchmark/`. It contains the effective configuration,
environment, both DFM checkpoints and assessments, and a `conditional_tail…` directory
for each event. Completed conditional assessments save samples, H checkpoints,
`summary.csv`, `comparisons.csv`, `h_diagnostics.csv`, the readable tables
`table_values.csv` and `table_distances.csv`, `conditional_distributions.png`, and
`run.json` with seeds, budgets, timings, and reference identity. The run folder holds
`dfm_distributions.png`, `dfm_values.csv` and `dfm_distances.csv` for the two DFMs.

To redraw these figures and tables from a saved run's samples, without training or
sampling again (about a minute):

```bash
python3 scripts/rebuild_reports.py artifacts/synthetic_unconditional_benchmark/<run>
```

Generated artifacts and the personal experiment journal stay local. A completed
unconditional stage does not imply that both conditional benchmarks have finished.

## Optional Modal benchmark

[scripts/modal_benchmark.py](scripts/modal_benchmark.py) runs **DFM (MLP) on the Chen
Gaussian population** at a chosen dimension. It is a separate configuration from
the market-calibrated two-model notebook. With Modal installed and authenticated:

```bash
modal run --detach scripts/modal_benchmark.py::high_dimension --d 2048 --k 16 --tails 0.05,0.01
```

One container fits the DFM, followed by one conditional benchmark container per
event. Results persist on the `thesis-runs` volume.

## Tests

```bash
PYTHONPATH=.:tests python3 -m unittest discover -s tests -p "test_*.py"
```
