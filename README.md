# Master Thesis: Diffusion Factor Models

Generate joint returns with a diffusion factor model (DFM), then condition them on a portfolio stress event using learned H guidance and an exact final step. A synthetic Gaussian population provides known conditional truth.

- [Benchmark notebook](synthetic_dfm_conditional_benchmark.ipynb): experiment, figures and tables.
- [Methods](METHODS.md): formulas, training choices and interpretation.

## Current experiment

The Chen et al. Gaussian population uses **500 assets, 8 factors, 2,500 training observations and 750 validation observations**. Two networks train on the same returns:

| Model | Score network |
| --- | --- |
| DFM (MLP) | Factor-aware score with an MLP |
| DFM (Gaussian + MLP) | Gaussian factor component plus an MLP correction |

Both model each asset's returns standardized by its training mean and standard deviation, and use PCA initialization, staged loading updates and log-time, variance-weighted score matching. We freeze **DFM (Gaussian + MLP)** for conditioning at the **5% and 1% equal-weight portfolio thresholds**.

CDG-ML and Refined CDG-ML use the same 40,000 training trajectories for 50 epochs. Five conditional sources are assessed: population truth, DFM rejection, CDG-ML, Refined CDG-ML and no guidance. The last three use an exact final step.

| Comparison | What it checks |
| --- | --- |
| DFM rejection vs population truth | Learned numerical generator |
| Guidance vs DFM rejection | Conditioning procedure |
| Guidance vs population truth | Combined error |

The notebook shows CDF and Q–Q plots, portfolio values and three distances to population truth: portfolio KS and covariance error against the analytic law, sliced W1 against a truth sample. Conditional figures and tables focus on DFM rejection and Refined CDG-ML; saved results retain all five sources. Distance tables include sampling-noise levels at matched sample sizes. A final table compares CDG-ML and Refined CDG-ML with DFM rejection, beside the distance between two independent rejection streams.

An exact final step enforces the event but does not ensure the correct conditional distribution. This Gaussian benchmark does not establish real-return tail accuracy.

## Run locally

From the repository root, using Python 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-synthetic.txt
jupyter notebook synthetic_dfm_conditional_benchmark.ipynb
```

Run the cells in order. The experiment generates its data and needs no external dataset.

## Results

Each execution saves its configuration, checkpoints, samples, figures and tables under `artifacts/synthetic_unconditional_benchmark/`. Use the saved configuration and completed assessments when interpreting a run.

## Tests

```bash
PYTHONPATH=.:tests python3 -m unittest discover -s tests -p "test_*.py"
```
