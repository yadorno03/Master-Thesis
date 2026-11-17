# Master Thesis: Diffusion Factor Models

Generate joint returns with a diffusion factor model (DFM), then condition them on a portfolio stress event using learned H guidance and an exact final step. A synthetic Gaussian population provides known conditional truth.

- [Benchmark notebook](synthetic_dfm_conditional_benchmark.ipynb): experiment, figures and tables.
- [Market notebook](market_dfm_conditional_benchmark.ipynb): the same DFMs, conditional models and tables on daily US equity returns, with hold-out market days as reference. It reads local market data that is not part of the repository (see below).
- [Methods](METHODS.md): formulas, training choices and interpretation.

## Current experiment

The Chen et al. Gaussian population uses **500 assets, 8 factors, 2,500 training observations and 750 validation observations**. Three networks train on the same returns:

| Model | Score network |
| --- | --- |
| DFM (MLP) | Factor-aware score with an MLP |
| DFM (Gaussian + MLP) | Gaussian factor component plus an MLP correction |
| DFM (Regimes + MLP) | Scale mixture of Gaussians (volatility regimes) plus an MLP correction |

All model each asset's returns standardized by its training mean and standard deviation, and use PCA initialization, staged loading updates and log-time, variance-weighted score matching. DFM (Gaussian + MLP) and DFM (Regimes + MLP) keep their analytic factor part at its fit to the training returns, train only the MLP and then refit the linear part of the network exactly. We freeze **DFM (Gaussian + MLP)** for conditioning at the **5% and 1% equal-weight portfolio thresholds**.

CDG-ML and Refined CDG-ML use the same 40,000 training trajectories for 50 epochs. Five conditional sources are assessed: population truth, DFM rejection, CDG-ML, Refined CDG-ML, and Refined CDG-ML corrected by sequential Monte Carlo. The three models use an exact final step.

| Comparison | What it checks |
| --- | --- |
| DFM rejection vs population truth | Learned numerical generator |
| Guidance vs DFM rejection | Conditioning procedure |
| Guidance vs population truth | Combined error |

The notebook shows CDF and Q–Q plots, portfolio values and three distances to population truth: portfolio KS and covariance error against the analytic law, sliced W1 against a truth sample. Conditional figures focus on DFM rejection, Refined CDG-ML and Refined CDG-ML + SMC; the tables list every source. Distance tables include sampling-noise levels at matched sample sizes. A final table compares every model with DFM rejection, beside the distance between two independent rejection streams, and another compares the DFM rejection of DFM (Gaussian + MLP) and DFM (Regimes + MLP) with population truth.

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

The market notebook uses the same environment. It reads Yahoo Finance adjusted closes from `data/raw/equities/yahoo_2026-10-06/adjusted_close.csv.gz` and index membership from `data/universes/us_equities_2026-10-06.csv`, and saves the panel it builds to `data/processed/market_panel.npz`; none of these files is in the repository. `MARKET_MODE=smoke` runs it on 12 assets with reduced budgets, and `MARKET_SAVED_RUN=<run folder>` displays a finished run without computing.
