# Master Thesis: Diffusion Factor Models

Gaussian return simulation with an unconditional diffusion factor model (DFM)
and learned H guidance for portfolio-tail conditioning. The
[benchmark notebook](synthetic_dfm_conditional_benchmark.ipynb) explains the
population, training objectives, conditioning coordinates, and diagnostics.

## Setup

The notebook's `"standard"` mode uses a Gaussian factor population calibrated on 500
real stocks: 8 factors, 2,500 training and 750 validation observations, about the
2,516 training and 756 validation days of the return panel. The panel
(`artifacts/market_data_cleaning/.../train_complete.npz`) is produced locally by the
market data cleaning step and is not in git; `CONFIG.market_panel` points to it.
Set `MODE = "smoke"` for 12 assets and reduced training budgets.
Edit `CONFIG_OVERRIDES` to change the DFM configuration; H budgets are set in `H_RECIPE`.
Restart the kernel and run all cells in order after switching branches.

The notebook trains two DFMs on the same data, both with PCA initialization and
log-balanced score training: the base DFM, with the score network of Chen et al. (2025),
and I3, the same score form whose network g adds an explicit Gaussian factor part to
its MLP. It freezes the I3 DFM. For the 5% and the 1% events of the equal-weight
portfolio, `run_conditional_benchmark` in [conditional.py](synthetic_benchmark/conditional.py)
trains CDG-ML of Guo et al. (2026) (without the five refinements) and Refined CDG-ML
(with all five) on the same 40,000 trajectories for 50 epochs. It then compares five
sources of conditional returns:

* the conditional ground truth;
* DFM rejection sampling, the law that guidance targets;
* Refined CDG-ML with the exact final sampling step;
* CDG-ML with the exact final sampling step;
* no guidance, with the exact final sampling step.

Every distance is reported next to a Monte Carlo level computed at the same sample size.

## Running on Modal

[scripts/modal_benchmark.py](scripts/modal_benchmark.py) runs the base DFM on the
Chen et al. (2025) population on Modal CPU containers, at any dimension. One container
trains the DFM, then one container per event probability runs the conditional benchmark:

```bash
modal run --detach scripts/modal_benchmark.py::high_dimension --d 2048 --k 16 --tails 0.05,0.01
```

## Tests

```bash
PYTHONPATH=.:tests python3 -m unittest discover -s tests -p "test_*.py"
```
