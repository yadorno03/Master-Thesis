# Master Thesis: Diffusion Factor Models

Gaussian return simulation with an unconditional diffusion factor model (DFM)
and learned H guidance for portfolio-tail conditioning. The
[benchmark notebook](synthetic_dfm_conditional_benchmark.ipynb) explains the
population, training objectives, conditioning coordinates, and diagnostics.

## Setup

The notebook's `"standard"` mode uses 128 assets and 8 factors (Chen et al., 2025, use 2,048 assets and 16 factors).
Set `MODE = "smoke"` for 12 assets, 8 factors, and reduced training budgets.
Edit `CONFIG_OVERRIDES`
to change the DFM configuration; H budgets are set in the conditional benchmark call.
Restart the kernel and run all cells in order after switching branches.

The notebook selects PCA initialization and log-balanced score training, then
freezes the DFM. `run_conditional_benchmark` in
[conditional.py](synthetic_benchmark/conditional.py) trains CDG-ML (Guo et al., 2026;
without the five refinements) and Refined CDG-ML (with all five) on the same 40,000
trajectories for 50 epochs. It then compares four sources of conditional returns:

* the conditional ground truth;
* DFM rejection sampling;
* CDG-ML;
* Refined CDG-ML with the exact final sampling step.

Every distance is reported next to a Monte Carlo level computed at the same sample size.

## Running on Modal

[scripts/modal_benchmark.py](scripts/modal_benchmark.py) runs the same pipeline on
Modal CPU containers at any dimension. One container trains the DFM, then one container
per event probability runs the conditional benchmark:

```bash
modal run --detach scripts/modal_benchmark.py::high_dimension --d 2048 --k 16 --tails 0.05,0.01
```

## Tests

```bash
PYTHONPATH=.:tests python3 -m unittest discover -s tests -p "test_*.py"
```
