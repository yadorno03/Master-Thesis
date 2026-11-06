# Master Thesis: Diffusion Factor Models

Gaussian return simulation with an unconditional diffusion factor model (DFM)
and learned H guidance for portfolio-tail conditioning. The
[benchmark notebook](synthetic_dfm_conditional_benchmark.ipynb) explains the
population, training objectives, conditioning coordinates, and diagnostics.

## Setup

The notebook's `"standard"` mode uses 128 assets and 8 factors (Chen et al., 2025, use 2,048 assets and 16 factors).
Set `MODE = "smoke"` for 12 assets, 8 factors, and reduced training budgets.
Edit `CONFIG_OVERRIDES`
to change the DFM configuration; H budgets are set in the final simulation call.
Restart the kernel and run all cells in order after switching branches.

The notebook selects PCA initialization and log-balanced score training, then
freezes the DFM and trains both CDG-ML (Guo et al., 2026; without the five refinements) and Refined CDG-ML
(with all five) on shared trajectory splits. Both H validation curves precede the
performance figure. Its portfolio CDF compares both models before and after
event filtering with DFM rejection sampling and the exact conditional CDF.
The Q–Q compares event-filtered CDG-ML, event-filtered Refined CDG-ML and DFM rejection
with conditional truth; the population loading projections compare only conditional
truth and event-filtered Refined CDG-ML. `compare_saved_h_run` reopens a completed run and shows
the same plots without retraining completed models. 
