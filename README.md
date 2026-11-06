# Master Thesis: Diffusion Factor Models

Gaussian return simulation with an unconditional diffusion factor model (DFM)
and learned H guidance for portfolio-tail conditioning. The
[benchmark notebook](synthetic_dfm_conditional_benchmark.ipynb) explains the
population, training objectives, conditioning coordinates, and diagnostics.

## Setup

Use Python 3.12 and run from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-synthetic.txt
python -m notebook synthetic_dfm_conditional_benchmark.ipynb
```

The notebook's `"standard"` mode uses 128 assets and 8 factors (Xu uses 2,048 assets and 16 factors).
Set `MODE = "smoke"` for 12 assets, 8 factors, and reduced training budgets.
Edit `CONFIG_OVERRIDES`
to change the DFM configuration; H budgets are set in the final simulation call.
Restart the kernel and run all cells in order after switching branches.

The notebook selects PCA initialization and log-balanced score training, then
freezes the DFM and trains both Raw H (without the five refinements) and Refined H
(with all five) on shared trajectory splits. Both H validation curves precede the
performance figure. Its portfolio CDF compares both models before and after
event filtering with DFM rejection sampling and the exact conditional CDF.
The Q–Q compares event-filtered Raw H, event-filtered Refined H and DFM rejection
with conditional truth; the population loading projections compare only conditional
truth and event-filtered Refined H. `compare_saved_h_run` reopens a completed run and shows
the same plots without retraining completed models. Old population designs require fresh DFM and H checkpoints.
Generated data, checkpoints, figures, and reports under `data/` and `artifacts/`
stay local and are ignored by Git.

## Scripts and checks

```bash
# Trace the six stages with 100 assets and reduced training budgets.
python scripts/trace_gaussian_experiment.py --mode smoke

# Compare DFM training configurations against observed validation returns.
python scripts/compare_xu_training.py --output artifacts/xu_training_comparison

# Opt-in late-time H probability targets on an existing frozen DFM (see below).
python scripts/compare_h_soft_targets.py --source-run <saved run> --stage estimate

# Run software checks, including the notebook with reduced budgets.
env -u SYNTHETIC_FROZEN_DFM_RUN \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
  python -m unittest discover -s tests -v
```

### Late-time soft H targets (opt-in)

Refined H regresses on the binary terminal outcome except at the last preterminal
row, where it uses the exact final Euler-step probability. `train_h` and
`HNetwork.sample_trajectories` accept `soft_target_mode` (`"off"` default,
`"reuse_final"`, `"continuation"`) with a positive remaining-time
`soft_target_window` to replace binary targets on late rows by probabilities.
`"reuse_final"` gives every selected row of a path that path's final-step
probability; `"continuation"` averages the final-step probability over `M` fresh
unguided branches from each selected state (own seeded streams, bounded
microbatches). Defaults and saved Raw/Refined H runs are unchanged.

```bash
SRC=artifacts/synthetic_unconditional_benchmark/<saved run with the frozen DFM>
python scripts/compare_h_soft_targets.py --source-run $SRC --stage estimate  # timed probes only
python scripts/compare_h_soft_targets.py --source-run $SRC --stage pilot     # 2,048/512/512 paths, 25 epochs
python scripts/compare_h_soft_targets.py --source-run $SRC --stage full \
    --init-seeds 2027 2028 2029                                              # original budgets
```

The runner loads the frozen DFM, never retrains it, and writes a fresh
`artifacts/h_soft_target_ablation/<timestamp>/` with `soft_target_comparison.json`,
per-arm checkpoints and samples, and CSV summaries. Init seeds on fixed data
measure optimization variability only; add `--data-offsets` for independent
trajectory data. Its `soft_target_comparison.png` summarizes the arms; the main
notebook does not display this ablation. Pilot budgets find runtime and integration problems; they do not
establish convergence.

Software checks verify execution and numerical contracts; convergence and
conditional-distribution recovery require separate experimental assessment.
The notebook test runs with reduced budgets in a temporary directory. Unsetting
`SYNTHETIC_FROZEN_DFM_RUN` ensures it trains from scratch without local checkpoints.
Additional equity-data tools use `requirements-equities.txt`.
