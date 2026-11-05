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

Set `MODE = "smoke"` in the notebook for a short run, or use `"standard"`
for the default 100-asset, eight-factor experiment. Edit `CONFIG_OVERRIDES`
to change the DFM configuration; H budgets are set in the final simulation call.
Restart the kernel and run all cells in order after switching branches.

The notebook selects PCA initialization and log-balanced score training, then
freezes the DFM and trains H with input scaling and extra late-time coverage.
It compares guided samples with exact Gaussian conditional truth and independent
DFM rejection samples. Old population designs require fresh DFM and H checkpoints.
Generated data, checkpoints, figures, and reports under `data/` and `artifacts/`
stay local and are ignored by Git.

## Scripts and checks

```bash
# Trace the six stages with 100 assets and reduced training budgets.
python scripts/trace_gaussian_experiment.py --mode smoke

# Compare DFM training configurations against observed validation returns.
python scripts/compare_xu_training.py --output artifacts/xu_training_comparison

# Run software checks, including the notebook with reduced budgets.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
  python -m unittest discover -s tests -v
```

Software checks verify execution and numerical contracts; convergence and
conditional-distribution recovery require separate experimental assessment.
Additional equity-data tools use `requirements-equities.txt`. Optional author
backend tests use `requirements-dfm.txt` in an isolated environment and skip
when that backend is unavailable.
