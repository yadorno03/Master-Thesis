# Focused conditional-return comparison

Start with [`focused_comparison.ipynb`](../focused_comparison.ipynb). It requires
only the personal factor MLP, OU/Euler sampler, frozen-reference H network and
rejection routines already on `main`. None of PRs #33–#42 is a dependency.
The notebook does not import or install the authors' U-Net/DDPM backend.

## Run

Use Python 3.12 in a virtual environment:

```sh
python -m pip install -r requirements-comparison.txt
python scripts/run_comparison_notebook.py --mode demo --output artifacts/comparison/demo.ipynb
# Execution check only, not meaningful fit accuracy:
python scripts/run_comparison_notebook.py --mode smoke --output artifacts/comparison/smoke.ipynb
python -m unittest discover -s tests -v
```

The output notebook must not exist. Each execution also creates a unique run
folder with `synthetic/` and `market/` subfolders. Those contain DFM/H bundles,
raw training/evaluation arrays, exact synthetic conditional moments, JSON/CSV
metrics, seeds, source snapshots, training logs and figures. The default demo
fits each DFM for 1,000 epochs and H for 40; those budgets are illustrative,
not established convergence settings. Change configuration in the notebook
before execution. Nothing selects seeds or hyperparameters against evaluation.

The synthetic data follow the Gaussian parameter recipe in
[DFM Appendix D](https://arxiv.org/html/2504.06566v5#A4), with reduced dimensions.
The portfolio-tail conditioning is an extension to that unconditional study.
Analytical conditional moments and exact truncated draws are evaluation-only;
training sees only observed returns, and H sees only fitted-DFM trajectory labels.

The real-data example uses [Kenneth French's ten industry daily portfolios](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
It downloads and hashes the public archive once, uses the value-weighted table,
converts percent to decimal returns, and reports missing-row deletions. The
fixed split is training 2010–2019 / held-out 2020–2024. The threshold and scaling
use training data only. Historical input revisions change its hash; retain the
cached archive to reproduce a run exactly. No market observations or model
weights are committed. Set `COMPARISON_RETURN_CSV` to use a local CSV with an
ordered date column and decimal-return columns covering the same windows.
This models daily cross-sectional scenarios, not serial dynamics.

## Read the results

- Compare both DFM rejection and H methods to **true population conditional**
  covariance on synthetic data. Compare to a separate **fitted DFM conditional**
  rejection sample to isolate guidance fidelity from DFM fit quality.
- Raw H guidance retains event violations in its metrics. H plus terminal
  rejection guarantees the event but can still have distributional bias.
- Every method has the same requested count and integration grid. Timing
  includes initialization, rejected proposals and overshoot. H setup cost is
  separate and included in the first-batch total. Warm-up, reference generation,
  evaluation and file I/O are outside generation timing.
- Sampling repeats share one fitted model. Their SD is not uncertainty across
  complete training runs. Small covariance error is not joint-law recovery.
- The real-data population covariance is unknown. Held-out historical tail
  covariance is descriptive and can reflect distribution shift or temporal
  dependence. Independent DFM rejection is only a numerical model reference.
- Inspect unconditional fit and H validation before interpreting conditional
  comparisons. More samples cannot fix model bias. Validate grid/early-stopping
  sensitivity and independent full fits before making scientific claims.

## Repository audit, 23 October 2026

At audit time, `main` was `56a3507`. There were ten open PRs (#33–#42), a
stack for author training, DDPM reference/sampling, ambient H, checkpoints,
runner, compatibility and batching. The first five author-stack changes had
already merged (#28–#32), including guidance strength, the QR coordinate fix
and shared preprocessing. The remaining stack is unnecessary for this notebook.
Some local synthetic/equity notebooks and diagnostics were uncommitted in the
original checkout; this PR is independently based on `main` and preserves
those files. Open PRs have not been closed or merged as part of this work.
