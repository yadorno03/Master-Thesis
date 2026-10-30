# Master-Thesis

`NeuralNets.py` provides `FactorScoreNetwork`, the factor-structured score
architecture. Its input is a return batch `[batch, d]`
and forward OU noising times `[batch, 1]`; its output has shape `[batch, d]`.
Call `project_orthogonal()` after optimizer updates to restore orthonormal loadings.

Use Python 3.12 with the project's dependency manifest:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-equities.txt
python -m unittest discover -s tests -v
```

After training with `DFM.train_dfm`, generate return scenarios with:

```python
from sampling import generate_samples
samples = generate_samples(model, num_samples=1000, num_steps=100)
```

The Euler–Maruyama sampler starts from Gaussian noise and runs for `T - t0`.
Its score uses forward time `T - s`; optional additive guidance receives elapsed
generative time `s`. `return_trajectory=True` includes initial and terminal states.
Finite-step samples approximate the learned early-stopped law.

`wrappers.HGuidance(h)` supplies `guidance_fn` by differentiating a pointwise
positive `h(state, time)` of shape `[batch, 1]`; `output_is_log=True` accepts log-H.
`QGuidance(h, q)` instead requires `q = grad h`. Optional orthonormal `B` uses
fixed features `state @ B`; this is not full-portfolio DFM conditioning.
Exact conditioning also requires the correct terminal-event H and initial law.

## Gaussian synthetic benchmark

The Gaussian notebook adds known population truth, validation-selected DFM and H
checkpoints, and comparisons of raw and event-filtered H-guided samples with
conditional truth and independent DFM rejection samples. Existing training,
preprocessing, author-backend, and guided-sampling tools remain available.

Use Python 3.12 and install the benchmark dependencies in your virtual environment:

```bash
python -m pip install -r requirements-synthetic.txt
# To exercise both the existing functionality and the benchmark:
python -m pip install -r requirements-equities.txt -r requirements-synthetic.txt
python -m pip check
```

Start Jupyter from the repository root, open
`synthetic_dfm_conditional_benchmark.ipynb`, and run all cells in order:

```bash
python -m notebook synthetic_dfm_conditional_benchmark.ipynb
```

After switching branches, reload the notebook from disk and restart its kernel.
The default `standard` configuration uses 100 assets, eight factors, one training
seed, 12,000/3,000/3,000 training/validation/evaluation observations, 1,000 DFM
epochs, and 400 sampling steps on CPU. Set `MODE` to `smoke`, `standard`,
`research`, or `large`, and edit `CONFIG_OVERRIDES` for dimensions, data sizes,
epochs, steps, architecture, optimizer, seeds, and device. The `smoke` preset uses
15 DFM epochs and 100 sampling steps. Research and large presets use three
unconditional training seeds; the H stage uses the first configured seed.

The final Gaussian call configures H separately: defaults are 200 epochs,
10,000/2,000/2,000 training/validation/test trajectories, and a portfolio event
at the population 5% quantile. It generates `CONFIG.n_eval` guided samples.
Edit that call to reduce H budgets, including when using `smoke`. Its
`ConditionalAssessmentConfig` requests 1,000 accepted samples from each of two
independent rejection streams, with a cap of 50,000 proposals per stream and
64 projection directions. The fitted DFM's event rate can differ from 5%; saved
diagnostics report proposal counts, failures, and shortfalls. Experiment budgets
are configured in the notebook, without environment-variable overrides.

Each execution writes a fresh timestamped directory under
`artifacts/synthetic_unconditional_benchmark/`. It contains configuration and
random-stream provenance, DFM/H checkpoints and validation curves, sample arrays,
metrics, rejection diagnostics, comparison figures, and `conclusions.md`.
The directory is ignored by Git. See
[the benchmark structure](docs/synthetic_benchmark_refactor.md) for the module layout.

Run the combined software checks with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
  python -m unittest discover -s tests -v
```

The notebook test executes every cell with reduced training and sampling budgets
and checks generated artifacts. Passing these software checks does not establish
research-quality convergence or conditional-law recovery. A full training run,
distribution diagnostics, and seed variability require separate assessment.
Optional author-backend tests require `requirements-dfm.txt`; otherwise they skip.

Mixture and market experiments remain on the separate
`codex/mixture-simulation` branch. The original Gaussian source is preserved on
`codex/market-rejection-target`; this integration imports only the Gaussian
benchmark and its H checkpoint-selection support.
