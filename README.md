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
