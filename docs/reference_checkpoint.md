# Equity reference checkpoints

`load_reference_checkpoint` reuses the equity notebook's saved DFM on CPU:

```python
from reference_checkpoint import load_reference_checkpoint
from conditioning import portfolio_event_in_model_coordinates
from h_model import HNetwork

ref, assets, mean, std = load_reference_checkpoint("unconditional_dfm.pt")
weights = [1 / len(assets)] * len(assets)
event = portfolio_event_in_model_coordinates(weights, -.02, mean, std)
h = HNetwork(ref, event)
```

The checkpoint must contain `model_config` (`d`, `k`, `sigma_max`, `hidden_dims`,
`T`, `t0`), the full DFM `state_dict`, ordered unique `assets`, and `preprocessing`
with one finite `mean` and positive finite `std` per asset. Float32/float64 weights
must be finite and share one precision. Missing/extra weights, incompatible
shapes, invalid horizons and nonorthonormal loadings fail without repair.

Use this exact asset order for raw portfolio weights and sample columns.
Raw returns are `mean + std * model_returns`; the event conversion retains the
mean offset and full portfolio. The loader preserves saved scaling and never
refits it; the checkpoint producer must establish that it was fitted on training
data only. Loading neither retrains the reference nor advances the CPU RNG.
H training and its setup-cost measurements remain a subsequent runner step.
