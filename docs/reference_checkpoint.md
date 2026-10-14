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

Train H for a fixed raw event, then save a bundle for the comparison runner:

```python
from pathlib import Path
from h_setup import train_h_from_checkpoint

Path("artifacts").mkdir(exist_ok=True)
h, metadata = train_h_from_checkpoint("unconditional_dfm.pt",
    weights={asset: 1 / len(assets) for asset in assets}, threshold=-.02,
    num_steps=128, train_trajectories=4096, held_out_trajectories=1024)
h.save_checkpoint("artifacts/h.pt", metadata=metadata)
```

Weights must name every checkpoint asset; mapping order does not matter.
The helper reuses saved scaling, trains only H, and rejects single-class training
labels. Independent held-out paths compare probability loss with the training
event-rate constant; this does not validate guidance gradients or market fit.
Metadata retains seeds, settings, source hash, losses and measured H initialization,
label generation and optimization costs. Validation/I/O/reference pretraining are
excluded. Use the saved bundle with the [comparison runner](guided_benchmark.md);
its generation costs include conditional initialization and rejected proposals.
