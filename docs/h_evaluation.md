# Held-out H probability evaluation

`evaluate_h(model, data, constant_probability=p, batch_size=256)` returns Python
floats `h_loss` and `constant_loss`. Both use the time-integrated squared error
from the trajectory regression objective, averaged over whole trajectories.
Partial final batches are weighted by trajectory count. Nonuniform time grids
use their interval weights, without an extra division by the number of times.

Fit `p` from training labels only and pass it explicitly. The evaluator does not
re-estimate the baseline from held-out labels. For example, after creating `model`:

```python
from h_model import train_h, evaluate_h
import torch

training = model.sample_trajectories(128, 32, generator=torch.Generator().manual_seed(11))
heldout = model.sample_trajectories(64, 32, generator=torch.Generator().manual_seed(12))
p = float(training.labels.mean())
train_h(model, training, epochs=20, generator=torch.Generator().manual_seed(13))
metrics = evaluate_h(model, heldout, constant_probability=p, batch_size=32)
```

This example uses CPU generators. On another device, generators must match it.
Evaluation validates the trajectory contract and runs without autograd. It does
not change module modes, weights or existing gradients for the supplied HNetwork.
The caller must preserve reference/event identity and independent whole-trajectory
training/evaluation splits; tensor validation cannot establish that provenance.
A lower held-out probability loss does not establish accurate spatial/log-probability
gradients or conditional samples. Checkpoint persistence is the next dependency stage.
