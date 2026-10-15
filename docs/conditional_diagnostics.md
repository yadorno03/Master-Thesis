# Conditional distribution diagnostics

Compare all guided draws with independent reference rejection in the same model
coordinates, asset order and frozen portfolio event:

```python
import torch
from guided_evaluation import conditional_diagnostics
from guided_sampling import generate_conditional_samples, sample_reference_conditional
from h_model import HNetwork

h, metadata = HNetwork.load_checkpoint("h.pt", device="cpu")
settings = dict(num_steps=100, batch_size=256, max_proposals=100000)
guided = generate_conditional_samples(h, 1000, **settings,
    generator=torch.Generator().manual_seed(2027))
reference = sample_reference_conditional(h, 1000, **settings,
    generator=torch.Generator().manual_seed(2028))
report = conditional_diagnostics(guided.samples.numpy(), reference.samples.numpy(),
    h.w.numpy(), h.C, seed=2029)
```

The JSON-compatible report includes both event rates, portfolio W1/KS distances,
mean bias and quantiles (1%, 5%, 50%, 95%, 99%), mean-vector/covariance RMSE, and
W1 averaged over seeded unit projections. Standard deviations and covariances use
the sample denominator N-1; covariance RMSE includes every matrix entry. Distances
depend on model-coordinate scaling. Neither input is filtered or subsampled.

Use fresh random streams and fix the event/model before evaluation. Rejection
targets the finite-grid conditional reference, not historical conditional returns.
Finite samples and projections limit these diagnostics; event satisfaction alone
does not establish distributional accuracy. Measure generation separately from
diagnostics, including initialization and rejected proposals; report H setup apart.
