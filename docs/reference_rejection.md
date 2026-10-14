# Independent reference rejection

After loading an H bundle, use its frozen reference, full portfolio event and
horizon to sample the finite-step conditional reference:

```python
import torch
from h_model import HNetwork
from guided_sampling import sample_reference_conditional

h, metadata = HNetwork.load_checkpoint("h.pt", device="cpu")
result = sample_reference_conditional(
    h, 1000, num_steps=100, batch_size=1024, max_proposals=100000,
    generator=torch.Generator(device=h.w.device).manual_seed(2026))
```

`samples` and `initial_states` contain paired accepted terminal draws and starts.
No H probabilities or guidance enter rejection. Exhausting the finite proposal
budget raises. Diagnostics count all proposals and acceptances, including final
batch overshoot; `returned` is the requested count.

For comparisons, use separate random streams for H training, guided sampling and
reference rejection. Match the reference, event, horizon, Euler steps, dtype,
device and returned sample count. Time the whole call, including Gaussian starts
and rejected paths (synchronize accelerators); report H setup/training separately.
Compare **all** guided draws, including event violations, to these reference draws.
This targets the finite-grid frozen reference, not the exact diffusion or market law.
