# Ambient guided samples

After loading an H bundle, generate full-portfolio guided draws with:

```python
import torch
from h_model import HNetwork
from guided_sampling import generate_conditional_samples

h, metadata = HNetwork.load_checkpoint("h.pt", device="cpu")
result = generate_conditional_samples(h, 1000, num_steps=100,
    generator=torch.Generator().manual_seed(2027))
```

The default initializes from the learned-H tilt; approximation and Euler errors
remain. `samples` retains every draw, including event violations. Diagnostics
report the full-sample event rate, initialization proposals and synchronized
initialization/integration times. Include both stages in generation cost; report
H training/setup separately. For comparisons, follow the matched settings and
independent random streams in [reference rejection](reference_rejection.md).
