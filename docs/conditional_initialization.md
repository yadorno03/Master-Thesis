# Conditional starts for full-portfolio guidance

Given a trained or loaded H bundle, initialize and sample in ambient space:

```python
from guided_sampling import DFMHGuidance, sample_conditional_initial
from sampling import generate_samples

x0, diagnostics = sample_conditional_initial(h, 1000, max_proposals=100000)
samples = generate_samples(h.conditioning.reference, len(x0), num_steps=100,
                           initial_state=x0, guidance_fn=DFMHGuidance(h))
```

Gaussian proposals are accepted with probability `H(0, G(0) x)`. This samples
its normalized learned tilt; it is the true conditional initial law only for
exact H. Budget exhaustion raises without substituting Gaussian starts.
Diagnostics count every proposal and acceptance, including batch overshoot.
Include this entire initialization call in generation timings. Terminal draws
remain unfiltered; learned H and finite-step integration can violate the event.
Independent reference-path rejection and matched runtime comparisons follow
in later slices.
