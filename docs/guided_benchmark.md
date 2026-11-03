# Guided versus rejection benchmark

`benchmark_guidance` compares the speed and sample quality of two methods:
**rejection** generates scenarios without guidance and keeps those satisfying
the portfolio condition; **guided sampling** uses a trained H network to steer
generation toward that condition. Rejection can be slow when the condition is rare.

Load a saved H model and run the comparison:

```python
from h_model import HNetwork
from guided_benchmark import benchmark_guidance

h, metadata = HNetwork.load_checkpoint("h.pt", device="cpu")
report, samples = benchmark_guidance(h, num_samples=512, num_steps=128,
    validation_samples=2048, repeats=5, batch_size=128, setup_seconds=None)
```

This requests 512 samples per method, using 128 time steps and batches of 128,
and repeats the comparison five times. Another 2,048 independent rejection
samples provide a reference for checking quality.

Both methods share the same numerical settings and proposal limit, using one
CPU thread. The benchmark runs a warm-up, alternates execution order and uses
separate random seeds. Previous thread limits are restored afterward.

`report["rows"]` contains timings, quality measures and sampler diagnostics for
each run. Generation time includes initialization, rejected proposals and surplus
samples, but excludes warm-up, validation, metrics, loading and file operations.
H training and preparation costs are recorded separately through `setup_seconds`;
keep `None` if unknown. `samples` retains every draw, including guided samples
that fail the condition.

**Compare speed and quality together.** Satisfying the condition does not ensure
the correct distribution: samples could miss extreme losses while all meeting
the threshold. Rejection represents the frozen model at the chosen time steps,
not market reality. Timings depend on the hardware and settings used.
