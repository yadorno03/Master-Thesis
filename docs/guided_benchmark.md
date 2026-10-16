# Guided versus rejection benchmark

```python
from h_model import HNetwork
from guided_benchmark import benchmark_guidance

h, metadata = HNetwork.load_checkpoint("h.pt", device="cpu")
report, samples = benchmark_guidance(h, num_samples=512, num_steps=128,
    validation_samples=2048, repeats=5, batch_size=128, setup_seconds=None)
```

The benchmark warms both methods, alternates their order, and uses separate seeds
for every run and a third rejection sample.
Both methods use the same output count, grid, precision, batch size and proposal
budget on one PyTorch/BLAS CPU thread. Previous thread limits are restored.
Each row reports generation seconds, all-draw accuracy and original sampler
diagnostics, including initialization timing and rejected/overshoot proposals.
Generation excludes warm-up, validation, metrics, loading and file I/O.
Pass separately measured H rollout/training/setup seconds; unknown reused-H cost
stays `None`. Retained arrays allow inspection of every draw, including violations.
Compare timing and accuracy together: event satisfaction is not conditional-law
accuracy, and faster generation does not imply equivalent quality. Rejection
targets the finite-grid frozen reference. Timings are local to hardware/settings.
