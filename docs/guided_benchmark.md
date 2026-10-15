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

Run the same comparison from a saved [H bundle](h_checkpoint.md):

```bash
python scripts/run_guided_dfm.py --h-checkpoint artifacts/h.pt \
  --output artifacts/comparison-01 --samples 512 --steps 128 \
  --timing-repeats 5 --timing-reference-samples 2048
```

The runner loads on CPU in saved precision; the bundle fixes the reference,
portfolio event and horizon. It uses learned-H conditional initialization and
ambient guidance. It neither trains H nor changes the event. `--steps` controls both
samplers; it need not match the H training grid, so check grid sensitivity.
`timing_benchmark.json` contains generation costs, proposal diagnostics and
all-draw accuracy. `timing_samples.npz` retains every repeat and the independent
validation sample in model coordinates. `run_summary.json` records the checkpoint
hash, event, metadata, versions, artifact hashes and completion/failure status.
Optional checkpoint metadata `h_setup_cost` must contain `total_seconds` for
measured label-generation/training/setup cost; retain its measurement scope there.
Missing setup metadata stays unknown (`null`), never zero or a newly measured cost.
Existing output directories are refused. Failed runs raise and record an error;
use a new directory to retry. Generated reports and draws belong in `artifacts/`.
