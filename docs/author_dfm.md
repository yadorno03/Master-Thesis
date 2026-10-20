# Author U-Net/DDPM backend and H extension

This backend calls the [authors' released implementation](https://github.com/xymmmm00/diffusion_factor_model/tree/99261df58dbc95315384ce34e3e4bb1a770a35fa)
at commit `99261df58dbc95315384ce34e3e4bb1a770a35fa`. Core and attention source
hashes are checked before import. The upstream project is MIT licensed; its
source is a pinned dependency, not a compressed rewrite of its U-Net.

The original `DFM.UnconditionalDFM`, `FrozenDFMReference`, and `HNetwork` remain
the factor-MLP/OU implementation. Their sampler now accepts `guidance_scale`:
one preserves existing drift, zero skips its correction, and larger values
strengthen it. Initial tilting is unchanged. QR projection now preserves signed
factor coordinates on an already orthonormal basis.

## Install and train on observed returns

```bash
python3 -m venv .venv-dfm
source .venv-dfm/bin/activate
python -m pip install -r requirements-dfm.txt
python run_author.py train --returns returns.npy --assets assets.json --output models/author
python run_author.py sample --dfm models/author/dfm.pt --output samples/returns.npy
```

`returns.npy` is a finite numeric N×d array with at least two rows.
`assets.json` lists unique column names in exactly that order. The default
image size is 32×64 (d=2048); `--height` and `--width` must multiply to d and
support the U-Net downsampling. The entry point preserves input column order,
like upstream `train.py`; supply any intended variance ordering consistently
with asset names and portfolio weights. No population moments are used.

Preprocessing matches the released training script: convert to float32,
subtract each training mean, divide by its sample standard deviation, and
reshape. Checkpoints retain the transform and asset order. Generated arrays
are inverse-transformed into original return units. Supply training rows only;
validation/test arrays reuse the saved transform without another fit.

Both pipelines now use [shared return preprocessing](return_preprocessing.md).
For a matched comparison in Python, fit `ReturnTransform` once and use
`NumpyStandardizer.from_transform(shared)` in the factor pipeline, then call
`train_author_dfm(training_returns, assets, output_dir, transform=shared)`.
The default fit preserves the released float32 recipe. A supplied float64
research fit retains its exact statistics; the U-Net input alone is cast to
float32. This alternative normalization is intentional and is recorded as
`preprocessing_dtype` in training metadata. Input asset order is validated.
See the [stack audit and merge procedure](dfm_stack_audit.md) for scope and
squash-merge instructions.

Defaults follow released config.py/train.py: base width 256; adaptive channel
multipliers; 200-step cosine diffusion with the upstream s=10; epsilon MSE;
AdamW lr=1e-4, betas=(.9,.99), decay=.01; 20-epoch warmup and 400-epoch cosine
phase; batch 32; 600 epochs; norm clipping at 1; EMA .999. The upstream Trainer
is invoked directly, including its scheduler and EMA update behavior. Final
outputs use ordinary final weights, as in upstream train.py, not EMA weights.
There is no additional-data checkpoint selection. Training also saves a final
portable checkpoint, even when the upstream periodic interval is not reached.

Sampling uses 8,192 outputs in batches of 128 by default. The default saved
index is 20: the state AFTER transition 20, not the clean index-zero state.
The discarded tail to zero is executed to preserve upstream random-number
advancement across batches. Bitwise parity is tested for successive batches.
Use `--save-index 0` during training only if intentionally changing the endpoint;
H always binds to the saved checkpoint's endpoint.

Explicit `--model-dim`, geometry, epoch and diffusion overrides are recorded.
Small test configurations are not evidence of reproduced paper performance.
The released d=2048 model has 1,588,285,441 parameters; FP32 weights, gradients,
Adam states and EMA alone require about 29.6 GiB, excluding activations.
The original training code uses CUDA when available, otherwise CPU.

## Train and use H on that DDPM

```bash
python run_author.py train-h --dfm models/author/dfm.pt --weights weights.json \
  --threshold -0.02 --trajectories 1024 --epochs 100 --output models/author_h.pt
python run_author.py guide --h models/author_h.pt --guidance-scale 1 \
  --output samples/guided.npy
```

`weights.json` lists raw-return portfolio weights in the checkpoint's asset
order. `--threshold` uses those original return units. Conversion preserves the
mean offset. H's checkpoint bundles its exact DFM weights, preprocessing,
endpoint, event, and network; an old factor-H checkpoint cannot be reused.

The U-Net has no explicit V/c factor map. `DDPMHNetwork` therefore learns
H(index,x) on the **full ambient state**, using preterminal DDPM trajectories
and the full terminal portfolio indicator. Brier regression gives each
transition equal weight and splits minibatches by whole trajectories. Only H
weights are updated. Trajectory data are bound to their exact in-memory
reference and event; endpoint rows are excluded from regression inputs.
Retained trajectory storage scales as N×J×d; `--batch-size` limits rollout and
training batch memory but does not remove the retained dataset's storage.
Training diagnostics report label counts and training loss, not held-out skill.

**H guidance is our extension, not a Xu-group algorithm.** It shifts a DDPM
transition mean by `guidance_scale * posterior_variance * grad_x log H`.
This is a first-order approximation to conditional transitions; even scale one
does not certify exact conditioning. It is not the continuous OU drift formula.
A bounded H-based initial rejection tilt is used by default; exhaustion raises.
No terminal filtering is performed, and the output metadata reports event rate.

Scale zero disables only the mean correction. For exact unguided equivalence,
use `--guidance-scale 0 --initialization gaussian`, the same seed and batching.
Other scales can change the conditional law; they do not guarantee the event.
Outputs and checkpoints refuse overwrite. Each generated array has JSON
metadata recording configuration, source revision, asset order and scale.

## Reproduction boundary and checks

The paper and released code differ: `latent_dim` is stored but does not control
the U-Net bottleneck, so no k parameter is pretended to enforce a factor count.
The paper's Appendix D describes centering; the released entry point also
standardizes. This stack follows the **pinned released numerical code**, with
these differences explicit. It does not certify the paper's tables or their
training seeds, nor alter the method to force an empirical-estimator victory.
The `released_config_defaults` metadata flag refers only to exposed settings,
not to a completed replication or AMP/hardware identity.

```bash
python -m unittest discover -s tests -p 'test_author*.py' -v
```

Tests execute real upstream layers, loss, training, checkpoint round trips and
DDPM transitions at small dimensions. They cover saved-time and RNG parity,
finite-difference H gradients, frozen-reference training, scaling, rejection
exhaustion and an end-to-end CLI workflow. The optional GitHub workflow installs
this backend and runs its tests; core tests still work without it. No notebook,
synthetic population generator, oracle, study runner, or study artifact is used.
