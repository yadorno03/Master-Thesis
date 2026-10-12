# H checkpoints

Persist H before using it in guided sampling:

```python
model.save_checkpoint("h.pt", metadata={"seed": 27, "steps": 100})
restored, metadata = HNetwork.load_checkpoint("h.pt", device="cpu")
```

Import `HNetwork` from `h_model`. The bundle fixes H weights and architecture,
the frozen DFM weights and architecture, forward horizon `T`, cutoff `t0`, and
the full event `w @ X_(T-t0) <= C` in model coordinates. Loading restores saved
float32/float64 precision, sets evaluation mode, and keeps reference parameters
frozen while permitting input gradients. H parameters remain trainable.

Optional metadata must be JSON-compatible. Record asset order, training-fitted
preprocessing, seeds and numerical settings there; these are caller-supplied,
not inferred or verified. Optimizer state and RNG state are not saved. Loading
preserves the caller's CPU RNG state and uses PyTorch's weights-only loader.
The format supports the standard DFM Linear/ReLU architecture. A successful
roundtrip establishes persistence, not H gradient accuracy or conditional fit.
