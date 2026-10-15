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

Metadata stores experiment context: asset order, training-fitted preprocessing,
seeds and numerical settings. Pass a Python dictionary with JSON-compatible
values; no JSON file or encoded JSON string is needed. Loading returns this
context alongside the restored model. The metadata values are caller-supplied;
the checkpoint API does not infer or verify their meaning.
Optimizer state and RNG state are not saved. Loading
preserves the caller's CPU RNG state and uses PyTorch's weights-only loader.
The format supports the standard DFM Linear/ReLU architecture. A successful
roundtrip establishes persistence, not H gradient accuracy or conditional fit.

The version-1 bundle format is unchanged: it stores `reference["state_dict"]`
for the score network and `h["state_dict"]` for the H MLP separately, plus the
immutable event. Existing bundles load directly with the new constructor;
there is no format bump or key migration for `save_checkpoint` artifacts.

A raw full-model `state_dict()` is a different format: its old
`conditioning.reference.*` keys are now `reference.*`, and `conditioning._w`
is now `_w`; `network.*` is unchanged. Such raw dictionaries require removing
the leading `conditioning.` from those keys before strict `load_state_dict`.
They do not store architecture, event threshold or horizon, so construct the
matching `HNetwork(reference, config, hidden_dims=...)` first. No raw full-model
artifacts were found in the inspected repository or local artifact search.
Whole-object pickles are not a supported persistence format; use the bundle API.
