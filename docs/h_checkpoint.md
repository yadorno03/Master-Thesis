# H checkpoints

## Saving and loading H checkpoints

A checkpoint stores a trained H model for reuse in guided sampling
without additional training:

```python
from h_model import HNetwork

model.save_checkpoint("h.pt", metadata={"seed": 27, "steps": 100})
restored, metadata = HNetwork.load_checkpoint("h.pt", device="cpu")
```

Here, `model` is a trained `HNetwork`. The saved file is a bundle containing
both H and its frozen DFM reference model, including their network structures
and learned weights. It also saves the time settings `T` and `t0`, and the
portfolio condition `w @ X_(T-t0) <= C`: the final portfolio value must be at
most `C`, using the model's coordinates.

Loading restores the saved float32 or float64 precision and sets evaluation
mode. The reference weights stay fixed, but gradients with respect to inputs
remain available for guidance. H's own weights remain trainable.

`metadata` records experiment details such as asset order, preprocessing fitted
during training, random seeds and numerical settings. Supply a Python dictionary
with JSON-compatible values, as above. Loading returns it alongside the model.
These details must be supplied explicitly; the checkpoint API does not check
their meaning.
