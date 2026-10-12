# Fixed full-portfolio event

`ConditioningConfig(w, C)` fixes the event `w @ X_terminal <= C` in model
return coordinates. It owns an immutable tuple of finite, nonzero portfolio
weights and a finite scalar threshold. Individual weights may be zero or negative;
there is no sum-to-one constraint or factor projection.

For raw returns `r = mean + std * x`, use the saved training preprocessing:

```python
from conditioning import portfolio_event_in_model_coordinates

event = portfolio_event_in_model_coordinates(
    weights=[0.5, -1.0], threshold=0.0,
    mean=[0.25, -0.125], std=[0.5, 2.0],
)
assert event.w == (0.25, -2.0)
assert event.C == -0.25
```

The conversion uses `w_model = std * w_raw` and
`C_model = C_raw - w_raw @ mean`, preserving the complete portfolio event,
including its mean offset. Inputs must follow the checkpoint's asset order;
this helper validates dimensions and values but cannot verify asset identities.
It neither fits preprocessing nor uses held-out observations.

This slice is independent of the frozen reference and coordinate map in PRs #5–#6.
Binding the event to that reference and adding reduced dynamics comes next.
Event configuration alone does not generate conditional samples.
