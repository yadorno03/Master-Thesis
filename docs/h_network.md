# H network and reduced conditioning dynamics

`HNetwork(reference, config, hidden_dims=(32, 32))` from `h_model` combines a
`FrozenDFMReference`, a portfolio event defined by `ConditioningConfig`, and a
neural network that estimates the probability of meeting that event at the final
time. It uses the reference model's numerical precision and device.
`ConditioningConfig` and raw-to-model event conversion are defined in
`conditioning.py`; see [portfolio events](portfolio_event.md).

## Reduced coordinates and dynamics

H works with reduced coordinates `(Z,U)`: `k` factor coordinates and one
portfolio coordinate. Time `t` runs in the generation direction from `0` to
`T-t0`; the corresponding forward score time is `s = T-t`.

- `features(t, x)` maps full states to row batches `y = (Z,U) = x @ G(t).T`
  of shape `[batch,k+1]`. `ell(t)` is the last row of `G(t)`.
- `mu(t, y)` gives the reduced drift, including the effect of the changing
  coordinate map: `dG/dt x + G b(t,x)`, where `b` is the full-state reference
  drift. Its linear Z term is `-Z/2`; the linear U term cancels. Different
  residual variances across assets are preserved.
- `A(t)` gives the reduced noise covariance `G(t) @ G(t).T`. It may be singular;
  no covariance inverse is required.
- `terminal_event(y)` checks `U <= C` at the final time `T-t0`, including equality.

These dynamics describe the reference process before guidance is added.

## Probability network and guidance

The constructor initializes a fully connected network with tanh hidden layers;
the default has two layers of 32 units. The input combines `(Z,U)` with time
normalized by `T-t0`. `model(t,y)` returns sigmoid probabilities of shape
`[batch,1]`. `model.log_prob(t,y)` computes log-probabilities directly from the
logits, preserving finite values and gradients even when the corresponding
floating-point probability rounds down to zero.

Only `model.network.parameters()` are trained; the reference remains frozen
and in evaluation mode. A newly constructed H is untrained. Training uses
[unguided trajectories and terminal labels](h_trajectories.md) with a weighted
squared-error loss; see [H training](h_training.md).

Input gradients remain available for guidance. In row-vector notation, the
correction is `grad_y log H @ G` in the full state space and
`grad_y log H @ A` in reduced coordinates. Exact conditioning also requires
the appropriate initial distribution, obtained by weighting the original
initial density by H; see [conditional initialization](conditional_initialization.md).

The smooth network output does not enforce the binary terminal condition.
Accurate probability predictions alone do not establish accurate gradients,
and learned H and finite time steps can produce guided samples that violate
the event; see [conditional sampling](conditional_sampling.md).
