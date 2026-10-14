# Smooth terminal-event probability network

`HNetwork(reference, config, hidden_dims=(32, 32))` binds a `FrozenDFMReference`
and `ConditioningConfig` directly. It owns the tanh MLP, portfolio weights,
coordinate methods and reduced dynamics, and inherits reference
precision/device. Inputs are generative time `t` and reduced row batches `(Z,U)`
of shape `[batch,k+1]`; the network appends time normalized by `T-t0`.
`model(t,y)` returns sigmoid probabilities `[batch,1]`. `model.log_prob(t,y)`
uses log-sigmoid on logits, retaining finite values and gradients when the
floating-point probability underflows to zero. Input gradients remain available.

Optimize `model.network.parameters()` to train H; the attached reference stays
frozen and in evaluation mode. Probability evaluation uses the architecture initialized in the preceding slice;
this is not a fitted H.
The smooth output does not enforce the discontinuous terminal indicator.
Unguided path collection and training with the existing weighted terminal-label
loss follow next. Probability fit alone will not establish gradient accuracy,
and conditional sampling will also require guidance and conditional initialization.
