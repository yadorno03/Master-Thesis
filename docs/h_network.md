# Smooth terminal-event probability network

`HNetwork(conditioning, hidden_dims=(32, 32))` attaches a tanh MLP to the frozen
reference and full-portfolio event from `DFMConditioning`. It inherits reference
precision/device. Inputs are generative time `t` and reduced row batches `(Z,U)`
of shape `[batch,k+1]`; the network appends time normalized by `T-t0`.
`model(t,y)` returns sigmoid probabilities `[batch,1]`. `model.log_prob(t,y)`
uses log-sigmoid on logits, retaining finite values and gradients when the
floating-point probability underflows to zero. Input gradients remain available.

Optimize `model.network.parameters()` to train H; the attached reference stays
frozen and in evaluation mode. This PR adds the architecture, not a fitted H.
The smooth output does not enforce the discontinuous terminal indicator.
Unguided path collection and training with the existing weighted terminal-label
loss follow next. Probability fit alone will not establish gradient accuracy,
and conditional sampling will also require guidance and conditional initialization.
