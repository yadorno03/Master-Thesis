# Reduced conditioning dynamics

`HNetwork(reference, config, hidden_dims=(32, 32))` from `h_model` binds the frozen score to the full portfolio event.
All times are generative: `t in [0, T-t0]`, with forward score time `s = T-t`.
`features(t, x)` returns row batches `(Z,U) = x @ G(t).T`; `ell(t)` is the last row of `G(t)`.
`mu(t, y)` includes the derivative of the time-dependent map: `dG/dt x + G b(t,x)`.
Its linear Z drift is `-Z/2`; the linear U drift cancels. Heterogeneous residual variances remain intact.
`A(t)` is `G(t) @ G(t).T`, including singular covariance; no inverse is required.
`terminal_event(y)` applies the inclusive full-portfolio threshold `U <= C` at `T-t0`.
The constructor also initializes the tanh MLP in reference precision/device.
Only `h.network.parameters()` are trainable; `h.reference` stays frozen.
`ConditioningConfig` and raw-to-model event conversion remain in `conditioning.py`.
These coefficients describe the unguided reference. Probability evaluation and path collection come next.
Later ambient guidance is `grad_y log H @ G`; reduced guidance is `grad_y log H @ A`.
Exact conditioning also requires the H-tilted initial law; these coefficients alone do not sample it.
