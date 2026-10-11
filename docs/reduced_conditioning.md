# Reduced conditioning dynamics

`DFMConditioning(reference, config)` binds the frozen score to the full portfolio event.
All times are generative: `t in [0, T-t0]`, with forward score time `s = T-t`.
`features(t, x)` returns row batches `(Z,U) = x @ G(t).T`; `ell(t)` is the last row of `G(t)`.
`mu(t, y)` includes the derivative of the time-dependent map: `dG/dt x + G b(t,x)`.
Its linear Z drift is `-Z/2`; the linear U drift cancels. Heterogeneous residual variances remain intact.
`A(t)` is `G(t) @ G(t).T`, including singular covariance; no inverse is required.
`terminal_event(y)` applies the inclusive full-portfolio threshold `U <= C` at `T-t0`.
These coefficients describe the unguided reference. H-network construction and path collection come next.
Later ambient guidance is `grad_y log H @ G`; reduced guidance is `grad_y log H @ A`.
Exact conditioning also requires the H-tilted initial law; these coefficients alone do not sample it.
