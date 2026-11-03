# Frozen reference

`FrozenDFMReference(model)` copies the trained score and fixes `T` and `t0`; input gradients remain available.
`score(x, s)` takes forward OU time `s` in `[t0, T]`; reverse generation uses `s = T - t` and stops at `t = T - t0`.

`G(t, w)` uses generative time `t` in `[0, T-t0]` and a nonzero portfolio `w`
in the model's return coordinates. With frozen loadings `B`, variances `c`, and `s = T-t`:

- `alpha_s = exp(-s/2)` and `Lambda_s = (1-exp(-s)) I + exp(-s) diag(c)`.
- `D_s = inverse(Lambda_s)` and `ell_t = (alpha_s/alpha_t0) D_s Lambda_t0 w`.
- `G_t = [B.T D_s; ell_t.T]`; row batches transform as `y = x @ G_t.T`.

A scalar time returns a matrix `G(t, w)` of dimension `[k+1, d]`; `[batch]` or `[batch, 1]` times return `[batch, k+1, d]`.
State and portfolio tensors must match the frozen reference's dtype and device.
At `tau = T-t0`, `ell_tau = w`, so the last coordinate is the complete portfolio
`w.T X_tau`, including its residual component. Earlier, that coordinate generally differs from `w.T X_t`.
The factor coordinates are `B.T D_s X_t`; fixed `B.T X_t` does not generally have closed dynamics.
`G_t` need not be orthonormal or full rank. Using the map requires no inverse of `G_t` or `G_t G_t.T`.
Input and time derivatives remain available while reference parameters stay frozen.
