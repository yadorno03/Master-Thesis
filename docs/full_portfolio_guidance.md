# Full-portfolio H guidance

After `h, metadata = HNetwork.load_checkpoint(path)`, construct
`guide = DFMHGuidance(h)` from `guided_sampling`. `guide(x, t)` returns the
ambient correction `grad_y log H @ G_t`; `guide.reduced(t, y)` returns
`grad_y log H @ (G_t @ G_t.T)`, including singular coordinate covariances.
Times are elapsed generative times strictly before `T-t0`. H must be pointwise
across samples. Calls preserve module modes and do not accumulate weight gradients.

The adapter works as `generate_samples(..., guidance_fn=guide)` with the attached
`h.conditioning.reference`. Conditional initial states must also be supplied:
Gaussian initialization plus this drift generally does not give the conditional law.
The initialization helper follows in the next slice. Only scale 1 is H's Doob
correction; learned H and finite Euler steps still introduce approximation error.
