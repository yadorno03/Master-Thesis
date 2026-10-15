# Full-portfolio H guidance

After `h, metadata = HNetwork.load_checkpoint(path)`, construct
`guide = DFMHGuidance(h)` from `guided_sampling`. `guide(x, t)` returns the
ambient correction `grad_y log H @ G_t`, including redundant coordinates.
Times are elapsed generative times strictly before `T-t0`. H must be pointwise
across samples. Calls preserve module modes and do not accumulate weight gradients.

The adapter works as `generate_samples(..., guidance_fn=guide)` with the attached
`h.reference`. Conditional initial states must also be supplied:
Gaussian initialization plus this drift generally does not give the conditional law.
The initialization helper follows in the next slice. Guidance uses H's Doob
correction directly; learned H and finite Euler steps still introduce approximation error.

Guidance and conditional initialization accept `HNetwork` instances, including
subclasses overriding `log_prob(t, y)` for analytic probability functions.
The adapter stores only `h_model` and uses its direct reference and coordinate methods.
