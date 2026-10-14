# Unguided training trajectories for H

`model.sample_trajectories(N, num_steps=J, generator=rng, batch_size=None)`
collects Gaussian-start Euler–Maruyama paths from the attached frozen reference.
It never evaluates the learned H or adds guidance. Labels use the full terminal
portfolio event, including equality: `w @ X_(T-t0) <= C`.

The returned `TerminalLabelData` contains detached tensors in reference precision/device:

- `times`: `[J]`, from zero through `(J-1)*(T-t0)/J`.
- `features`: `[N,J,k+1]`, with the time-dependent `(Z,U)` map at each stored time.
- `labels`: `[N,1]`, one binary terminal label per whole trajectory.
- `weights`: `[J]`, each equal to `1/J`, for the existing time-integrated regression loss.

The terminal endpoint supplies labels only; it is excluded from regression features.
Split train/validation data by whole trajectories, never by individual time rows.
Use independent random streams for held-out paths. Collection does not train H.
`batch_size` bounds the number of ambient paths generated at once and retains
partial final batches. All reduced features are still retained in memory.
A seed reproduces a fixed batching setup; changing batching may change draws.
These data describe the finite-grid reference, not an exact continuous-time law.
Reference-aware validation and H optimization are the next dependency stage.
