# H-only trajectory regression

`train_h(model, data, epochs=100, batch_size=64, lr=1e-3, generator=None)`
optimizes only `model.network` with Adam and the existing time-weighted squared loss.
It shuffles whole trajectories, retains partial batches and returns epoch losses
weighted by trajectory count. These record predictions before each minibatch update.
The shuffle generator must match the data device; the reference must remain frozen
with no parameter gradients. Original module modes are restored, including on failure.

Data must be detached, finite and in reference precision/device, with nonempty
`[N,J,k+1]` features, binary `[N,1]` labels, increasing `[J]` times starting at zero
and excluding `T-t0`, and positive weights equal to successive time intervals divided
by `T-t0`. Nonuniform valid grids are supported. Invalid data fail before optimization.
These checks cannot establish that data came from this exact reference and event;
the caller must preserve that provenance and split data by whole trajectories.
Held-out evaluation and its training-label constant baseline follow next.
Training loss alone establishes neither probability generalization nor guidance-gradient accuracy.
