# H-only trajectory regression

`train_h(model, data, epochs=100, batch_size=64, lr=1e-3, generator=None)`
optimizes only `model.network` with Adam and the existing time-weighted squared loss.
It shuffles whole trajectories, retains partial batches and returns epoch losses
weighted by trajectory count. These record predictions before each minibatch update.
The shuffle generator must match the data device; the reference must remain frozen
with no parameter gradients. Original module modes are restored, including on failure.
