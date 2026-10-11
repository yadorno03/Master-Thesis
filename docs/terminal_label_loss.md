# Terminal-label regression objective

`h_model.py` introduces `TerminalLabelData` and `terminal_label_loss` for the
upcoming H-network regression on unguided reference paths. Let `tau = T-t0`.
Each path receives one binary label for the full terminal portfolio event.
Its regression inputs use generative times `0 = t[0] < ... < t[J-1] < tau`;
the terminal state supplies the label and is excluded from prediction inputs.

The data container holds `times [J]`, reduced `features [N,J,k+1]`,
`labels [N,1]`, and `weights [J]`. Use whole trajectories for train/validation
splits so time points from the same path cannot leak between those sets.

For probability predictions `p [N,J,1]`, the objective is

```text
weight[j] = (t[j+1] - t[j]) / tau, with t[J] = tau
loss = sum_i sum_j weight[j] * (p[i,j,0] - label[i,0])**2 / N
```

Time weights sum to one; there is no additional division by `J`, terminal-time
term, or class balancing. When aggregating minibatch losses, weight each by its
trajectory count, including the final partial batch. The loss retains autograd.

This low-level helper checks shapes. Callers must supply nonempty finite tensors
with compatible precision/device, binary labels, probabilities in `[0,1]`, and
positive time weights derived from the reference horizon. The data container
does not validate these properties; reference-aware validation comes with training.

This independent slice can run on PR #2. Frozen-reference/event binding, reduced
dynamics, H-network construction, path collection, and optimization follow in
later slices. A low probability loss alone does not establish the accuracy of
spatial gradients or conditional samples.
