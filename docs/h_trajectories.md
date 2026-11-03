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

During training, H minimizes the mean squared error between its predicted
probability at each stored time and the trajectory's binary terminal label:

$$
\mathcal{L}(\theta)
= \frac{1}{N}\sum_{i=1}^{N}\sum_{j=0}^{J-1}
\alpha_j\left[H_\theta(t_j,Z_{i,j},U_{i,j})-y_i\right]^2,
\qquad
y_i=\mathbf{1}\!\left\{w^\top X^{(i)}_{T-t_0}\le C\right\}.
$$

Here, $N$ is the number of trajectories, $J$ is the number of stored times,
and $\alpha_j=1/J$ for the uniform grid above. The same label $y_i$ is used
at every stored time of trajectory $i$: it equals 1 if the final portfolio
condition is met, and 0 otherwise. Time weights already average over time;
there is no additional division by $J$.
