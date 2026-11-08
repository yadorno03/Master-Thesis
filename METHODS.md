# Methods

Technical details for the [benchmark notebook](synthetic_dfm_conditional_benchmark.ipynb): definitions, derivations, training recipes, numerical safeguards, and the code behind each step. The notebook gives the motivation and the results; sections 1–6 below carry the titles of its sections 1–6. Setup and the codebase are described in the [README](README.md).

## Notation and sources

Returns are $R\in\mathbb R^d$ with $k\ll d$ factors. For a nonzero portfolio $w\in\mathbb R^d$ and a threshold $C$, the event is

$$
E=\{w^\top R\le C\},
$$

and the target is the joint law of $R$ given $E$.

Forward noising time is $s\in[t_0,T]$ and generation time is $t\in[0,\tau]$, with $\tau=T-t_0$ and $s=T-t$. The sampler takes $J$ Euler steps of size $\Delta t=\tau/J$ at times $t_j=j\Delta t$; section 5 of the notebook writes $N$ and $dt$ for $J$ and $\Delta t$. $X_t$ is the ambient generated state and $Y_t=G_tX_t$ the reduced conditioning state. Population parameters are $B,D,S$. Fitted DFM parameters are $V,c$ and, for the Gaussian component, a factor prior $f\sim N(m,S_f)$, written $F\sim N(m,S)$ in section 2 of the notebook. In tensor shapes, $N$ or $n$ counts samples or trajectories.

- [Chen et al. (2025)](https://arxiv.org/abs/2504.06566v5): the OU forward process, the factor-aware score form, denoising score matching, and the reverse sampler. Their experiments use a U-Net; **DFM (MLP)** implements the score form directly with a small MLP.
- [Guo et al. (2026)](https://arxiv.org/abs/2602.05533v2): the terminal-event probability $H$, its Doob guidance and guidance scale, and the CDG-ML probability regression. Their covariation loss (CDG-MCL) is not implemented; `QGuidance` in [wrappers.py](wrappers.py) is only an inference adapter.
- Project adaptations: the market-calibrated population, **DFM (Gaussian + MLP)**, the $k+1$ reduced coordinates, the five refinements of **Refined CDG-ML**, the exact final step, and the evaluation design.

## 1. Construct the synthetic population

### 1.1 Population choices

The current notebook uses the Gaussian factor population of Chen et al. (2025),
with $d=500,k=8$ and this project's fixed population seed 912. Draw
$\mu_{F,j}\sim\operatorname{Unif}(0,0.1)$, set $\sigma_{F,j}=1.5\mu_{F,j}$,
draw $\beta_{ij}\sim N(0,1)$ and $\sigma_{\varepsilon,i}\sim\operatorname{Unif}(0,0.4)$,
then generate $R=\beta F+\varepsilon$, with independent Gaussian factors
$F\sim N(\mu_F,\operatorname{diag}(\sigma_F^2))$ and residuals
$\varepsilon\sim N(0,\operatorname{diag}(\sigma_\varepsilon^2))$.
The code stores the equivalent QR coordinates $\beta=BA$,
$m=A\mu_F$, $S=A\operatorname{diag}(\sigma_F^2)A^\top$ and $D_i=\sigma_{\varepsilon,i}^2$.
Here $S$ is generally full. The notebook keeps raw returns and the previous
training and sampling budgets; it does not reproduce the paper's U-Net preprocessing.

The alternative `population="market"` calibrates a Gaussian population as follows.

Draw the cohort once with seed 500 and standardize each asset by its training-period mean and sample standard deviation. Let $z$ be the standardized returns, $\widehat\Sigma_z$ their covariance (the sample correlation matrix), $B=[b_1,\ldots,b_k]$ its leading orthonormal eigenvectors, and $\lambda_j$ the matching eigenvalues. Then

$$
D_i=\widehat{\operatorname{Var}}\bigl(z_i-(BB^\top z)_i\bigr),\qquad
S=\operatorname{diag}(S_{11},\ldots,S_{kk}),\quad
S_{jj}=\lambda_j-\sum_iB_{ij}^2D_i,
$$

$$
R=BF+\varepsilon,\quad F\sim N(0,S),\quad
\varepsilon\sim N(0,\operatorname{diag}D),\quad F\perp\varepsilon,
\qquad \mu=0,\quad \Sigma=BSB^\top+\operatorname{diag}D.
$$

Calibration raises an error if any $D_i$ or $S_{jj}$ is nonpositive. Since $B^\top b_j$ is the $j$-th unit vector, $b_j^\top\Sigma b_j=S_{jj}+\sum_iB_{ij}^2D_i=\lambda_j$. With heterogeneous $D_i$, the other eigenvalues and the eigenbasis of $\Sigma$ need not match those of $\widehat\Sigma_z$.

Cohort identity, panel checksum, dimensions, and seeds identify market checkpoints.
Chen checkpoints instead identify the dimensions, population seed and QR sign convention.

### 1.2 Sampling code

`m[0]` and `S[0]` store the factor mean and covariance. `sample_returns` returns the factors associated with stored $B$. `sample` uses factor-based Gaussian draws and an exact conditional correction when requested (section 1.3); the notebook uses `sample`, drawing through the factor construction to avoid a dense covariance decomposition.

| Formula / operation | Function in [synthetic_conditional_truth.py](synthetic_conditional_truth.py) | Inputs → outputs |
| --- | --- | --- |
| Fix $B,D,m,S$ (Chen, default) | `make_population(d, k=k, seed=912)` | Dimensions and population seed → `FactorPopulation` |
| Fix $B,D,m,S$ (market option) | `make_market_population(panel, d, k, cohort_seed=500)` | Panel and dimensions → `FactorPopulation` |
| Draw $R\sim N(\mu,\Sigma)$ | `population.sample(n, rng)` | Count and NumPy generator → float64 `[n,d]` |
| Draw explicit $F,\varepsilon$ | `population.sample_returns(n, rng)` | Count and generator → returns `[n,d]`, factors `[n,k]`, regime labels `[n]` |
| Compute $\mu,\Sigma$ | `population.moments()` | No arguments → `[d]`, `[d,d]` |
| Choose $C_q$ with $P(w^\top R\le C_q)=q$ | `population.threshold(q)` | Tail probability → scalar threshold |

### 1.3 Exact conditional law

For a Gaussian population $R\sim N(\mu,\Sigma)$, let

$$
a=w^\top\mu,\quad v=w^\top\Sigma w,\quad
z_C=\frac{C-a}{\sqrt v},\quad \lambda_C=\frac{\phi(z_C)}{\Phi(z_C)}.
$$

Then $P(E)=\Phi(z_C)$, the $q$-quantile threshold is $C_q=a+\sqrt v\,\Phi^{-1}(q)$, and the exact conditional moments are

$$
\mathbb E[R\mid E]=\mu-\frac{\Sigma w}{\sqrt v}\lambda_C,
\qquad
\operatorname{Cov}(R\mid E)=\Sigma-\frac{(\Sigma w)(\Sigma w)^\top}{v}\lambda_C(\lambda_C+z_C).
$$

`population.moments(c=C)` returns these exact moments. The conditional law is a truncated multivariate Gaussian, not an ordinary Gaussian with these moments. `population.sample(N, rng, c=C)` draws it by replacing an unconditional Gaussian draw's portfolio coordinate with an independent truncated-normal draw, while preserving the independent Gaussian regression residual:

$$
R^E=R+\frac{\Sigma w}{v}\bigl(P-w^\top R\bigr),\qquad
P\sim N(a,v)\ \text{truncated above at } C.
$$

This is exact because the residual of $R$ after regression on $w^\top R$ is Gaussian and independent of $w^\top R$. `population.cdf(p,c=C)` is the exact conditional portfolio CDF, $\Phi\bigl((\min(x,C)-a)/\sqrt v\bigr)/\Phi(z_C)$ at portfolio value $x$.

## 2. Train and sample the unconditional DFM

### 2.1 Forward process and score targets

The forward OU corruption is

$$
R_s=\alpha_sR+\sqrt{h_s}\,\xi,\qquad
\alpha_s=e^{-s/2},\quad h_s=1-e^{-s},\quad \xi\sim N(0,I_d).
$$

[`UnconditionalDFM.q_sample`](DFM.py) maps returns `[N,d]`, times `[N,1]`, and noise `[N,d]` to corrupted returns `[N,d]`.

If $R\sim N(\mu,\Sigma_0)$, then $R_s\sim N(\alpha_s\mu,\Sigma_s)$ with $\Sigma_s=\alpha_s^2\Sigma_0+h_sI_d$, and the analytic OU score is

$$
\nabla_r\log p_s(r)=-\Sigma_s^{-1}(r-\alpha_s\mu).
$$

It is exact for that Gaussian law, which may only approximate the data. Training uses instead the score of the corruption kernel for a fixed clean $R$,

$$
\nabla_r\log p_s(r\mid R)=-\frac{r-\alpha_sR}{h_s}=-\frac{\xi}{\sqrt{h_s}},
$$

whose conditional expectation given $R_s=r$ is the marginal score. Regression on these noisy targets therefore learns the marginal score without evaluating the marginal density. A non-Gaussian population generally has non-Gaussian marginals at finite $s$.

### 2.2 Score networks

Both networks use

$$
D_s=\operatorname{diag}(h_s+\alpha_s^2c_i)^{-1}=\Lambda_s^{-1},\qquad
z=V^\top D_sr,\qquad
s_\theta(r,s)=\alpha_sD_sVg(z,s)-D_sr,
$$

with orthonormal $V\in\mathbb R^{d\times k}$ and $c_i=\sigma_{\max}\operatorname{sigmoid}(c_{\rm raw,i})$, where the benchmark sets $\sigma_{\max}=2$. Diagonal matrices are applied as vectors, and $g$ acts on $k$ coordinates before the lift to $d$ dimensions. The MLP maps `[N,k+1]` to `[N,k]`; the complete score maps `[N,d]` and `[N,1]` to `[N,d]`.

**DFM (MLP)** ([`FactorScoreNetwork`](NeuralNets.py), configuration key `mlp`) sets $g=g_\zeta$, a two-hidden-layer ReLU MLP of $(z,s)$.

**DFM (Gaussian + MLP)** ([`GaussianMLPScoreNetwork`](NeuralNets.py), key `gaussian_mlp`) sets $g=g_{\mathrm G}+g_\zeta$. For the factor model $f\sim N(m,S_f)$, $R=Vf+\varepsilon$, $\varepsilon\sim N(0,\operatorname{diag}c)$, and $M_s=V^\top D_sV$, completing the square in the density of $f$ given $R_s$ gives the posterior mean

$$
g_{\mathrm G}(z,s)=m+(I_k+\alpha_s^2S_fM_s)^{-1}S_f\alpha_s(z-\alpha_sM_sm).
$$

The code solves this $k\times k$ system instead of forming an inverse. `prior_mean` stores $m$, and the lower triangle $L$ of `prior_chol` gives $S_f=LL^\top$. [`GaussianMLPScoreNetwork`](NeuralNets.py) initializes $m,S_f$ from the training returns and initializes the MLP's last layer at zero, so the initial score is the analytic OU score of the initial Gaussian factor model. The clean mean of this model is $Vm$.

### 2.3 Training (refinements 1 and 2)

**Initialization and optimization (refinement 1).** From the covariance of centered training observations, take the leading $k$ eigenvectors as $V_0$. We initialize the idiosyncratic variances from the variation left unexplained by these principal components, clipped to $[10^{-4},\,0.95\,\sigma_{\max}]$. For DFM (Gaussian + MLP), once $V$ and $c$ are set,

$$
m=V^\top\bar R,\qquad
S_f=V^\top\widehat\Sigma V-V^\top\operatorname{diag}(c)\,V,
$$

symmetrized, with eigenvalues floored at $10^{-4}$ before the Cholesky factorization. Only training observations enter this initializer; the DFM subsequently models raw synthetic returns. Adam updates $c_{\rm raw}$, $\zeta$, and (for Gaussian + MLP) $m,S_f$ at learning rate $10^{-3}$. In standard mode, hold $V$ fixed for 100 epochs, then update it at $10^{-4}$; reduced-budget runs shorten the warmup to retain a joint-training epoch. `project_orthogonal()` restores orthonormal columns by QR after each step. The triangular QR factor is given a positive diagonal, so no column changes sign; a sign flip would change the signed coordinates the MLP sees. The notebook selects these settings with `dfm_initialization="pca"`, `dfm_loading_warmup_epochs=100`, and `dfm_loading_lr_scale=0.1`.

**Time sampling and denoising loss (refinement 2).** Sample forward noising times from the log-uniform density

$$
u\sim\mathcal U(0,1),\qquad s=t_0(T/t_0)^u,\qquad
q(s)=\frac{1}{s\log(T/t_0)},\quad t_0\le s\le T,
$$

and minimize the noise-variance-weighted objective

$$
\mathcal L_{\rm DFM}(\theta)=
\mathbb E_{R,\,s\sim q,\,\xi}
\left[h_s\left\|s_\theta(R_s,s)+\frac{\xi}{\sqrt{h_s}}\right\|_2^2\right],
\qquad e_* = \arg\min_e\widehat{\mathcal L}_{\rm val}(\theta_e).
$$

The squared norm sums over assets; the empirical loss averages over observations. The target $-\xi/\sqrt{h_s}$ has expected squared norm $d/h_s$, so multiplication by $h_s$ removes this target-scale growth. Log-time sampling and this weight intentionally change the time weighting of the original uniform-time objective of Chen et al. (2025). Because the weights are positive and depend only on time, the unrestricted population minimizer remains the true score at each noising time. The notebook selects this objective with `dfm_time_sampling="log_balanced"`. The alternative `log_importance` multiplies the error by $s\log(T/t_0)/(T-t_0)$, which restores the uniform-time objective; generic training in [DFM.py](DFM.py) samples times uniformly.

Validation evaluates this same objective with fixed independent times and noise. The lowest-loss checkpoint $e_*$ is restored and frozen. The notebook implements both refinements through [`fit_dfm`](synthetic_benchmark/unconditional.py), called by `fit_and_save_dfm`; it returns the model, per-epoch losses, and selected epoch. Population moments and true factors are not supplied to the optimizer.

### 2.4 Reverse sampling

[`generate_samples`](sampling.py) integrates

$$
dX_t=b(t,X_t)\,dt+dW_t,\qquad
b(t,x)=\tfrac12x+s_\theta(x,T-t),\qquad X_0\sim N(0,I_d),
$$

by Euler–Maruyama, $X_{j+1}=X_j+\Delta t\,b(t_j,X_j)+\sqrt{\Delta t}\,\xi_j$. `generate_samples(reference, n, num_steps=J, generator=rng)` returns `[n,d]`; `return_trajectory=True` returns `[n,J+1,d]`, including both endpoints. It also accepts supplied initial states, Brownian increments of variance $\Delta t$, a guidance drift (section 5.2), and a final-step function (section 5.3), so samplers can be compared on the same noise.

**Fitted Gaussian reference** (in the code, not run by the notebook). `GaussianFactorScore` in [dfm_variants.py](dfm_variants.py) fits $N(\widehat\mu,\widehat V\widehat S\widehat V^\top+\operatorname{diag}\widehat c)$ to training rows and evaluates its analytic OU score with a small linear solve instead of a dense inverse. Direct draws measure the Gaussian fit; diffusion draws with this exact score measure what the sampler alone adds (finite $T$, Gaussian start, stopping at $t_0$, discretization), free of neural score error. Unlike DFM (Gaussian + MLP), whose clean mean is $Vm$, it keeps an arbitrary $d$-dimensional mean.

## 3. Freeze the DFM and construct conditioning coordinates

### 3.1 Frozen reference and reduced coordinates

`FrozenDFMReference` deep-copies the trained score, keeps the copy in evaluation mode, and disables its parameter gradients; input derivatives remain available for guidance. H changes the sampling procedure, not these parameters, so a different generator requires a new H.

With $\Lambda_s=D_s^{-1}$,

$$
\ell_t=\frac{\alpha_s}{\alpha_{t_0}}D_s\Lambda_{t_0}w,\qquad
G_t=\begin{bmatrix}V^\top D_s\\\ell_t^\top\end{bmatrix},\qquad
Y_t=G_tX_t=(Z_t,U_t),
$$

and $\ell_\tau=w$, hence $U_\tau=w^\top X_\tau$. `reference.G(t, w)` takes a scalar generation time and `[d]` weights and returns `[k+1,d]`. Batched times return `[N,k+1,d]`. `HNetwork.features(t,x)` maps `[N,d]` to `[N,k+1]`; at a shared scalar time this is `x @ G(t).T`. The path collector simulates all $d$ ambient coordinates and projects the recorded states through $G_t$.

[`ConditioningConfig(w,C)`](conditioning.py) fixes the event. `initialize_h` in [conditional.py](synthetic_benchmark/conditional.py) returns this event, an `HNetwork`, and its `DFMHGuidance`. The benchmark sets `log_remaining_time=True` for Refined CDG-ML and `False` for CDG-ML. The event uses synthetic standardized-return coordinates.

### 3.2 Closed reduced dynamics

For the continuous reference SDE, the reduced process has diffusion covariance $A_t=G_tG_t^\top$. With $n=g(z,s)$ denoting the full inner map (Gaussian posterior mean plus MLP correction for the selected network), its closed drift is

$$
\mu_Z(t,y)=-\tfrac12z+\alpha_s V^\top D_s^2Vn,\qquad
\mu_U(t,y)=\frac{\alpha_s^2}{\alpha_{t_0}}
w^\top\Lambda_{t_0}D_s^2Vn.
$$

The time derivative of $G_t$ cancels the linear terms of the ambient drift, so neither orthogonality nor invertibility of $G_t$ is needed. Two qualifications apply:

- The optional `HNetwork.mu` helper calls `g_zeta` alone and so omits the Gaussian component of DFM (Gaussian + MLP). The benchmark does not use it: it collects and guides ambient paths through the full score.
- Closure holds for the continuous dynamics, not for the Euler chain, whose next projected state uses $G_{t_{j+1}}$ rather than $G_{t_j}$. An H learned from projected finite-grid paths estimates the event probability given these features, which need not equal the full-state probability at every ambient state.

### 3.3 Raw-return coordinates

For returns preprocessed as $x=(R_{\rm raw}-\mu)/\sigma$ (the separate saved-equity-checkpoint workflow, [return_preprocessing.py](return_preprocessing.py)), the event is unchanged with $w_{\rm model}=w_{\rm raw}\odot\sigma$ and $C_{\rm model}=C_{\rm raw}-w_{\rm raw}^\top\mu$. Asset order and the mean offset are part of the event's identity.

## 4. Compare CDG-ML and Refined CDG-ML

### 4.1 Training paths

`h.sample_trajectories(N, num_steps=J, generator=rng, batch_size=256)` generates unguided paths from the frozen DFM with ordinary Gaussian starts. For path $i$, it assigns one label $L_i=\mathbf1\{w^\top X_\tau^{(i)}\le C\}$ and retains $(t_j,G_{t_j}X_{t_j}^{(i)})$ at every preterminal time. It returns `TerminalLabelData`:

| Field | Shape | Meaning |
| --- | --- | --- |
| `times` | `[J]` | Generation times from zero, excluding $\tau$ |
| `features` | `[N,J,k+1]` | Projected ambient path states |
| `labels` | `[N,1]` | One terminal event indicator per path |
| `weights` | `[J]` | $\omega_j=\Delta t_j/\tau$, summing to one |
| `last_probabilities` | `[N,1]` | Exact final Euler-step event probabilities |
| `last_states` | `[N,d]` | Ambient states before the final step, for diagnostics |

`sample_h_training_data` generates independent training, validation, and test sets of whole trajectories, checking both label classes. Terminal outcomes supply the labels; network inputs include only preterminal states. Splitting by whole trajectories keeps all time rows of a path in one split. The number of event paths depends on the frozen DFM's own event probability, not on the nominal population tail probability.

### 4.2 Population target

With positive time-dependent weights, the unrestricted population minimizer remains the conditional event probability,

$$
H_*(t,y)=\mathbb P(E\mid Y_t=y).
$$

In continuous time, this function defines the Doob $h$-transform through its logarithmic gradient (Guo et al., 2026). In the notebook, the training target corresponds to the discretized DFM and the information retained in the reduced coordinates.

### 4.3 CDG-ML

$H_\phi$ is a `(64,64)` Tanh MLP with sigmoid output and $k+2$ inputs $(Y_t,t/\tau)$. CDG-ML minimizes the interval-weighted Brier loss over all $J$ stored times of each path,

$$
\frac1N\sum_{i=1}^N\sum_{j=0}^{J-1}\omega_j\bigl(H_\phi(t_j,Y_{t_j}^{(i)})-L_i\bigr)^2,
$$

without class balancing. The hard terminal boundary is not imposed as a network constraint.

### 4.4 Refined CDG-ML

Refined CDG-ML makes five changes to CDG-ML, motivated in section 4 of the notebook: (1) time-dependent standardization of the reduced coordinates, (2) a log-remaining-time input, (3) an exact probability target at the last preterminal time, (4) more weight on late times, and (5) BCE instead of the Brier loss.

The `(64,64)` Tanh [`HNetwork`](h_model.py) implements refinements (1) and (2):

$$
\widetilde y=\frac{y-m_t^Y}{\sigma_t^Y},\qquad
H_\phi(t,y)=\operatorname{sigmoid}\!\left(
f_\phi\!\left(\widetilde y,\frac t\tau,
\log\frac{\max(\tau-t,\Delta t)}{\tau}\right)\right).
$$

The coordinatewise statistics are fitted on training paths only, interpolated in time, and saved with the checkpoint; scales have a floor of $10^{-6}$. They use the population (not sample) variance, and times outside the recorded grid are clamped to it. The input has $k+3$ coordinates.

`h_time_weights` defines the time probabilities (refinements 3 and 4)

$$
q_j=0.5\omega_j+
0.4\frac{\omega_j/(\tau-t_j)}{\sum_l\omega_l/(\tau-t_l)}
+0.1\mathbf1\{j=J-1\},\qquad \sum_jq_j=1.
$$

With $b_j(x)=\tfrac12x+s_\theta(x,T-t_j)$, the targets are

$$
p_{ij}=\begin{cases}
L_i,&j<J-1,\\[2mm]
\displaystyle\Phi\!\left(
\frac{C-w^\top[X_{J-1}^{(i)}+b_{J-1}(X_{J-1}^{(i)})\Delta t]}
{\sqrt{\Delta t}\,\|w\|_2}\right),&j=J-1.
\end{cases}
$$

The last target is exact for one frozen ambient Euler step conditional on full $x$; pointwise representability from reduced $G_tx$ is not assumed. It equals $P(L_i=1\mid X_{J-1}^{(i)})$, so regressing on it instead of $L_i$ leaves the feature-conditional minimizer unchanged. `HNetwork.exact_last_step_probability` computes it; it is the normalizer $\Phi(a(x))$ of the exact final step (section 5.3).

The objective is (refinement 5)

$$
\mathcal L_H=\frac1N\sum_{i=1}^N\sum_{j=0}^{J-1}q_j
\operatorname{BCE}\!\left(H_\phi(t_j,Y_{t_j}^{(i)}),p_{ij}\right).
$$

Training estimates the time sum by drawing 32 times per trajectory, with replacement, from $q$. No importance correction is applied: $q$ is the intended weighting, and because it depends only on time, the minimizer of section 4.2 is unchanged. BCE is evaluated from logits, which also accepts the soft final target.

### 4.5 Checkpoint selection

Each variant restores its best checkpoint under its own validation objective. The held-out binary-label Brier score and the gradient diagnostics of section 6.3 are computed identically for both. Equal epochs and paths do not mean equal computation: CDG-ML uses all $J$ times of every trajectory, Refined CDG-ML 32 sampled times.

## 5. Generate conditional returns

### 5.1 H-tilted initial states

The initial law is tilted as $p_0(x\mid E)\propto\varphi_d(x)\,H(0,G_0x)$. [`sample_conditional_initial`](guided_sampling.py) proposes $x\sim N(0,I_d)$, accepts with probability $H_\phi(0,G_0x)$, and returns accepted states `[N,d]` and acceptance diagnostics. Exhausting its proposal budget raises an error rather than falling back to Gaussian starts; by default the budget is $\max(100N,\text{batch size})$ proposals in total.

### 5.2 Guided Euler steps

The ambient guidance is the lifted gradient

$$
\nabla_x\log H_\phi(t,G_tx)=G_t^\top\nabla_y\log H_\phi(t,G_tx).
$$

`DFMHGuidance(h)(state, time)` returns this additive `[N,d]` drift. It evaluates $\log H_\phi$ stably from the logit and differentiates with respect to the input only, without accumulating parameter gradients; it requires autograd (not `inference_mode`) and a preterminal time. `sample_h_guided` first draws all accepted starts, then calls `generate_samples` in batches with `guidance_fn=guidance` and scale 1. The optional scale multiplying the correction follows Guo et al. (2026). It returns all terminal rows as NumPy `[N,d]`. With `final_step="exact"`, it uses the conditioned final transition instead of a guided Euler step.

### 5.3 Exact final step

With $\mu(x)=x+\Delta t\,b(t_{J-1},x)$, the unguided last step is $N(\mu(x),\Delta t\,I_d)$; conditioned on $E$, it is this Gaussian restricted to the half-space $w^\top x_J\le C$. Its normalizer $\Phi(a(x))$, with

$$
a(x)=\frac{C-w^\top\mu(x)}{\sqrt{\Delta t}\,\lVert w\rVert},
$$

is the exact last-step probability `exact_last_step_probability`. With $u=w/\lVert w\rVert$, $U\sim\mathcal U(0,1)$ and $Z'\sim N(0,I_d)$ independent, [`exact_final_step`](guided_sampling.py) draws

$$
X_J=\mu(X_{J-1})+\sqrt{\Delta t}\,\big[\eta\,u+(I-uu^\top)Z'\big],\qquad
\eta=\Phi^{-1}\!\big(U\,\Phi(a(X_{J-1}))\big),
$$

a standard normal truncated to $(-\infty,a]$ along the portfolio direction and unconstrained orthogonal noise. In exact arithmetic this is the conditioned final Euler kernel. It needs no population parameters, and its mean still depends on the learned score. It differs from the probability target of section 4.4, a scalar used to train H; this transition draws a whole return vector during generation. The generic sampling functions default to an Euler last step; the benchmark requests the exact step for both H models and for the no-guidance source.

The CDF and inverse CDF are computed in float64 and cast to the state dtype. The inverse-CDF argument is floored at $10^{-300}$, and when the event mass underflows, the directional component is set to the bound, so extreme incoming states can place mass exactly on $C$. Float32 arithmetic can leave exceedances of about $10^{-6}$. An ideal truncated normal has no mass on the boundary.

The construction assumes one linear half-space and an isotropic Euler innovation; other events or innovation covariances need a different conditional kernel.

### 5.4 Conditional-kernel derivation

Let $K_j(x,dy)$ be the unguided Euler kernel from $t_j$ to $t_{j+1}$, and define the full-state terminal probabilities

$$
h_j(x)=P(E\mid X_j=x),\qquad h_J(x)=\mathbf1\{w^\top x\le C\}.
$$

Bayes' rule gives the exact conditional kernel of the finite chain and its backward recursion,

$$
K_j^E(x,dy)=K_j(x,dy)\,\frac{h_{j+1}(y)}{h_j(x)},\qquad
h_j(x)=\int K_j(x,dy)\,h_{j+1}(y),
$$

wherever $h_j(x)>0$; the initial law is tilted by $h_0$. At the last transition, $h_J$ is an indicator, so conditioning truncates a Gaussian at the half-space (section 5.3). One step earlier, the weight is already a smooth probability,

$$
h_{J-1}(y)=\Phi\left(\frac{C-w^\top\mu(y)}{\sqrt{\Delta t}\,\lVert w\rVert}\right),
$$

so the conditioned kernel from $t_{J-2}$ is a Gaussian multiplied by this CDF and renormalized, not a truncated Gaussian. Farther back, $h_j$ integrates over all remaining transitions, and the nonlinear learned drift gives neither a closed form nor a simple sampler.

Truncating every intermediate portfolio value imposes a different path constraint. With independent standard normal $Z_1,Z_2$, let $X_1=Z_1$ and $X_2=X_1+Z_2$. Under $X_2\le0$,

$$
p(X_1=x\mid X_2\le0)=2\phi(x)\Phi(-x),
$$

which is positive for $x>0$; truncating $X_1$ at zero removes these valid paths.

Conversely, unguided paths reach the last step with density $p_{J-1}(x)$ instead of

$$
p_{J-1}(x\mid E)=\frac{p_{J-1}(x)\,h_{J-1}(x)}{P(E)}.
$$

A path whose incoming mean lies far above $C$ must then draw its final portfolio innovation, of scale $\sqrt{\Delta t}\,\lVert w\rVert$, deep in the lower tail, and it lands just below the threshold. This produces the boundary concentration of the no-guidance source.

The guided sources approximate the $h_j$ with H. Their remaining error comes from the learned probability and its gradient, the reduced features, and the Euler discretization of the guided drift; comparison with DFM rejection on the same grid measures these errors together. Exact sequential conditioning is possible when the $h_j$ and samplers for the weighted kernels are available: linear Gaussian chains admit analytic future probabilities, a Gaussian terminal law can be sampled directly (section 1.3), and full-path rejection is always valid.

## 6. Evaluate the five sources of conditional returns

### 6.1 Reference samples and Monte Carlo levels

[`sample_dfm_rejection`](synthetic_benchmark/conditional.py) runs the frozen DFM's sampler with its Gaussian start, finite horizon, stopping time, and Euler grid, without H, and accepts terminal rows with $w^\top x\le C$ exactly, using CPU NumPy float64 portfolio products. It conditions this numerical DFM and does not correct its error with respect to the population; distances to population truth do not add up to an error decomposition. Each stream stops at its proposal cap; a shortfall is recorded as a partial result, and comparisons use the sizes actually obtained. Nonfinite sampler outputs stop the run instead of being dropped.

Each distance is paired with a same-law comparison at the same sample size: truth against truth, or rejection against rejection. Larger guided or truth samples are compared with rejection through matched prefixes after fixed, seeded, value-independent permutations without replacement, and projection directions are shared within an assessment. "Near the Monte Carlo level" describes observed same-law variability, not a formal test of equal laws.

### 6.2 Distances and event diagnostics

The unconditional assessment reports covariance and mean errors, portfolio KS and quantiles, sliced W1, factor-direction and residual variance ratios, and frequencies below the exact population tail thresholds. The conditional assessment adds the analytic conditional portfolio CDF and the exact conditional moments of section 1.3. Sliced W1 averages one-dimensional W1 distances over 64 random unit directions; a small value neither checks every dependence nor excludes a defect at the portfolio boundary.

The displayed event rate allows a tolerance of $10^{-5}$, because the exact final step can put a row on $C$ and float32 can then exceed $C$ slightly; strict exceedances and their size are separate from this rate. "On threshold" means $|w^\top x-C|\le10^{-5}$, and the near-threshold band extends $3\sqrt{\Delta t}\,\lVert w\rVert$ below $C$. Compare this band with population truth, whose conditional density is often largest near the boundary.

### 6.3 H diagnostics

Both H models are scored on the same held-out test paths: binary-label Brier score with a constant-probability baseline, first-layer Tanh saturation (share of activations above 0.99 in absolute value), last-step probability RMSE, and the relative RMSE and cosine of the spatial gradient. The gradient check compares the ambient $\nabla_x\log H_\phi$ with $\nabla_x\log P(E\mid x)$ at final-preterminal test states whose exact last-step probability lies in $(0.01,0.99)$; this selection depends on the predicted final portfolio mean, not on the realized outcome. The Brier score and probability RMSE use the full test set.

### 6.4 Seeds, budgets, and timing

Independent named random streams separate observations, model initialization, corruption, H splits, guidance, rejection, matching, and projections. Timings separate path collection, H fitting, initialization, and generation. If guidance costs less per row than rejection, the break-even number of rows is the setup cost divided by the per-row saving, with rejection costed per accepted row; timings and break-even counts depend on the machine.

## Checkpoints and code map

H checkpoints contain the frozen reference, score architecture, event, H architecture, normalization statistics, time features, and metadata. `HNetwork.load_checkpoint` rebuilds the bound model; an H weight file alone is unusable without its reference and event. DFM checkpoints and population metadata are checked before reuse, and changing asset order, preprocessing, generator, event, or time grid invalidates a comparison.

| Concern | Main code |
| --- | --- |
| Score architectures and Gaussian posterior | [NeuralNets.py](NeuralNets.py) |
| DFM wrapper and numerical integration | [DFM.py](DFM.py), [sampling.py](sampling.py) |
| Training and validation selection | [synthetic_benchmark/unconditional.py](synthetic_benchmark/unconditional.py) |
| Frozen reference and return/event transforms | [reference.py](reference.py), [return_preprocessing.py](return_preprocessing.py), [conditioning.py](conditioning.py) |
| Coordinates, H training, and checkpoint bundles | [h_model.py](h_model.py), [h_setup.py](h_setup.py) |
| H initialization, lifted guidance, exact final step | [guided_sampling.py](guided_sampling.py), [tests/test_exact_final_step.py](tests/test_exact_final_step.py) |
| Population truth and provenance | [synthetic_conditional_truth.py](synthetic_conditional_truth.py) |
| Benchmark runs and diagnostics | [synthetic_benchmark/conditional.py](synthetic_benchmark/conditional.py), [reporting.py](synthetic_benchmark/reporting.py) |
