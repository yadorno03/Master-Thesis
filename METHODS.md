# Methods

The [notebook](synthetic_dfm_conditional_benchmark.ipynb) tests whether a diffusion factor model (DFM) and learned H guidance reproduce joint returns conditional on a portfolio stress event. Population truth checks the complete procedure; rejection from the fitted DFM checks the conditioning step.

| Setting | Presented experiment |
| --- | ---: |
| Assets / factors | 500 / 8 |
| DFM training / validation / evaluation observations | 2,500 / 750 / 3,000 |
| Returns seen by the DFM | Standardized per asset with training statistics |
| DFM epochs / hidden widths | 1,500 / (64, 64) |
| Reverse horizon / stopping time / Euler steps | $T=8$ / $t_0=0.005$ / $J=400$ |
| H training / validation / test trajectories per event | 40,000 / 2,000 / 2,000 |
| H epochs / hidden widths | 50 / (64, 64) |
| Conditional samples per source / accepted DFM rejection samples | 3,000 / 1,000 |

## 1. Construct the synthetic population

We use the Gaussian factor specification of [Chen et al. (2025)](https://arxiv.org/abs/2504.06566v5):

$$
R=\beta F+\varepsilon,\qquad
F\sim N(\mu_F,\operatorname{diag}(\sigma_F^2)),\qquad
\varepsilon\sim N(0,\operatorname{diag}(\sigma_\varepsilon^2)),\qquad F\perp\varepsilon,
$$

$$
\mu_{F,j}\overset{\mathrm{iid}}\sim\operatorname{Unif}(0,0.1),\quad
\sigma_{F,j}=1.5\mu_{F,j},\quad
\beta_{ij}\overset{\mathrm{iid}}\sim N(0,1),\quad
\sigma_{\varepsilon,i}\overset{\mathrm{iid}}\sim\operatorname{Unif}(0,0.4).
$$

Draw $\mu_F$, $\beta$ and $\sigma_\varepsilon$ independently with population seed 912, then hold all population parameters fixed across independent observation sets. Population returns retain their raw scale and nonzero mean; the DFM standardizes them (section 2).

The code uses equivalent orthonormal coordinates, $\beta=BA$, $B^\top B=I_k$:

$$
\widetilde F=AF\sim N(m,S),\quad m=A\mu_F,\quad
S=A\operatorname{diag}(\sigma_F^2)A^\top,\quad D_i=\sigma_{\varepsilon,i}^2,
\qquad \mu=Bm,\quad \Sigma=BSB^\top+\operatorname{diag}D.
$$

$S$ is generally full. The DFM receives observations only and learns its own loadings.

For the equal-weight portfolio, define

$$
w=\mathbf1/d,\quad P=w^\top R\sim N(a,v),\quad
a=w^\top\mu,\quad v=w^\top\Sigma w,\quad
E_q=\{P\le C_q\},\quad C_q=a+\sqrt v\,\Phi^{-1}(q).
$$

We assess $q=5\%$ and $1\%$, giving thresholds −1.1577% and −2.1632%. Exact conditional samples are obtained by replacing the Gaussian portfolio coordinate:

$$
R^E=R+\frac{\Sigma w}{v}(P^E-w^\top R),\qquad
P^E\sim N(a,v)\text{ truncated above at }C_q,
$$

with $P^E$ independent of $R$. Conditional moments and the portfolio CDF are available analytically.

## 2. Train and sample the unconditional DFM

**Standardized returns.** With the training mean $\hat m_i$ and (unbiased) standard deviation $\hat\sigma_i$ of each asset, the DFM models

$$
\tilde R_i=(R_i-\hat m_i)/\hat\sigma_i,
$$

and a generated state $X_\tau$ gives the returns $\hat m+\hat\sigma\odot X_\tau$. Chen et al. (2025) standardize each stock's returns in the same way before training on market data. Validation and evaluation returns never enter $\hat m,\hat\sigma$; the loss, the PCA initialization and the Gaussian prior all use standardized training returns. A diagonal rescaling keeps the factor structure: the loadings become $\operatorname{diag}(\hat\sigma)^{-1}\beta$ and the residual covariance stays diagonal.

The scale matters for the sampler. The OU process ends near $N(0,I_d)$, and generation stops at $t_0=0.005$ on a uniform grid with $\Delta t\simeq0.02$. On raw returns, 19% of the residual variances $D_i$ lie below $h_{t_0}\simeq0.005$. Even with the exact population score, this grid inflates the residual variance by about a third on raw returns, against about 3% on standardized returns. The residual part makes up about half of the equal-weight portfolio's variance.

The OU forward process and factor-aware score follow Chen et al.:

$$
\tilde R_s=\alpha_s\tilde R+\sqrt{h_s}\,\xi,\qquad
\alpha_s=e^{-s/2},\quad h_s=1-e^{-s},\quad \xi\sim N(0,I_d),
$$

$$
D_s=\operatorname{diag}(h_s+\alpha_s^2c_i)^{-1},\quad
z=V^\top D_sr,\qquad
s_\theta(r,s)=\alpha_sD_sVg(z,s)-D_sr.
$$

$V$ has orthonormal columns and $c_i\in(0,2)$. We compare two inner maps:

- **DFM (MLP):** $g=g_\zeta(z,s)$, a two-hidden-layer ReLU network.
- **DFM (Gaussian + MLP):** a Gaussian factor posterior mean plus the same MLP correction. With fitted prior $f\sim N(m_f,S_f)$ and $M_s=V^\top D_sV$,

$$
g(z,s)=m_f+(I_k+\alpha_s^2S_fM_s)^{-1}S_f\alpha_s(z-\alpha_sM_sm_f)+g_\zeta(z,s).
$$

The Gaussian prior is initialized from training observations; its MLP correction starts at zero. Both networks use the same data and two training refinements:

1. **PCA initialization and staged updates.** Initialize $V$ from training principal components and $c$ from residual variances. Hold $V$ fixed for 100 epochs, then train it at $10^{-4}$; Adam updates the other parameters at $10^{-3}$. Reorthogonalize $V$ after each update.
2. **Log-time, variance-weighted score matching.** Draw $s$ with density $q(s)=[s\log(T/t_0)]^{-1}$ and minimize

$$
\mathcal L_{\rm DFM}=\mathbb E\!\left[h_s\left\|s_\theta(\tilde R_s,s)+\xi/\sqrt{h_s}\right\|_2^2\right].
$$

This changes Chen's uniform time weighting while preserving the unrestricted population score minimizer. Each network restores its lowest validation-loss checkpoint.

Generation uses $s=T-t$, $\tau=T-t_0$, $\Delta t=\tau/J$ and Euler–Maruyama:

$$
X_0\sim N(0,I_d),\qquad b(t,x)=\tfrac12x+s_\theta(x,T-t),\qquad
X_{j+1}=X_j+\Delta t\,b(t_j,X_j)+\sqrt{\Delta t}\,\xi_j,
$$

and the generated returns are $\hat m+\hat\sigma\odot X_J$.

## 3. Freeze the DFM and construct conditioning coordinates

We freeze **DFM (Gaussian + MLP)**. Its states are standardized returns, so the event $w^\top R\le C_q$ reads $\tilde w^\top X_J\le\tilde C_q$ with

$$
\tilde w=\hat\sigma\odot w,\qquad \tilde C_q=C_q-w^\top\hat m.
$$

We condition through $k+1$ coordinates:

$$
\Lambda_s=D_s^{-1},\qquad
\ell_t=\frac{\alpha_s}{\alpha_{t_0}}D_s\Lambda_{t_0}\tilde w,\qquad
G_t=\begin{bmatrix}V^\top D_s\\\ell_t^\top\end{bmatrix},\qquad
Y_t=G_tX_t=(Z_t,U_t).
$$

At $t=\tau$, $\ell_\tau=\tilde w$, so $U_\tau=\tilde w^\top X_\tau=w^\top R-w^\top\hat m$ retains the complete portfolio event, including residual returns. H training leaves the DFM parameters unchanged.

## 4. Compare CDG-ML and Refined CDG-ML

Following [Guo et al. (2026)](https://arxiv.org/abs/2602.05533v2), H estimates

$$
H(t,y)=P(\tilde w^\top X_J\le \tilde C_q\mid Y_t=y).
$$

Both H models use the same unguided paths from the frozen DFM, with terminal labels $L_i=\mathbf1\{\tilde w^\top X_J^{(i)}\le \tilde C_q\}$, the return event of each path. Training, validation and test sets are split by whole trajectories. The labels reflect the numerical DFM's event frequency.

Both networks have two Tanh hidden layers and sigmoid outputs. Refined CDG-ML makes five changes:

| Component | CDG-ML | Refined CDG-ML |
| --- | --- | --- |
| Reduced inputs | $Y_t$ | Time-dependent standardization fitted on training paths |
| Time inputs | $t/\tau$ | Add $\log[\max(\tau-t,\Delta t)/\tau]$ |
| Targets | Binary terminal label at every time | Replace the last target by its exact one-step event probability (section 5) |
| Time weights | $\omega_j=\Delta t/\tau$ | Mixture $q_j$ below, emphasizing late times |
| Loss | Brier (squared error) | Binary cross-entropy from logits |

$$
q_j=0.5\omega_j+
0.4\frac{\omega_j/(\tau-t_j)}{\sum_l\omega_l/(\tau-t_l)}
+0.1\,\mathbf1\{j=J-1\}.
$$

Both train on all 400 times. CDG-ML evaluates every time of each trajectory at each step; Refined CDG-ML applies $q_j$ by evaluating each trajectory at 32 times drawn from $q_j$, redrawn at every step. Each restores its best checkpoint under its own validation objective, using learning rate $10^{-3}$, trajectory minibatches of 64 and validation every 157 optimizer steps. Equal paths and epochs therefore do not imply equal computation.

## 5. Generate conditional returns

Initial states are proposed from $N(0,I_d)$ and accepted with probability $H_\phi(0,G_0x)$, sampling the learned H tilt. Before the final step, guidance adds

$$
\nabla_x\log H_\phi(t,G_tx)=G_t^\top\nabla_y\log H_\phi(t,y)\big|_{y=G_tx}
$$

to the drift $b(t,x)$, with guidance scale 1.

### Exact final step

The last step, from $t_{J-1}$ to $\tau$, uses no H: it is the unguided DFM Euler step, drawn conditional on $E_q$. CDG-ML and Refined CDG-ML both end with it. Guidance only shifts the mean of each Gaussian step, which approximates conditioning on $E_q$ well while many steps remain. On the last step, conditioning instead cuts the Gaussian at the threshold. No mean shift reproduces this, but it can be sampled exactly.

From $X_{J-1}=x$, the unguided step is $N(\mu_J(x),\Delta t\,I_d)$ with $\mu_J(x)=x+\Delta t\,b(t_{J-1},x)$. The event depends only on the portfolio coordinate $\tilde w^\top X_J$, which is normal given $x$, so the step lands in $E_q$ with probability

$$
p_{\rm last}(x)=p_{J-1}(x)=\Phi(a_J(x)),\qquad
a_J(x)=\frac{\tilde C_q-\tilde w^\top\mu_J(x)}{\sqrt{\Delta t}\,\|\tilde w\|}.
$$

To draw the step given $E_q$, write it as $X_J=\mu_J(x)+\sqrt{\Delta t}\,\xi$ with $\xi\sim N(0,I_d)$, and let $\eta=u^\top\xi$ be the noise along the portfolio direction $u=\tilde w/\|\tilde w\|$. In terms of $\eta$, the event $E_q$ is exactly $\eta\le a_J(x)$. Since $\xi$ is standard normal, $\eta\sim N(0,1)$ is independent of $(I_d-uu^\top)\xi$, the noise orthogonal to $u$. Conditioning on $E_q$ therefore truncates $\eta$ above at $a_J(x)$ and leaves the orthogonal noise unchanged. With independent $U\sim\mathcal U(0,1)$ and $Z\sim N(0,I_d)$, inverse-CDF sampling draws the truncated $\eta$ and the step as

$$
\eta=\Phi^{-1}\big(U\,\Phi(a_J(x))\big),\qquad
X_J=\mu_J(x)+\sqrt{\Delta t}\,[\eta u+(I_d-uu^\top)Z].
$$

Thus $X_J$ follows the unguided last step conditioned on $E_q$, the law one would get by redrawing that step until it lands in $E_q$. Here a single draw suffices, so every sample satisfies the event and none is discarded. The step does not fix earlier steps. The output has the correct conditional law only if $X_{J-1}$ already follows its law given $E_q$, which is what the H-tilted start and guidance aim for. Without guidance, paths far from the event are forced into it in one step, so their portfolio returns end just below the threshold.

## 6. Evaluate the four sources of conditional returns

| Source | Construction |
| --- | --- |
| Conditional ground truth | Exact population law given $E_q$ |
| DFM rejection | Unguided DFM draws, mapped to returns, satisfying $E_q$, on the same Euler grid |
| CDG-ML | Baseline H guidance and exact final step |
| Refined CDG-ML | Five H refinements and exact final step |
