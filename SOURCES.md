# Paper sources and adaptations

Attribution for the current [experiment](synthetic_dfm_conditional_benchmark.ipynb). The formulas and settings are in [METHODS.md](METHODS.md).

## Ideas taken from papers

| Paper | Ideas used here | How they are used |
| --- | --- | --- |
| [Chen et al., *Diffusion Factor Models* (2025; v5 revised 2026)](https://arxiv.org/html/2504.06566v5) | Gaussian factor population; QR factor coordinates; OU corruption; denoising score matching; factor-aware score with a low-dimensional ReLU network; reverse diffusion sampling. | The main DFM foundation. See §§2.1–2.2, §3.2 (Eq. 18) and Appendix D. Network size, population dimensions and training settings are chosen for this experiment. |
| [Guo et al., *Conditional Diffusion Guidance under Hard Constraint* (2026, v2)](https://arxiv.org/html/2602.05533v2) | Terminal-event probability H; CDG-ML regression of terminal labels along frozen-model paths; Doob guidance through the gradient of log H; guidance scaling; distinction between probability and gradient accuracy. | The conditioning foundation, applied here to portfolio events and reduced coordinates. See §§3.1–3.2.1 and §5.2. |
| [Bai, *Inferential Theory for Factor Models of Large Dimensions* (2003)](https://doi.org/10.1111/1468-0262.00392) | Principal-components estimation of factor structure. | Adapted to initialize the DFM loadings and residual variances from training observations. |
| [Song and Ermon, *Generative Modeling by Estimating Gradients of the Data Distribution* (2019)](https://arxiv.org/html/1907.05600v3#S4.SS1) | Geometrically spaced noise levels and noise-variance weighting (§§4.1–4.2). | Motivates covering small noise scales. Our log-uniform sampling of continuous OU time is an adaptation. |
| [Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations* (2021)](https://arxiv.org/html/2011.13456v2#S3.SS3) | Weight score error by the inverse expected squared target-score magnitude (§3.3, Eq. 7). | For our OU corruption, this yields the noise-variance weight $h_s$, up to a constant. |
| [Ho et al., *Denoising Diffusion Probabilistic Models* (2020)](https://arxiv.org/html/2006.11239v2#S3.SS4) | Noise-prediction mean squared error (§3.4, Eq. 14). | Supports the equivalent interpretation of our $h_s$-weighted score loss as noise-prediction error. |
| [Karras et al., *Elucidating the Design Space of Diffusion-Based Generative Models* (2022)](https://arxiv.org/abs/2206.00364) | Diffusion-network preconditioning with an analytical skip component and a learned residual. | Inspiration for Gaussian + MLP. Our Gaussian factor posterior formula and its neural correction are a project adaptation. |

## Adaptations made in this project

| Component | Attribution |
| --- | --- |
| Gaussian + MLP | Combines Chen's outer score form, standard Gaussian conditioning and the residual-design motivation of Karras et al. |
| DFM training recipe | Adds a loading warmup and smaller loading learning rate after PCA initialization. Combines log-uniform OU times with $h_s$ weighting; no single cited paper supplies this exact recipe. |
| Factor-plus-portfolio coordinates | Derives the specific time-dependent $k+1$ conditioning state from the DFM score and the portfolio event. Gradient lifting uses the chain rule. |
| Five H refinements | Combines time-dependent input standardization, log remaining time, an exact last-step probability target, greater late-time weight and BCE. These are numerical choices for this experiment; the probability target integrates out the final Gaussian noise. |
| Conditional initial states | Uses the standard Bayes tilting identity through acceptance sampling with learned $H_\phi(0,G_0x_0)$. This samples the learned H tilt. |
| Exact final step | Conditions the final Gaussian Euler transition on the portfolio half-space. This uses standard Gaussian truncation and is exact for that transition. |
| Evaluation | Combines analytic population truth, independent DFM rejection and guided samples to assess generator and conditioning errors separately, with matched-sample noise levels. |

QR, Gaussian conditioning, rejection sampling, input standardization, BCE and the statistical distances are established tools. The adaptations above describe their use in this study, without claiming invention of those tools. Chen's empirical U-Net and Guo's CDG-MCL are outside the presented experiment.
