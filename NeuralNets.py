import math

import torch
import torch.nn as nn


class FactorScoreNetwork(nn.Module):
    """
    Core score network for the Diffusion Factor Model (DFM).
    Approximates the score function using the factor-aware decomposition:
        s_theta(r, t) = alpha_t * D_t * V * g_zeta(V^T * D_t * r, t) - D_t * r
    This is the score network of Chen et al. (2025), with an MLP g_zeta.

    Attributes:
        d (int): Asset return dimension.
        k (int): Latent factor dimension (k << d).
        sigma_max (float): Maximum boundary for noise variance c.
        V (nn.Parameter): Trainable matrix of shape (d, k) with orthonormal columns.
        c_raw (nn.Parameter): Trainable vector of shape (d), representing raw noise variances.
        g_zeta (nn.Module): ReLU MLP taking the k-dimensional projection + time as input.
        refit_knots (int): With refit_knots > 0, g gains a linear term Gamma(t) u in u = V^T D_t r,
            Gamma piecewise linear in log t between refit_knots knots (buffers refit_log_times and
            refit_gamma). It is zero until synthetic_benchmark.unconditional.refit_linear_part fits it.
    """
    def __init__(self, d, k, sigma_max=1.0, hidden_dims=[128, 128], extra_inputs=0, refit_knots=0):
        super().__init__()
        self.d = d
        self.k = k
        self.sigma_max = sigma_max
        self.refit_knots = int(refit_knots)
        if self.refit_knots == 1 or self.refit_knots < 0:
            raise ValueError('refit_knots must be 0 (no refit) or at least 2')
        if self.refit_knots:
            # Any increasing knots: Gamma is zero until it is fitted.
            self.register_buffer('refit_log_times', torch.linspace(0.0, 1.0, self.refit_knots))
            self.register_buffer('refit_gamma', torch.zeros(self.refit_knots, k, k))

        # Trainable factor loading matrix V (orthogonal columns, shape d x k)
        self.V = nn.Parameter(torch.empty(d, k))
        nn.init.orthogonal_(self.V)

        # Trainable idiosyncratic noise variance c_raw (shape d)
        # Bounded in [0, sigma_max] via scaled sigmoid in the forward pass.
        self.c_raw = nn.Parameter(torch.empty(d))
        nn.init.uniform_(self.c_raw, -1.0, 1.0)
        # ReLU-based neural network g_zeta (takes projection of size k + time 1 -> output size k);
        # a subclass may add inputs between the projection and the time.
        layers = []
        in_dim = k + 1 + extra_inputs
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, k))
        self.g_zeta = nn.Sequential(*layers)

    @torch.no_grad()
    def project_orthogonal(self):
        """
        Retract V to orthonormal columns using QR with positive diagonal R.

        The sign convention makes this map the identity on an already
        orthonormal V and continuous nearby. Bare QR can flip columns without
        changing their span, but g_zeta uses their signed coordinates: flipping
        V alone would abruptly change the learned score.
        Call after each optimizer step.
        """
        q, r = torch.linalg.qr(self.V)
        diagonal = r.diagonal()
        signs = torch.where(diagonal < 0, -torch.ones_like(diagonal),
                            torch.ones_like(diagonal))
        self.V.copy_(q * signs)

    def refit_hats(self, t):
        """Weights [B, K] of the refit knots at times t [B, 1]: hat functions of log t, constant outside."""
        knots = self.refit_log_times
        position = ((t[:, 0].log() - knots[0]) / (knots[1] - knots[0])).clamp(0, len(knots) - 1)
        lower = position.floor().clamp(max=len(knots) - 2)
        upper = (position - lower).unsqueeze(-1)
        index = torch.arange(len(knots), device=t.device)
        return ((index == lower.unsqueeze(-1)) * (1 - upper) + (index == lower.unsqueeze(-1) + 1) * upper).to(t.dtype)

    def linear_refit(self, u, t):
        """The refit term Gamma(t) u of g, [B, k], for projections u [B, k]; zero without a refit."""
        if not self.refit_knots:
            return torch.zeros_like(u)
        gamma = torch.einsum('bm,mij->bij', self.refit_hats(t), self.refit_gamma)
        return (gamma @ u.unsqueeze(-1)).squeeze(-1)

    def forward(self, r, t):
        """Score s_theta(r, t) for returns r of shape (B, d) and times t of shape (B, 1)."""
        alpha_t = torch.exp(-t / 2.0)                    # (B, 1)
        h_t = -torch.expm1(-t)                           # (B, 1)
        c = torch.sigmoid(self.c_raw) * self.sigma_max   # (d,), in (0, sigma_max)
        D_t = 1.0 / (h_t + (alpha_t ** 2) * c.unsqueeze(0))  # diagonal of D_t, (B, d)
        D_t_r = D_t * r
        projected = torch.matmul(D_t_r, self.V)          # V^T D_t r, (B, k)
        g_out = self.g_zeta(torch.cat([projected, t], dim=-1)) + self.linear_refit(projected, t)
        lifted = torch.matmul(g_out, self.V.t())         # V g, (B, d)
        return alpha_t * D_t * lifted - D_t_r


class GaussianMLPScoreNetwork(FactorScoreNetwork):
    """The same score form, with g = exact Gaussian factor posterior mean + MLP correction.

    This is our modification of the network g of Chen et al. (2025); the outer
    score form, and hence the reduced coordinates used by H guidance, is unchanged.
    For r = V f + eps, f ~ N(m, S), eps ~ N(0, diag(c)), the noised return
    r_t = alpha r + sqrt(h) xi has
        E[f | r_t] = m + (I + alpha^2 S M_t)^{-1} S alpha (u - alpha M_t m),
    with u = V^T D_t r_t and M_t = V^T D_t V. This affine map is computed in closed
    form; the MLP only learns the departure from it. m and S (through a Cholesky
    factor) stay trainable; the MLP's last layer starts at zero, so training starts
    from the Gaussian model fitted by initialize_prior.
    """

    def __init__(self, d, k, sigma_max=1.0, hidden_dims=(128, 128), extra_inputs=0, refit_knots=0):
        super().__init__(d, k, sigma_max, list(hidden_dims), extra_inputs, refit_knots)
        self.prior_mean = nn.Parameter(torch.zeros(k))
        self.prior_chol = nn.Parameter(torch.eye(k))
        nn.init.zeros_(self.g_zeta[-1].weight)
        nn.init.zeros_(self.g_zeta[-1].bias)

    @torch.no_grad()
    def initialize_prior(self, observations):
        """Prior mean and covariance of V^T r from training rows, after V and c are initialized."""
        V = self.V
        c = torch.sigmoid(self.c_raw) * self.sigma_max
        x = observations.to(V.dtype)
        mean = x.mean(0)
        centered = x - mean
        S = V.T @ (centered.T @ centered / (len(x) - 1)) @ V - V.T @ (c[:, None] * V)
        w, U = torch.linalg.eigh((S + S.T) / 2)
        S = (U * w.clamp(min=1e-4)) @ U.T
        self.prior_mean.copy_(V.T @ mean)
        self.prior_chol.copy_(torch.linalg.cholesky(S))

    def forward(self, r, t):
        alpha, h = torch.exp(-t / 2.0), -torch.expm1(-t)
        c = torch.sigmoid(self.c_raw) * self.sigma_max
        D = 1.0 / (h + alpha ** 2 * c.unsqueeze(0))            # [B, d]
        Dr = D * r
        u = Dr @ self.V                                         # [B, k]
        M = torch.einsum('dk,bd,dl->bkl', self.V, D, self.V)    # [B, k, k]
        L = torch.tril(self.prior_chol)
        S = L @ L.T  # (I + alpha^2 S M) stays invertible for any PSD S
        a = alpha.unsqueeze(-1)                                 # [B, 1, 1]
        rhs = alpha * (u - alpha * (M @ self.prior_mean))       # [B, k]
        A = torch.eye(self.k, dtype=L.dtype, device=L.device) + a ** 2 * (S @ M)
        g_lin = self.prior_mean + torch.linalg.solve(A, (rhs @ S.T).unsqueeze(-1)).squeeze(-1)
        g = g_lin + self.g_zeta(torch.cat([u, t], dim=-1)) + self.linear_refit(u, t)
        return alpha * D * (g @ self.V.t()) - Dr


REGIME_PARAMETERS = ('log_factor_scale', 'log_residual_scale', 'regime_logits')


def normalized_regimes(log_factor_scale, log_residual_scale, regime_logits):
    """Squared scales a_j^2 [J_a], b_l^2 [J_b] with mean 1 under the table, and log pi [J_a, J_b]."""
    log_pi = torch.log_softmax(regime_logits.flatten(), 0).view_as(regime_logits)
    pi = log_pi.exp()
    a2, b2 = torch.exp(2 * log_factor_scale), torch.exp(2 * log_residual_scale)
    return a2 / (pi.sum(1) @ a2), b2 / (pi.sum(0) @ b2), log_pi


def regime_terms(x, V, alpha, h, c, mean, chol, a2, b2, log_pi):
    """Components of the noised scale mixture X_s = alpha (V F + eps) + sqrt(h) xi at states x [B, d].

    Regime (j, l) has F ~ N(mean, a_j^2 S), S = chol chol^T, and eps ~ N(0, b_l^2 diag(c)).
    alpha and h are [B, 1], or [1, 1] when every row shares one time. Returns
        log_joint [B, J_a, J_b]: log pi_jl + log N(x; alpha V mean, alpha^2 a_j^2 V S V^T + D_l^{-1}),
        shift [B, J_a, J_b, k]:  E[F | x, j, l] - mean,
        D [J_b, B or 1, d]:      diagonal of D_l = diag(h + alpha^2 b_l^2 c)^{-1}.
    Only k x k systems are solved (matrix inversion lemma), through the symmetric
    K_jl = I + alpha^2 a_j^2 L^T M_l L, with L = chol and M_l = V^T D_l V.
    """
    d, k = V.shape
    L = torch.tril(chol)
    eye = torch.eye(k, dtype=V.dtype, device=V.device)
    scales = a2.view(1, -1, 1, 1)
    log_joint, shifts, diagonals = [], [], []
    for l in range(len(b2)):
        D = 1.0 / (h + alpha ** 2 * b2[l] * c.unsqueeze(0))                # [B, d]
        Dx = D * x
        u = Dx @ V                                                          # [B, k]
        M = (V.unsqueeze(0) * D.unsqueeze(-1)).transpose(1, 2) @ V          # [B, k, k]
        r = u - alpha * (M @ mean)                                          # [B, k]
        a = (alpha ** 2).unsqueeze(-1).unsqueeze(1)                         # [B, 1, 1, 1]
        R = torch.linalg.cholesky(eye + a * scales * (L.T @ M @ L).unsqueeze(1))   # [B, J_a, k, k]
        # (I + alpha^2 a_j^2 S M)^{-1} a_j^2 S = a_j^2 L K^{-1} L^T
        W = scales * (L @ torch.cholesky_inverse(R) @ L.T)
        shift = alpha.unsqueeze(1) * torch.einsum('bjkl,bl->bjk', W, r)     # [B, J_a, k]
        quad = ((x * Dx).sum(1, keepdim=True) - 2 * alpha * (u @ mean).unsqueeze(1)
                + alpha ** 2 * ((M @ mean) @ mean).unsqueeze(1) - alpha * (shift * r.unsqueeze(1)).sum(-1))
        logdet = -D.log().sum(1, keepdim=True) + 2 * R.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        log_joint.append(log_pi[:, l] - 0.5 * (quad + logdet + d * math.log(2 * math.pi)))
        shifts.append(shift)
        diagonals.append(D)
    return torch.stack(log_joint, -1), torch.stack(shifts, 2), torch.stack(diagonals)


class RegimeMLPScoreNetwork(GaussianMLPScoreNetwork):
    """Gaussian + MLP with the Gaussian part replaced by a scale mixture of Gaussians.

    The analytic part is the exact score of the scale mixture (Andrews et al., 1974)
        r = V f + eps,  (j, l) ~ pi,  f | j ~ N(m, a_j^2 S),  eps | l ~ N(0, b_l^2 diag(c)),
    with J_a factor scales, J_b residual scales and a free joint table pi: a common
    daily volatility level makes factors and residuals large together. The scales
    are normalized so that sum pi a^2 = sum pi b^2 = 1: V S V^T + diag(c) stays the
    covariance and c keeps its meaning. The score of the mixture is the posterior
    average of the regime scores, each of the Gaussian + MLP closed form with
    S -> a_j^2 S and D -> D_l = diag(h + alpha^2 b_l^2 c)^{-1}.

    The MLP correction (and any linear refit) is added in the coordinates of b = 1, as in
    Gaussian + MLP, with one more input: the posterior mean of log b_l, the volatility level.
    V, c_raw, sigma_max, d and k keep their meaning, so the reduced coordinates
    of H guidance are unchanged.
    """

    def __init__(self, d, k, sigma_max=1.0, hidden_dims=(128, 128), regimes=(4, 3), refit_knots=0):
        super().__init__(d, k, sigma_max, hidden_dims, extra_inputs=1, refit_knots=refit_knots)
        factor_scales, residual_scales = (int(count) for count in regimes)
        if factor_scales < 1 or residual_scales < 1:
            raise ValueError('regimes must be two positive counts: factor scales and residual scales')
        self.regimes = (factor_scales, residual_scales)
        # Log-spaced starting scales; a single scale is 1.
        self.log_factor_scale = nn.Parameter(
            torch.linspace(math.log(0.6), math.log(2.7), factor_scales) if factor_scales > 1 else torch.zeros(1))
        self.log_residual_scale = nn.Parameter(
            torch.linspace(math.log(0.7), math.log(1.8), residual_scales) if residual_scales > 1
            else torch.zeros(1))
        self.regime_logits = nn.Parameter(torch.zeros(factor_scales, residual_scales))

    def regime_scales(self):
        """Normalized scales a [J_a], b [J_b] and the probability table pi [J_a, J_b]."""
        a2, b2, log_pi = normalized_regimes(self.log_factor_scale, self.log_residual_scale, self.regime_logits)
        return a2.sqrt(), b2.sqrt(), log_pi.exp()

    def _terms(self, r, t):
        alpha, h = torch.exp(-t / 2.0), -torch.expm1(-t)
        c = torch.sigmoid(self.c_raw) * self.sigma_max
        a2, b2, log_pi = normalized_regimes(self.log_factor_scale, self.log_residual_scale, self.regime_logits)
        return alpha, h, c, b2, regime_terms(r, self.V, alpha, h, c, self.prior_mean, self.prior_chol,
                                             a2, b2, log_pi)

    def log_density(self, r, t):
        """Log density of the analytic part (the noised scale mixture) at time t, [B]."""
        return torch.logsumexp(self._terms(r, t)[-1][0].flatten(1), 1)

    def forward(self, r, t):
        alpha, h, c, b2, (log_joint, shift, D) = self._terms(r, t)
        weights = torch.softmax(log_joint.flatten(1), 1).view_as(log_joint)       # omega_jl, [B, J_a, J_b]
        score = torch.zeros_like(r)
        for l in range(len(b2)):
            mass = weights[:, :, l].sum(1, keepdim=True)                           # [B, 1]
            g = mass * self.prior_mean + (weights[:, :, l].unsqueeze(-1) * shift[:, :, l]).sum(1)
            score = score + alpha * D[l] * (g @ self.V.t()) - mass * D[l] * r
        level = weights.sum(1) @ b2.log().unsqueeze(-1) / 2                        # E[log b | r], [B, 1]
        D1 = 1.0 / (h + alpha ** 2 * c.unsqueeze(0))
        u = (D1 * r) @ self.V
        correction = self.g_zeta(torch.cat([u, level, t], dim=-1)) + self.linear_refit(u, t)
        return score + alpha * D1 * (correction @ self.V.t())

    @torch.no_grad()
    def initialize_regimes(self, observations, steps=400, lr=0.02, moment_penalty=100.0):
        """Fit the scale mixture to training rows by maximum likelihood, with V fixed.

        Call after V, c and the prior are initialized. The exact log-likelihood
        of the rows (the density above at alpha = 1, h = 0) is maximized over the
        scales, the table, m, S and c by full-batch Adam in float64. The scales
        are normalized, so V S V^T + diag(c) is the covariance of the fitted law;
        ``moment_penalty`` times the squared relative distance of its factor
        block and of its variances to those of the rows is subtracted from the
        log-likelihood per row: without it the likelihood trades second moments
        for tails. Returns the log-likelihood per row before and after, and the
        relative distances of the fitted factor covariance and variances to the
        sample ones.
        """
        x = observations.detach().double()
        V = self.V.detach().double()
        names = ('prior_mean', 'prior_chol', 'c_raw', *REGIME_PARAMETERS)
        free = {name: getattr(self, name).detach().double().clone().requires_grad_(True) for name in names}
        centered = x - x.mean(0)
        target_factor = V.T @ (centered.T @ centered / (len(x) - 1)) @ V
        target_variance = x.var(0)
        one, zero = x.new_ones(1, 1), x.new_zeros(1, 1)

        def fit_terms():
            c = torch.sigmoid(free['c_raw']) * self.sigma_max
            a2, b2, log_pi = normalized_regimes(*(free[name] for name in REGIME_PARAMETERS))
            log_joint = regime_terms(x, V, one, zero, c, free['prior_mean'], free['prior_chol'], a2, b2, log_pi)[0]
            L = torch.tril(free['prior_chol'])
            S = L @ L.T
            factor_error = torch.linalg.norm(S + V.T @ (c[:, None] * V) - target_factor) / torch.linalg.norm(
                target_factor)
            variance_error = ((((V @ S) * V).sum(1) + c) / target_variance - 1).square().mean().sqrt()
            return torch.logsumexp(log_joint.flatten(1), 1).mean(), factor_error, variance_error

        optimizer = torch.optim.Adam(list(free.values()), lr=lr)
        with torch.enable_grad():
            initial = float(fit_terms()[0])
            for _ in range(steps):
                optimizer.zero_grad()
                likelihood, factor_error, variance_error = fit_terms()
                (-likelihood + moment_penalty * (factor_error ** 2 + variance_error ** 2)).backward()
                optimizer.step()
            likelihood, factor_error, variance_error = fit_terms()
        if not all(torch.isfinite(value).all() for value in free.values()):
            raise ValueError('Nonfinite regime initialization')
        for name, value in free.items():
            getattr(self, name).copy_(value.detach().to(self.V.dtype))
        return dict(initial_log_likelihood=initial, log_likelihood=float(likelihood),
                    factor_covariance_error=float(factor_error), variance_error=float(variance_error))


# Score networks a DFM checkpoint can name: 'mlp' is the network of Chen et al. (2025), with an
# MLP g; 'gaussian_mlp' adds the exact Gaussian factor part to that MLP; 'regime_mlp' replaces
# that Gaussian part by a scale mixture of Gaussians (volatility regimes).
SCORE_NETWORKS = {'mlp': FactorScoreNetwork, 'gaussian_mlp': GaussianMLPScoreNetwork,
                  'regime_mlp': RegimeMLPScoreNetwork}


def make_score_network(kind, d, k, sigma_max=1.0, hidden_dims=(128, 128), regimes=None, refit_knots=0):
    """Construct a registered score network by name; ``regimes`` is read by 'regime_mlp' only."""
    if kind not in SCORE_NETWORKS:
        raise ValueError(f'Unknown score network {kind!r}; expected one of {sorted(SCORE_NETWORKS)}')
    if kind == 'regime_mlp':
        return RegimeMLPScoreNetwork(d, k, sigma_max, list(hidden_dims), regimes=tuple(regimes or (4, 3)),
                                     refit_knots=refit_knots)
    return SCORE_NETWORKS[kind](d, k, sigma_max, list(hidden_dims), refit_knots=refit_knots)


def score_network_kind(network):
    """Registered name of a score network instance (exact type, not a subclass)."""
    for kind, cls in SCORE_NETWORKS.items():
        if type(network) is cls:
            return kind
    raise ValueError(f'Unregistered score network type {type(network).__name__}')
