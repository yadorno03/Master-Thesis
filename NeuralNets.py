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
    """
    def __init__(self, d, k, sigma_max=1.0, hidden_dims=[128, 128]):
        super().__init__()
        self.d = d
        self.k = k
        self.sigma_max = sigma_max

        # Trainable factor loading matrix V (orthogonal columns, shape d x k)
        self.V = nn.Parameter(torch.empty(d, k))
        nn.init.orthogonal_(self.V)

        # Trainable idiosyncratic noise variance c_raw (shape d)
        # Bounded in [0, sigma_max] via scaled sigmoid in the forward pass.
        self.c_raw = nn.Parameter(torch.empty(d))
        nn.init.uniform_(self.c_raw, -1.0, 1.0)
        # ReLU-based neural network g_zeta (takes projection of size k + time 1 -> output size k)
        layers = []
        in_dim = k + 1
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

    def forward(self, r, t):
        """Score s_theta(r, t) for returns r of shape (B, d) and times t of shape (B, 1)."""
        alpha_t = torch.exp(-t / 2.0)                    # (B, 1)
        h_t = -torch.expm1(-t)                           # (B, 1)
        c = torch.sigmoid(self.c_raw) * self.sigma_max   # (d,), in (0, sigma_max)
        D_t = 1.0 / (h_t + (alpha_t ** 2) * c.unsqueeze(0))  # diagonal of D_t, (B, d)
        D_t_r = D_t * r
        projected = torch.matmul(D_t_r, self.V)          # V^T D_t r, (B, k)
        g_out = self.g_zeta(torch.cat([projected, t], dim=-1))
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

    def __init__(self, d, k, sigma_max=1.0, hidden_dims=(128, 128)):
        super().__init__(d, k, sigma_max, list(hidden_dims))
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
        g = g_lin + self.g_zeta(torch.cat([u, t], dim=-1))
        return alpha * D * (g @ self.V.t()) - Dr


# Score networks a DFM checkpoint can name: 'mlp' is the network of Chen et al. (2025), with an
# MLP g; 'gaussian_mlp' adds the exact Gaussian factor part to that MLP.
SCORE_NETWORKS = {'mlp': FactorScoreNetwork, 'gaussian_mlp': GaussianMLPScoreNetwork}


def make_score_network(kind, d, k, sigma_max=1.0, hidden_dims=(128, 128)):
    """Construct a registered score network by name."""
    if kind not in SCORE_NETWORKS:
        raise ValueError(f'Unknown score network {kind!r}; expected one of {sorted(SCORE_NETWORKS)}')
    return SCORE_NETWORKS[kind](d, k, sigma_max, list(hidden_dims))


def score_network_kind(network):
    """Registered name of a score network instance (exact type, not a subclass)."""
    for kind, cls in SCORE_NETWORKS.items():
        if type(network) is cls:
            return kind
    raise ValueError(f'Unregistered score network type {type(network).__name__}')
