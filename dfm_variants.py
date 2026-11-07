"""Experimental upgrades of the factor DFM for high dimension (plan I0-I4).

All variants keep the score form of Chen et al. (2025), Eq. (18),
    s(r, t) = alpha_t D_t V g(V^T D_t r, t) - D_t r,
so the reduced coordinates used by H guidance remain available.

I0  fit_gaussian_factor_model / GaussianFactorScore: a low-rank-plus-diagonal Gaussian fitted
    directly to training returns, and its exact diffusion score (a reference, not a DFM).
I1  factor_selection_score: compares generated and validation covariances along the leading
    validation principal directions (checkpoint selection without population knowledge).
I2  FactorPreconditioner: shrinks chosen principal directions before training (r~ = P r);
    samples are mapped back with P^{-1}; linear events stay linear.
I3  GaussianResidualScoreNetwork: g = exact Gaussian factor posterior mean + zero-initialized MLP.
I4  StandardizedInputScoreNetwork: g sees V^T D_t r standardized per coordinate and per time.
"""
import math

import numpy as np
import torch
from torch import nn

from NeuralNets import FactorScoreNetwork


# ----------------------------------------------------------------------------- I0
def fit_gaussian_factor_model(x, k, min_variance=1e-4):
    """Fit mean, k principal loadings, factor covariance and residual variances (training rows only)."""
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(0)
    centered = x - mean
    covariance = centered.T @ centered / (len(x) - 1)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    V = vectors[:, ::-1][:, :k].copy()
    residual = centered - (centered @ V) @ V.T
    c = np.maximum(residual.var(0, ddof=1), min_variance)
    S = V.T @ covariance @ V - V.T @ (c[:, None] * V)
    w, U = np.linalg.eigh((S + S.T) / 2)
    S = (U * np.maximum(w, min_variance)) @ U.T
    return dict(mean=mean, V=V, S=S, c=c, eigenvalues=eigenvalues[::-1].copy())


def sample_gaussian_factor_model(params, n, rng):
    """Direct draws from the fitted low-rank-plus-diagonal Gaussian."""
    L = np.linalg.cholesky(params['S'])
    factors = rng.standard_normal((n, len(L))) @ L.T
    noise = rng.standard_normal((n, len(params['c']))) * np.sqrt(params['c'])
    return params['mean'] + factors @ params['V'].T + noise


class GaussianFactorScore(nn.Module):
    """Exact score of N(mean, V S V^T + diag(c)) under the OU forward process, per-row times."""

    def __init__(self, params):
        super().__init__()
        f64 = torch.float64
        self.register_buffer('mean', torch.as_tensor(params['mean'], dtype=f64))
        self.register_buffer('V', torch.as_tensor(params['V'], dtype=f64))
        self.register_buffer('S_inv', torch.linalg.inv(torch.as_tensor(params['S'], dtype=f64)))
        self.register_buffer('c', torch.as_tensor(params['c'], dtype=f64))

    def forward(self, r, t):
        x, t = r.to(torch.float64), t.to(torch.float64).reshape(-1, 1)
        alpha, h = torch.exp(-t / 2), -torch.expm1(-t)
        lam = h + alpha ** 2 * self.c                       # [B, d]
        v = (x - alpha * self.mean) / lam
        M = alpha[:, :, None] ** 2 * torch.einsum('dk,bd,dl->bkl', self.V, 1 / lam, self.V)
        y = alpha * (v @ self.V)
        z = torch.linalg.solve(self.S_inv + M, y.unsqueeze(-1)).squeeze(-1)
        return (-(v - alpha * (z @ self.V.T) / lam)).to(r.dtype)


class AnalyticGaussianDFM(nn.Module):
    """Wrapper exposing GaussianFactorScore to sampling.generate_samples."""

    def __init__(self, params, T, t0):
        super().__init__()
        self.factor_score_network = GaussianFactorScore(params)
        self.d, self.T, self.t0 = len(params['c']), float(T), float(t0)
        self._dtype_anchor = nn.Parameter(torch.zeros(1), requires_grad=False)


# ----------------------------------------------------------------------------- I1
def selection_reference(x_val, k, identifiable_ratio=5.0):
    """Leading validation principal directions, their variances and the residual variance."""
    x = np.asarray(x_val, dtype=np.float64)
    covariance = np.cov(x, rowvar=False)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    eigenvalues, vectors = eigenvalues[::-1], vectors[:, ::-1]
    residual = eigenvalues[k:].sum()
    c_bar = residual / (len(eigenvalues) - k)
    keep = eigenvalues[:k] >= identifiable_ratio * c_bar
    return dict(basis=vectors[:, :k][:, keep], variances=eigenvalues[:k][keep], residual=residual,
                full_basis=vectors[:, :k])


def factor_selection_score(generated, reference):
    """I1: mean |log variance ratio| on identifiable directions + |log residual variance ratio|."""
    x = np.asarray(generated, dtype=np.float64)
    covariance = np.cov(x, rowvar=False)
    variances = np.einsum('dk,de,ek->k', reference['basis'], covariance, reference['basis'])
    full = reference['full_basis']
    residual = np.trace(covariance) - np.einsum('dk,de,ek->', full, covariance, full)
    log_ratios = np.log(variances / reference['variances'])
    score = np.abs(log_ratios).mean() + abs(np.log(residual / reference['residual']))
    return float(score), log_ratios


# ----------------------------------------------------------------------------- I2
class FactorPreconditioner:
    """r~ = P r with P = I + V_r (diag(s) - I) V_r^T; s_j = sqrt(rho * c_bar / lambda_j).

    mode='pc1' shrinks only the first principal direction; mode='rho' shrinks every leading
    direction (j <= k) with lambda_j > rho * c_bar. Fitted on training rows only.
    """

    def __init__(self, V, s):
        self.V = np.asarray(V, dtype=np.float64).reshape(len(V), -1)
        self.s = np.asarray(s, dtype=np.float64).reshape(-1)

    @classmethod
    def fit(cls, x, k, rho=20.0, mode='rho'):
        x = np.asarray(x, dtype=np.float64)
        centered = x - x.mean(0)
        eigenvalues, vectors = np.linalg.eigh(centered.T @ centered / (len(x) - 1))
        eigenvalues, vectors = eigenvalues[::-1], vectors[:, ::-1]
        c_bar = eigenvalues[k:].sum() / (len(eigenvalues) - k)
        count = 1 if mode == 'pc1' else k
        chosen = [j for j in range(count) if eigenvalues[j] > rho * c_bar]
        pre = cls(vectors[:, chosen], np.sqrt(rho * c_bar / eigenvalues[chosen]))
        pre.info = dict(mode=mode, rho=rho, c_bar=float(c_bar), shrunk=len(chosen),
                        eigenvalues=eigenvalues[:k].tolist())
        return pre

    def _apply(self, x, factors):
        x = np.asarray(x, dtype=np.float64)
        return x + ((x @ self.V) * (factors - 1)) @ self.V.T

    def forward(self, x):
        return self._apply(x, self.s)

    def inverse(self, x):
        return self._apply(x, 1 / self.s)

    def transformed_weights(self, w):
        """w' r = (P^{-1} w)' r~ because P is symmetric."""
        return self.inverse(np.asarray(w, dtype=np.float64)[None])[0]


# ----------------------------------------------------------------------------- I3
class GaussianResidualScoreNetwork(FactorScoreNetwork):
    """g(u, t) = E_Gauss[f | r_t] + MLP(u, t), with an exact Gaussian factor posterior.

    For r = V f + eps, f ~ N(m, S), eps ~ N(0, diag(c)): E[f | r_t] =
    m + (I + alpha^2 S M_t)^{-1} S alpha (u - alpha M_t m), u = V^T D_t r, M_t = V^T D_t V.
    m and S (through a Cholesky factor) are trainable; the MLP's last layer starts at zero.
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


# ----------------------------------------------------------------------------- I4
class StandardizedInputScoreNetwork(FactorScoreNetwork):
    """g sees (u - mu_t) / sd_t per coordinate, with statistics on a log-time grid (training rows)."""

    def __init__(self, d, k, sigma_max=1.0, hidden_dims=(128, 128), T=1.0, t0=1e-4, grid_size=32):
        super().__init__(d, k, sigma_max, list(hidden_dims))
        self.register_buffer('log_grid', torch.linspace(math.log(t0), math.log(T), grid_size))
        self.register_buffer('input_mean', torch.zeros(grid_size, k))
        self.register_buffer('input_std', torch.ones(grid_size, k))

    def _projected(self, r, t):
        alpha, h = torch.exp(-t / 2.0), -torch.expm1(-t)
        c = torch.sigmoid(self.c_raw) * self.sigma_max
        D = 1.0 / (h + alpha ** 2 * c.unsqueeze(0))
        return D, D * r

    @torch.no_grad()
    def refresh_input_statistics(self, observations, generator=None, rows=2048):
        x = observations[torch.randperm(len(observations), generator=generator)[:rows]].to(self.V.dtype)
        for i, log_t in enumerate(self.log_grid):
            t = torch.full((len(x), 1), float(log_t.exp()), dtype=x.dtype)
            alpha, h = torch.exp(-t / 2), -torch.expm1(-t)
            xt = alpha * x + torch.sqrt(h) * torch.randn(x.shape, generator=generator, dtype=x.dtype)
            u = self._projected(xt, t)[1] @ self.V
            self.input_mean[i] = u.mean(0)
            self.input_std[i] = u.std(0).clamp(min=1e-6)

    def _interpolate(self, table, t):
        position = (torch.log(t[:, 0]) - self.log_grid[0]) / (self.log_grid[1] - self.log_grid[0])
        position = position.clamp(0, len(self.log_grid) - 1 - 1e-6)
        low = position.floor().long()
        weight = (position - low).unsqueeze(-1)
        return (1 - weight) * table[low] + weight * table[low + 1]

    def forward(self, r, t):
        alpha = torch.exp(-t / 2.0)
        D, Dr = self._projected(r, t)
        u = Dr @ self.V
        z = (u - self._interpolate(self.input_mean, t)) / self._interpolate(self.input_std, t)
        g = self.g_zeta(torch.cat([z, t], dim=-1))
        return alpha * D * (g @ self.V.t()) - Dr
