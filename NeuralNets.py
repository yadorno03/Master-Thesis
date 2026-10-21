import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class FactorScoreNetwork(nn.Module):
    """
    Core score network for the Diffusion Factor Model (DFM).
    Approximates the score function using the factor-aware decomposition:
        s_theta(r, t) = alpha_t * D_t * V * g_zeta(V^T * D_t * r, t) - D_t * r

    Attributes:
        d (int): Asset return dimension.
        k (int): Latent factor dimension (k << d).
        sigma_max (float): Maximum boundary for noise variance c.
        V (nn.Parameter): Trainable matrix of shape (d, k) with orthogonal columns.
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
        """
        Computes the score approximation for a batch of returns r and times t.

        Args:
            r (Tensor): Batch of returns of shape (B, d).
            t (Tensor): Batch of times of shape (B, 1).

        Returns:
            Tensor: Computed score s_theta(r, t) of shape (B, d).
        """
        # B = batch_size
        # 1. Compute alpha_t and h_t
        alpha_t = torch.exp(-t / 2.0)  # Shape: (B, 1)
        h_t = -torch.expm1(-t)          # Shape: (B, 1)

        # 2. Compute noise variance c in [0, sigma_max]
        c = torch.sigmoid(self.c_raw) * self.sigma_max  # Shape: (d,)

        # 3. Compute diagonal terms: denom = h_t + alpha_t^2 * c_i
        # denom shape: (B, d) via broadcasting
        denom = h_t + (alpha_t ** 2) * c.unsqueeze(0)
        D_t = 1.0 / denom  # Shape: (B, d)

        # 4. Perform element-wise multiplication for D_t * r (since D_t is diagonal)
        # D_t_r shape: (B, d)
        D_t_r = D_t * r

        # 5. Project to k-dimensional latent space: V^T * D_t * r
        # projected shape: (B, k)
        projected = torch.matmul(D_t_r, self.V)

        # 6. Concatenate projection and time: shape (B, k + 1)
        g_input = torch.cat([projected, t], dim=-1)

        # 7. Pass through g_zeta: shape (B, k)
        g_out = self.g_zeta(g_input)

        # 8. Reconstruct to d-dimensional space: shape (B, d)
        lifted = torch.matmul(g_out, self.V.t())

        # 9. Compute the first term: alpha_t * D_t * V * g_out
        first_term = alpha_t * D_t * lifted  # Shape: (B, d)

        # 10. Combine with the linear skip shortcut: s_theta(r, t) = first_term - D_t * r
        score = first_term - D_t_r  # Shape: (B, d)

        return score
