import unittest

import torch

from NeuralNets import FactorScoreNetwork


class FactorScoreNetworkTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = FactorScoreNetwork(4, 2, hidden_dims=[8]).double()
        self.x = torch.randn(3, 4, dtype=torch.float64)
        self.t = torch.tensor([[0.01], [0.4], [1.0]], dtype=torch.float64)

    def test_score_matches_dense_matrix_formula_and_input_gradient(self):
        self.x.requires_grad_()
        rows = []
        variances = self.model.c_raw.sigmoid() * self.model.sigma_max
        for x, t in zip(self.x, self.t):
            alpha = torch.exp(-t / 2)
            covariance = torch.diag(-torch.expm1(-t) + alpha.square() * variances)
            precision = torch.linalg.inv(covariance)
            features = self.model.V.T @ precision @ x
            factor = self.model.g_zeta(torch.cat((features, t)))
            rows.append(alpha * precision @ self.model.V @ factor - precision @ x)
        expected = torch.stack(rows)
        actual = self.model(self.x, self.t)
        torch.testing.assert_close(actual, expected)
        actual_grad, = torch.autograd.grad(actual.square().sum(), self.x)
        expected_grad, = torch.autograd.grad(expected.square().sum(), self.x)
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_zero_factor_network_recovers_gaussian_score(self):
        with torch.no_grad():
            for parameter in self.model.g_zeta.parameters():
                parameter.zero_()
        variance = -torch.expm1(-self.t) + self.t.neg().exp() * self.model.c_raw.sigmoid()
        torch.testing.assert_close(self.model(self.x, self.t), -self.x / variance)

    def test_projection_restores_orthonormal_loading_columns(self):
        with torch.no_grad():
            self.model.V.add_(torch.randn_like(self.model.V))
        self.model.project_orthogonal()
        torch.testing.assert_close(self.model.V.T @ self.model.V,
                                   torch.eye(2, dtype=torch.float64))

    def test_projection_preserves_an_already_orthonormal_basis_and_score(self):
        # Signs are part of the learned coordinates, not a free change of basis
        # unless the input/output layers of g are transformed as well.
        with torch.no_grad():
            q, _ = torch.linalg.qr(self.model.V)
            self.model.V.copy_(q * q.new_tensor([-1., 1.]))
            before = self.model.V.clone()
            score = self.model(self.x, self.t).clone()
        self.model.project_orthogonal()
        torch.testing.assert_close(self.model.V, before, rtol=0, atol=1e-12)
        torch.testing.assert_close(self.model(self.x, self.t), score, rtol=0, atol=1e-12)

    def test_projection_is_continuous_near_an_orthonormal_basis(self):
        with torch.no_grad():
            q, _ = torch.linalg.qr(self.model.V)
            before = q * q.new_tensor([-1., 1.])
            perturbation = torch.randn_like(before) * 1e-7
            self.model.V.copy_(before + perturbation)
        self.model.project_orthogonal()
        self.assertLess(float((self.model.V - before).norm()),
                        5 * float(perturbation.norm()))

    def test_gaussian_factor_posterior_recovers_independent_dense_normal_score(self):
        # A known Gaussian factor posterior plugs into equation (18). Compare
        # its ambient score to the full Gaussian covariance inverse, rather
        # than to another transcription of the same encoder/decoder formula.
        V = self.model.V.detach().clone()
        c = V.new_tensor([.05, .2, .8, .4])
        factor_cov = V.new_tensor([[2., .4], [.4, 1.]])
        factor_mean = V.new_tensor([.2, -.3])

        class ExactPosterior(torch.nn.Module):
            def forward(self, inputs):
                z, t = inputs[:, :-1], inputs[:, -1:]
                alpha = torch.exp(-t / 2)
                precision = (-torch.expm1(-t) + alpha.square() * c).reciprocal()
                inverse_prior = torch.linalg.inv(factor_cov)
                inverse_posterior = (inverse_prior + alpha.square().unsqueeze(-1)
                    * (V.T.unsqueeze(0) @ (precision.unsqueeze(-1) * V)))
                rhs = alpha * z + inverse_prior @ factor_mean
                return torch.linalg.solve(inverse_posterior, rhs.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            self.model.c_raw.copy_(torch.logit(c))
        self.model.g_zeta = ExactPosterior()
        covariance = V @ factor_cov @ V.T + torch.diag(c)
        mean = V @ factor_mean
        alpha = torch.exp(-self.t / 2)
        marginal_cov = (alpha.square().unsqueeze(-1) * covariance
                        + (-torch.expm1(-self.t)).unsqueeze(-1) * torch.eye(4, dtype=V.dtype))
        expected = -torch.linalg.solve(marginal_cov,
            (self.x - alpha * mean).unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(self.model(self.x, self.t), expected, rtol=1e-11, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
