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


if __name__ == "__main__":
    unittest.main()
