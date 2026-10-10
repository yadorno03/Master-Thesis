import math
import unittest

import torch

from DFM import UnconditionalDFM
from reference import FrozenDFMReference


class ReferenceCoordinateTests(unittest.TestCase):
    def fixture(self, dtype=torch.float64):
        model = UnconditionalDFM(4, 2, sigma_max=2.0, t0=0.2, T=1.4,
                                 hidden_dims=[4]).to(dtype)
        basis, _ = torch.linalg.qr(torch.tensor(
            [[1., 2.], [2., -1.], [1., 1.], [-1., 2.]], dtype=dtype))
        variances = torch.tensor([0.15, 0.4, 0.8, 1.5], dtype=dtype)
        with torch.no_grad():
            model.factor_score_network.V.copy_(basis)
            model.factor_score_network.c_raw.copy_(torch.logit(variances / 2))
        return model, basis, variances

    def test_dense_formula_and_time_shapes(self):
        for dtype in (torch.float32, torch.float64):
            model, basis, variances = self.fixture(dtype)
            ref = FrozenDFMReference(model)
            w = torch.tensor([0.4, -0.2, 0.1, 0.7], dtype=dtype)
            times = torch.tensor([0., 0.45, ref.T-ref.t0], dtype=dtype)
            identity = torch.eye(4, dtype=dtype)
            sigma = torch.diag(variances)
            lambda0 = (1-math.exp(-ref.t0))*identity + math.exp(-ref.t0)*sigma
            expected = []
            for t in times:
                s = ref.T-float(t)
                covariance = (1-math.exp(-s))*identity + math.exp(-s)*sigma
                factors = torch.linalg.solve(covariance, basis).T
                ell = math.exp((ref.t0-s)/2)*torch.linalg.solve(covariance, lambda0 @ w)
                expected.append(torch.cat((factors, ell[None, :]), dim=0))
            expected = torch.stack(expected)
            for time in (times, times[:, None]):
                actual = ref.G(time, w)
                self.assertEqual(actual.shape, (3, 3, 4))
                torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(ref.G(0., w), expected[0])
            self.assertEqual(ref.G([0.], w).shape, (1, 3, 4))
            torch.testing.assert_close(ref.G(ref.T-ref.t0, w)[-1], w)

    def test_coordinate_derivative_and_input_gradient(self):
        model, _, variances = self.fixture()
        ref = FrozenDFMReference(model)
        w = torch.tensor([0.4, -0.2, 0.1, 0.7], dtype=torch.float64)
        t = torch.tensor(0.45, dtype=torch.float64, requires_grad=True)
        matrix = ref.G(t, w)
        derivative = torch.autograd.functional.jacobian(lambda time: ref.G(time, w), t)
        s = ref.T-t.detach()
        diagonal = (1-torch.exp(-s)+torch.exp(-s)*variances).reciprocal()
        expected = torch.cat((matrix[:-1]*(diagonal-1),
                              matrix[-1:]*(diagonal-0.5)))
        torch.testing.assert_close(derivative, expected)
        x = torch.arange(8., dtype=torch.float64).reshape(2, 4).requires_grad_()
        cotangent = torch.tensor([[1., 2., -1.], [-2., 1., 3.]], dtype=x.dtype)
        gradient, = torch.autograd.grad(((x @ matrix.T)*cotangent).sum(), x)
        torch.testing.assert_close(gradient, cotangent @ matrix)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in ref.parameters()))

    def test_residual_portfolio_and_redundant_coordinates(self):
        model, _, _ = self.fixture()
        with torch.no_grad():
            model.factor_score_network.V.copy_(torch.eye(4, 2, dtype=torch.float64))
        ref = FrozenDFMReference(model)
        x = torch.tensor([[1., 2., 3., 4.]], dtype=torch.float64)
        residual_w = torch.tensor([0., 0., 1., -0.5], dtype=x.dtype)
        terminal = ref.G(ref.T-ref.t0, residual_w)
        torch.testing.assert_close((x @ terminal.T)[:, -1], x @ residual_w)
        self.assertGreater(abs(float(x @ residual_w)), 0.)
        redundant_w = torch.tensor([1., 0., 0., 0.], dtype=x.dtype)
        matrix = ref.G(0.45, redundant_w)
        self.assertEqual(int(torch.linalg.matrix_rank(matrix)), 2)
        self.assertTrue(torch.isfinite(matrix).all())

    def test_invalid_portfolios_and_generative_times(self):
        model, _, _ = self.fixture()
        ref = FrozenDFMReference(model)
        w = torch.ones(4, dtype=torch.float64)
        for invalid in (torch.zeros_like(w), w.float(), w[:3], w[None, :], w*float('nan')):
            with self.assertRaises(ValueError):
                ref.G(0., invalid)
        for time in (-0.01, ref.T, float('nan'), True, 1j, [], [[0., 0.]]):
            with self.assertRaises(ValueError):
                ref.G(time, w)
