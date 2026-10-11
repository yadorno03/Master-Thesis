import unittest

import torch

from conditioning import ConditioningConfig, DFMConditioning
from reference import FrozenDFMReference
import test_reference_coordinates


class ConditioningTests(unittest.TestCase):
    def fixture(self, dtype=torch.float64, weights=(.4, -.2, .1, .7)):
        with torch.random.fork_rng():
            torch.manual_seed(19)
            model, _, _ = test_reference_coordinates.ReferenceCoordinateTests().fixture(dtype)
        return DFMConditioning(FrozenDFMReference(model), ConditioningConfig(weights, .2))

    def test_reduced_drift_is_transformed_ambient_drift(self):
        for dtype in (torch.float32, torch.float64):
            c = self.fixture(dtype)
            x = torch.arange(12., dtype=dtype).reshape(3, 4) / 5
            times = torch.tensor([0., .45, 1.], dtype=dtype)
            expected = []
            for t, row in zip(times, x):
                G = c.G(t)
                dG = torch.autograd.functional.jacobian(c.G, t)
                drift = .5*row + c.reference.score(row[None], c.reference.T-t)[0]
                expected.append(dG @ row + G @ drift)
            for t in (times, times[:, None]):
                y = c.features(t, x)
                torch.testing.assert_close(c.mu(t, y), torch.stack(expected))
            torch.testing.assert_close(c.features(.45, x), x @ c.G(.45).T)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in c.parameters()))

    def test_covariance_terminal_event_and_input_gradients(self):
        c = self.fixture()
        t, tau = .3, c.reference.T-c.reference.t0
        x = torch.arange(12., dtype=torch.float64).reshape(3, 4).requires_grad_()
        gradient, = torch.autograd.grad(c.features(t, x).sum(), x)
        torch.testing.assert_close(gradient, c.G(t).sum(0).expand_as(x))
        for time in (t, torch.tensor([0., t, tau], dtype=x.dtype)):
            G = c.G(time)
            torch.testing.assert_close(c.A(time), G @ G.transpose(-1, -2))
            torch.testing.assert_close(c.ell(time), G[..., -1, :])
        y = c.features(tau, x.detach())
        torch.testing.assert_close(y[:, -1], x.detach() @ c.w)
        y[:, -1] = y.new_tensor([c.C-.1, c.C, c.C+.1])
        self.assertEqual(c.terminal_event(y).tolist(), [True, True, False])
        w = c.w
        w.zero_()
        self.assertTrue(bool(c.w.any()))

    def test_residual_portfolios_and_singular_covariance(self):
        for weights, rank in [((0., 0., 1., 0.), 3), ((1., 0., 0., 0.), 2)]:
            c = self.fixture(weights=weights)
            with torch.no_grad():
                c.reference._score.V.copy_(torch.eye(4, 2, dtype=torch.float64))
            self.assertEqual(int(torch.linalg.matrix_rank(c.A(.3))), rank)
            x = torch.ones(2, 4, dtype=torch.float64)
            self.assertTrue(bool(torch.isfinite(c.mu(.3, c.features(.3, x))).all()))
            torch.testing.assert_close(c.features(c.reference.T-c.reference.t0, x)[:, -1], x @ c.w)

    def test_invalid_binding_and_state_batches(self):
        c = self.fixture()
        for ref, config, error in [(None, c.config, TypeError), (c.reference, None, TypeError),
                (c.reference, ConditioningConfig((1.,), 0.), ValueError)]:
            with self.assertRaises(error):
                DFMConditioning(ref, config)
        x = torch.ones(2, 4, dtype=torch.float64)
        for time, state in [([0., .1, .2], x), (.1, x.float()), (.1, x[:, :3]),
                            (.1, x[:0]), (.1, x*float('nan')), (-.1, x)]:
            with self.assertRaises(ValueError):
                c.features(time, state)
        for method in (lambda y: c.mu(.1, y), c.terminal_event):
            with self.assertRaises(ValueError):
                method(x)
