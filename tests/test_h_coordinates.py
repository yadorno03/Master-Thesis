import unittest

import torch

from conditioning import ConditioningConfig
from h_model import HNetwork
from reference import FrozenDFMReference
import test_reference_coordinates


class HCoordinateTests(unittest.TestCase):
    def fixture(self, dtype=torch.float64, weights=(.4, -.2, .1, .7)):
        with torch.random.fork_rng():
            torch.manual_seed(19)
            model, _, _ = test_reference_coordinates.ReferenceCoordinateTests().fixture(dtype)
            return HNetwork(FrozenDFMReference(model), ConditioningConfig(weights, .2))

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
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in c.reference.parameters()))

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
                HNetwork(ref, config)
        x = torch.ones(2, 4, dtype=torch.float64)
        for time, state in [([0., .1, .2], x), (.1, x.float()), (.1, x[:, :3]),
                            (.1, x[:0]), (.1, x*float('nan')), (-.1, x)]:
            with self.assertRaises(ValueError):
                c.features(time, state)
        for method in (lambda y: c.mu(.1, y), c.terminal_event):
            with self.assertRaises(ValueError):
                method(x)

    def test_direct_binding_architecture_and_frozen_modes(self):
        for dtype in (torch.float32, torch.float64):
            h = self.fixture(dtype)
            self.assertEqual(set(h._modules), {"reference", "network"})
            self.assertEqual(set(h._buffers), {"_w"})
            self.assertEqual(h.hidden_dims, (32, 32))
            self.assertEqual(h.horizon, h.reference.T-h.reference.t0)
            self.assertEqual(h.network[0].in_features, 4)
            self.assertEqual(h.network[-1].out_features, 1)
            self.assertEqual(h.w.dtype, dtype)
            self.assertTrue(all(p.dtype == dtype for p in h.parameters()))
            trainable = {id(p) for p in h.parameters() if p.requires_grad}
            self.assertEqual(trainable, {id(p) for p in h.network.parameters()})
            h.train()
            self.assertTrue(h.network.training)
            self.assertFalse(any(m.training for m in h.reference.modules()))
            self.assertIn("_w", h.state_dict())
            self.assertTrue(any(key.startswith("reference.") for key in h.state_dict()))
            with self.assertRaises(AttributeError):
                h.config = ConditioningConfig((1., 0., 0., 0.), 0.)

    def test_constructor_precision_and_architecture_validation(self):
        h = self.fixture(torch.float32)
        for weights, threshold in [((1e100, 0., 0., 0.), 0.),
                                   ((1e-100, 0., 0., 0.), 0.),
                                   ((1., 0., 0., 0.), 1e100)]:
            with self.assertRaises(ValueError):
                HNetwork(h.reference, ConditioningConfig(weights, threshold))
        for widths in ((), (0,), (-1,), (True,), (2.5,)):
            with self.assertRaises(ValueError):
                HNetwork(h.reference, h.config, hidden_dims=widths)

    def test_initialization_matches_standalone_mlp_and_dtype_conversion(self):
        h = self.fixture()
        with torch.random.fork_rng():
            torch.manual_seed(91)
            expected = torch.nn.Sequential(torch.nn.Linear(4, 7), torch.nn.Tanh(),
                torch.nn.Linear(7, 5), torch.nn.Tanh(), torch.nn.Linear(5, 1)).double()
            expected_rng = torch.get_rng_state()
            torch.manual_seed(91)
            actual = HNetwork(h.reference, h.config, hidden_dims=(7, 5))
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(actual.network.state_dict()[key], value, rtol=0, atol=0)
        actual.float()
        self.assertEqual(actual.w.dtype, torch.float32)
        self.assertTrue(all(p.dtype == torch.float32 for p in actual.parameters()))
        x = torch.ones(2, 4)
        torch.testing.assert_close(actual.features(actual.horizon, x)[:, -1], x @ actual.w)
