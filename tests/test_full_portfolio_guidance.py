import unittest
from unittest.mock import patch

import torch

from guided_sampling import DFMHGuidance
from h_model import HNetwork
from sampling import generate_samples
import test_h_coordinates
import test_h_network


class FullPortfolioGuidanceTests(unittest.TestCase):
    def test_chain_rule_and_modes(self):
        for dtype in (torch.float32, torch.float64):
            h = test_h_network.HNetworkTests().fixture(dtype)
            h.network[1].eval()  # preserve mixed modes, not just the root flag
            modes = [m.training for m in h.modules()]
            c = h
            x = torch.arange(12., dtype=dtype).reshape(3, 4).requires_grad_()
            for t in (.3, torch.tensor([.1, .3, .7], dtype=dtype)):
                expected, = torch.autograd.grad(h.log_prob(t, c.features(t, x)).sum(), x)
                with torch.no_grad():
                    guide = DFMHGuidance(h)
                    ambient = guide(x, t)
                torch.testing.assert_close(ambient, expected)
                self.assertFalse(ambient.requires_grad)
            self.assertEqual(modes, [m.training for m in h.modules()])
            self.assertTrue(all(p.grad is None for p in h.parameters()))

    def test_residual_and_redundant_portfolios_have_analytic_gradient(self):
        for weights, rank in [((0., 0., 1., 0.), 3), ((1., 0., 0., 0.), 2)]:
            c = test_h_coordinates.HCoordinateTests().fixture(weights=weights)
            with torch.no_grad():
                c.reference._score.V.copy_(torch.eye(4, 2, dtype=torch.float64))
            h = HNetwork(c.reference, c.config, hidden_dims=(2,))
            x = torch.ones(2, 4, dtype=torch.float64)
            t = .3
            y = c.features(t, x)
            self.assertEqual(int(torch.linalg.matrix_rank(c.A(t))), rank)
            with patch.object(h, 'log_prob', side_effect=lambda t, y: -torch.nn.functional.softplus(y[:, -1:])):
                guide = DFMHGuidance(h)
                expected = -torch.sigmoid(y[:, -1:]) * c.ell(t)
                torch.testing.assert_close(guide(x, t), expected)
                self.assertGreater(float(expected.abs().sum()), 0.)

    def test_sampler_uses_elapsed_time_and_ambient_correction(self):
        h = test_h_network.HNetworkTests().fixture()
        c = h
        x = torch.ones(2, c.reference.d, dtype=torch.float64)
        dw = torch.zeros(2, *x.shape, dtype=x.dtype)
        dt = h.horizon / 2
        expected = x.clone()
        guide = DFMHGuidance(h)
        for t in (0., dt):
            expected = expected + dt * (.5 * expected + c.reference.score(
                expected, c.reference.T - t) + guide(expected, t))
        actual = generate_samples(c.reference, 2, num_steps=2, initial_state=x,
                                  brownian_increments=dw, guidance_fn=guide)
        torch.testing.assert_close(actual, expected)

    def test_invalid_probabilities_boundary_and_autograd_context(self):
        h = test_h_network.HNetworkTests().fixture()
        x = torch.zeros(2, h.reference.d, dtype=torch.float64)
        guide = DFMHGuidance(h)
        modes = [m.training for m in h.modules()]
        for value in (float('nan'), float('-inf'), .1):
            with patch.object(h, 'log_prob', return_value=x.new_full((2, 1), value)):
                with self.assertRaisesRegex(ValueError, 'log_prob'):
                    guide(x, .2)
            self.assertEqual(modes, [m.training for m in h.modules()])
        with patch.object(h, 'log_prob', return_value=x.new_zeros(2, 1)):
            with self.assertRaisesRegex(ValueError, 'differentiably connected'):
                guide(x, .2)
        with self.assertRaisesRegex(ValueError, 'preterminal'):
            guide(x, h.horizon)
        with torch.inference_mode(), self.assertRaisesRegex(ValueError, 'autograd'):
            guide(x, .2)
        with self.assertRaises(TypeError):
            DFMHGuidance(None)
