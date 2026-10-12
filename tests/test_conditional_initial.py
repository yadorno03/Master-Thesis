import math
import unittest
from unittest.mock import patch

import torch

from DFM import UnconditionalDFM
from conditioning import ConditioningConfig, DFMConditioning
from guided_sampling import sample_conditional_initial
from reference import FrozenDFMReference
import test_h_network


class GaussianH(torch.nn.Module):
    """Exact event probability for stationary OU with score=-x."""

    def __init__(self):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(51)
            model = UnconditionalDFM(3, 1, sigma_max=2., T=1.1, t0=.1,
                                     hidden_dims=[8]).double()
        with torch.no_grad():
            model.factor_score_network.c_raw.zero_()
            for p in model.factor_score_network.g_zeta.parameters():
                p.zero_()
        self.conditioning = DFMConditioning(
            FrozenDFMReference(model), ConditioningConfig((.6, -.3, .7), -.3))

    def log_prob(self, t, y):
        c = self.conditioning
        remaining = c.reference.T - c.reference.t0 - t
        variance = -math.expm1(-remaining) * c.w.square().sum()
        return torch.special.log_ndtr((c.C - y[:, -1:]) / variance.sqrt())


class ConditionalInitialTests(unittest.TestCase):
    def test_exact_h_initial_mean_covariance_and_acceptance(self):
        h = GaussianH()
        c = h.conditioning
        x, info = sample_conditional_initial(
            h, 16000, batch_size=2048, generator=torch.Generator().manual_seed(23))
        sigma = float(c.w.norm())
        a = c.C / sigma
        q = .5 * math.erfc(-a / math.sqrt(2))
        mills = math.exp(-a*a/2) / math.sqrt(2*math.pi) / q
        direction = c.w / sigma
        expected_mean = -math.exp(-.5) * direction * mills
        expected_cov = torch.eye(3, dtype=x.dtype) - math.exp(-1) * (
            a*mills + mills*mills) * torch.outer(direction, direction)
        torch.testing.assert_close(x.mean(0), expected_mean, atol=.025, rtol=0)
        torch.testing.assert_close(x.T.cov(), expected_cov, atol=.035, rtol=0)
        self.assertAlmostEqual(info['acceptance_rate'], q, delta=.015)
        self.assertEqual(info['returned'], len(x))
        self.assertGreaterEqual(info['accepted'], len(x))
        self.assertFalse(x.requires_grad)

    def test_seed_dtype_modes_and_overshoot_accounting(self):
        for dtype in (torch.float32, torch.float64):
            h = test_h_network.HNetworkTests().fixture(dtype)
            h.network[1].eval()
            modes = [m.training for m in h.modules()]
            with patch.object(h, 'log_prob', side_effect=lambda t, y: y.new_zeros(len(y), 1)):
                results = [sample_conditional_initial(h, 3, batch_size=5,
                    generator=torch.Generator().manual_seed(4)) for _ in range(2)]
            torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
            x, info = results[0]
            self.assertEqual((x.shape, x.dtype), ((3, 4), dtype))
            self.assertEqual((info['proposals'], info['accepted'], info['returned']), (5, 5, 3))
            self.assertEqual(modes, [m.training for m in h.modules()])
            self.assertTrue(all(p.grad is None for p in h.parameters()))

    def test_budget_exhaustion_invalid_probabilities_and_mode_restoration(self):
        h = test_h_network.HNetworkTests().fixture()
        modes = [m.training for m in h.modules()]
        for value, error in [(float('-inf'), RuntimeError), (float('nan'), ValueError),
                             (.1, ValueError)]:
            with patch.object(h, 'log_prob', side_effect=lambda t, y: y.new_full((len(y), 1), value)):
                with self.assertRaises(error):
                    sample_conditional_initial(h, 2, batch_size=4, max_proposals=10)
            self.assertEqual(modes, [m.training for m in h.modules()])
        for kwargs in ({'num_samples': 0}, {'num_samples': True},
                       {'num_samples': 2, 'batch_size': 0},
                       {'num_samples': 2, 'max_proposals': 1}):
            with self.assertRaises(ValueError):
                sample_conditional_initial(h, **kwargs)
