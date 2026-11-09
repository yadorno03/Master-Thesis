"""Exact conditioned final step of the guided sampler."""
import math
import unittest

import torch
from scipy.stats import ks_2samp

import test_h_network
from guided_sampling import exact_final_step


class ExactFinalStepTests(unittest.TestCase):
    def setUp(self):
        self.model = test_h_network.HNetworkTests().fixture(torch.float64)

    def test_matches_rejection_of_the_euler_step(self):
        h, steps = self.model, 10
        dt = h.horizon / steps
        x = torch.randn(1, h.reference.d, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
        x = x.repeat(40000, 1)
        exact = exact_final_step(h, x, steps, torch.Generator().manual_seed(5))
        self.assertTrue(bool((exact @ h._w <= h.C + 1e-12).all()))
        # Plain Euler step as in generate_samples, then keep the event rows.
        with torch.no_grad():
            s = torch.full((len(x), 1), (steps - 1) * dt, dtype=x.dtype)
            mean = x + dt * (.5 * x + h.reference.factor_score_network(x, h.reference.T - s))
        euler = mean + math.sqrt(dt) * torch.randn(x.shape, generator=torch.Generator().manual_seed(6),
                                                    dtype=x.dtype)
        kept = euler[euler @ h._w <= h.C]
        self.assertGreater(len(kept), 2000)
        for projection in (h._w, torch.ones(h.reference.d, dtype=x.dtype), torch.arange(h.reference.d, dtype=x.dtype)):
            statistic = ks_2samp((exact @ projection).numpy(), (kept @ projection).numpy()).statistic
            self.assertLess(statistic, .03)

    def test_far_infeasible_states_land_on_the_threshold(self):
        h = self.model
        x = (10 * h._w / h._w.norm() ** 2 * (abs(h.C) + 50)).repeat(5, 1)
        exact = exact_final_step(h, x, 10, torch.Generator().manual_seed(1))
        self.assertTrue(torch.isfinite(exact).all())
        self.assertTrue(bool((exact @ h._w <= h.C + 1e-9).all()))


if __name__ == '__main__':
    unittest.main()
