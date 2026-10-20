import importlib.util
import unittest
import torch
from conditioning import ConditioningConfig
from author_h import DDPMHNetwork
from test_author_reference import small_reference


@unittest.skipUnless(importlib.util.find_spec('diffusion_factor_model'), 'Install requirements-dfm.txt')
class DDPMHTests(unittest.TestCase):
    def test_full_state_gradient_against_finite_differences(self):
        h = DDPMHNetwork(small_reference(), ConditioningConfig([.25]*4, .1), hidden_dims=(8,))
        x = h.reference.normal(2)
        gradient = h.gradient(2, x)
        for column in range(4):
            delta = torch.zeros_like(x); delta[:, column] = .001
            numerical = (h.log_prob(2, x+delta) - h.log_prob(2, x-delta))[:,0] / .002
            torch.testing.assert_close(gradient[:,column], numerical, atol=1e-4, rtol=1e-2)
        self.assertTrue(all(p.grad is None for p in h.parameters()))
        h.train()
        self.assertFalse(h.reference.training)
        self.assertTrue(torch.equal(h.terminal_event(x), x.mean(1) <= .1))
        with self.assertRaises(ValueError):
            h.log_prob(0, x)  # hard terminal endpoint is excluded
