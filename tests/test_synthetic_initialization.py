"""Exercise the benchmark's H-weighted starts, not just the shared sampler."""
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy.stats import norm, wasserstein_distance

from guided_sampling import DFMHGuidance
from synthetic_benchmark.conditional import sample_h_guided
from test_conditional_initial import BatchedGaussianH


class SyntheticInitializationTests(unittest.TestCase):
    def test_exact_h_recovers_conditional_portfolio_without_terminal_filtering(self):
        h = BatchedGaussianH()
        before = {key: value.clone() for key, value in h.state_dict().items()}
        modes = [module.training for module in h.modules()]
        rng_before = torch.get_rng_state().clone()
        diagnostics = {}
        samples = sample_h_guided(
            h, DFMHGuidance(h), n_samples=6000, batch_size=1000,
            steps=300, seed=171, diagnostics=diagnostics,
        )
        portfolio = samples @ h.w.numpy()
        sigma = float(h.w.norm())
        probability = norm.cdf(h.C / sigma)
        truth = sigma * norm.ppf(np.random.default_rng(172).uniform(0, probability, 6000))
        exact_mean = -sigma * norm.pdf(h.C / sigma) / probability
        # Gaussian starts produced W1=0.103 and a mean error of 0.101 with exact H.
        self.assertLess(wasserstein_distance(portfolio, truth), .04)
        self.assertLess(abs(portfolio.mean() - exact_mean), .04)
        self.assertEqual(samples.shape, (6000, 3))
        self.assertTrue(np.any(portfolio > h.C))  # Keep finite-step event violations.
        self.assertFalse(diagnostics['terminal_filtering'])
        self.assertEqual(diagnostics['initialization']['method'], 'h_rejection')
        self.assertEqual(diagnostics['initialization']['returned'], 6000)
        self.assertEqual(diagnostics['initialization']['max_proposals'], 600000)
        self.assertLessEqual(diagnostics['initialization']['proposals'], 600000)
        self.assertEqual(modes, [module.training for module in h.modules()])
        torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
        for key, value in h.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in h.parameters()))

    def test_initialization_budget_is_global_and_exhaustion_does_not_sample(self):
        h = BatchedGaussianH()
        sizes = []

        def rare_events(t, y, **kwargs):
            sizes.append(len(y))
            # One accepted start per proposal batch, too few for the request.
            result = y.new_full((len(y), 1), -math.inf)
            result[0] = 0.
            return result

        with patch.object(h, 'log_prob', side_effect=rare_events), \
             patch('synthetic_benchmark.conditional.generate_samples') as integrate:
            with self.assertRaisesRegex(RuntimeError, 'exhausted 7 proposals'):
                sample_h_guided(h, DFMHGuidance(h), n_samples=5, batch_size=3,
                                steps=2, seed=71, initialization_max_proposals=7)
            integrate.assert_not_called()
        self.assertEqual(sizes, [3, 3, 1])

    def test_seed_reproduces_samples_and_initialization_diagnostics(self):
        h = BatchedGaussianH()
        runs, diagnostics = [], []
        for _ in range(2):
            info = {}
            runs.append(sample_h_guided(
                h, DFMHGuidance(h), n_samples=13, batch_size=5,
                steps=8, seed=92, initialization_max_proposals=1000, diagnostics=info,
            ))
            diagnostics.append(info)
        np.testing.assert_array_equal(*runs)
        self.assertEqual(*diagnostics)
        info = diagnostics[0]['initialization']
        self.assertEqual(info['returned'], 13)
        self.assertGreaterEqual(info['accepted'], 13)
        self.assertGreaterEqual(info['proposals'], info['accepted'])
        self.assertEqual(info['max_proposals'], 1000)

    def test_invalid_sampling_settings_fail_before_initialization(self):
        h = BatchedGaussianH()
        settings = dict(n_samples=5, batch_size=3, steps=2, seed=71)
        with patch('synthetic_benchmark.conditional.sample_conditional_initial') as initialize:
            for key, value in (('n_samples', 0), ('batch_size', True), ('steps', 0),
                               ('seed', -1), ('guidance_scale', float('nan'))):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    sample_h_guided(h, DFMHGuidance(h), **dict(settings, **{key: value}))
            initialize.assert_not_called()


if __name__ == '__main__':
    unittest.main()
