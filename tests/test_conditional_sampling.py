import unittest
from unittest.mock import patch

import torch
from scipy.stats import wasserstein_distance

from guided_evaluation import conditional_diagnostics
from guided_sampling import DFMHGuidance, generate_conditional_samples, sample_reference_conditional
from sampling import generate_samples
from test_conditional_initial import GaussianH


class BatchedGaussianH(GaussianH):
    def log_prob(self, t, y):
        t = self._batch(t, y, 2)
        variance = -torch.expm1(t - self.horizon) * self.w.square().sum()
        return torch.special.log_ndtr((self.C - y[:, -1:]) / variance.sqrt())


class ConditionalSamplingTests(unittest.TestCase):
    def test_guided_distribution_against_independent_rejection_without_filtering(self):
        h = BatchedGaussianH()
        reference = sample_reference_conditional(
            h, 3500, num_steps=160, batch_size=2048,
            generator=torch.Generator().manual_seed(40))
        result = generate_conditional_samples(
            h, 3500, num_steps=160, batch_size=2048,
            generator=torch.Generator().manual_seed(41))
        x, target = result.samples, reference.samples
        self.assertEqual(x.shape, (3500, 3))
        rate = float((x @ h.w <= h.C).double().mean())
        self.assertEqual(result.diagnostics['event_rate'], rate)
        self.assertGreater(rate, .97)
        self.assertLess(rate, 1.)  # Finite-grid violations must remain observable.
        self.assertFalse(result.diagnostics['terminal_filtering'])
        self.assertLess(wasserstein_distance(x @ h.w, target @ h.w), .045)
        self.assertLess(float((x.mean(0) - target.mean(0)).square().mean().sqrt()), .05)
        self.assertLess(float((x.T.cov() - target.T.cov()).square().mean().sqrt()), .07)
        report = conditional_diagnostics(x.numpy(), target.numpy(), h.w.numpy(), h.C)
        self.assertEqual(report['samples'], 3500)
        self.assertEqual(report['event_rate'], rate)
        self.assertEqual(report['benchmark_event_rate'], 1.)
        self.assertAlmostEqual(report['portfolio_wasserstein'],
                               wasserstein_distance(x @ h.w, target @ h.w))
        self.assertLess(report['mean_vector_rmse'], .05)
        self.assertLess(report['covariance_rmse'], .07)

    def test_supplied_noise_batching_trajectories_timing_and_state_preservation(self):
        for dtype in (torch.float32, torch.float64):
            h = BatchedGaussianH().to(dtype=dtype)
            h.network[1].eval()
            modes = [m.training for m in h.modules()]
            saved = {k: v.clone() for k, v in h.state_dict().items()}
            x0 = torch.randn(7, 3, dtype=dtype, generator=torch.Generator().manual_seed(1))
            dw = torch.randn(8, 7, 3, dtype=dtype, generator=torch.Generator().manual_seed(2)) / 8**.5
            expected = generate_samples(h.reference, 7, num_steps=8, initial_state=x0,
                                        brownian_increments=dw, return_trajectory=True,
                                        guidance_fn=DFMHGuidance(h))
            with patch('guided_sampling.time.perf_counter', side_effect=[10., 12., 13., 16.]):
                result = generate_conditional_samples(
                    h, 7, num_steps=8, initial_state=x0, batch_size=3,
                    brownian_increments=dw, return_trajectory=True)
            torch.testing.assert_close(result.samples, expected)
            torch.testing.assert_close(result.initial_states, x0)
            self.assertNotEqual(result.initial_states.data_ptr(), x0.data_ptr())
            self.assertEqual(result.diagnostics['initialization']['method'], 'supplied')
            self.assertEqual(result.diagnostics['timing'],
                             {'initialization_seconds': 2., 'integration_seconds': 3.})
            self.assertEqual(modes, [m.training for m in h.modules()])
            for key, value in h.state_dict().items():
                torch.testing.assert_close(value, saved[key], rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in h.parameters()))
            self.assertFalse(result.samples.requires_grad)

    def test_initialization_choices_reproducibility_and_failures(self):
        h = BatchedGaussianH()
        for method in ('h_rejection', 'reference_rejection', 'gaussian'):
            results = [generate_conditional_samples(
                h, 12, num_steps=8, initialization=method, batch_size=5,
                return_trajectory=True, generator=torch.Generator().manual_seed(92))
                for _ in range(2)]
            torch.testing.assert_close(results[0].samples, results[1].samples, rtol=0, atol=0)
            torch.testing.assert_close(results[0].samples[:, 0], results[0].initial_states)
            info = results[0].diagnostics['initialization']
            self.assertEqual(info['method'], method)
            if method != 'gaussian':
                self.assertGreaterEqual(info['proposals'], info['accepted'])
                self.assertEqual(info['returned'], 12)
        for kwargs in ({'num_steps': 0}, {'batch_size': 0}, {'initialization': 'unknown'},
                       {'return_trajectory': 1}, {'initial_state': h.w.new_zeros(1, 3)},
                       {'brownian_increments': h.w.new_zeros(100, 1, 3)}):
            with self.assertRaises(ValueError):
                generate_conditional_samples(h, 2, **kwargs)
        modes = [m.training for m in h.modules()]
        with patch.object(h, 'log_prob', side_effect=lambda t, y: y.new_full((len(y), 1), -torch.inf)):
            with self.assertRaisesRegex(RuntimeError, 'exhausted 4 proposals'):
                generate_conditional_samples(h, 2, batch_size=3, max_proposals=4)
        with patch.object(h.reference, 'factor_score_network', side_effect=RuntimeError('score')):
            with self.assertRaisesRegex(RuntimeError, 'score'):
                generate_conditional_samples(h, 2, initialization='gaussian')
        self.assertEqual(modes, [m.training for m in h.modules()])
