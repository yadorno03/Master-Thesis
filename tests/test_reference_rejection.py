import math
import unittest
from unittest.mock import patch

import torch

from guided_sampling import sample_reference_conditional
from test_conditional_initial import GaussianH


class ReferenceRejectionTests(unittest.TestCase):
    def test_finite_grid_gaussian_terminal_and_initial_moments(self):
        h = GaussianH()
        steps = 2  # coarse grid separates the Euler target from continuous-time OU
        dt = h.horizon / steps
        decay = 1 - dt / 2
        variance = decay**(2*steps) + dt * sum(decay**(2*j) for j in range(steps))
        sigma = math.sqrt(variance)
        a = h.C / (sigma * float(h.w.norm()))
        q = .5 * math.erfc(-a / math.sqrt(2))
        mills = math.exp(-a*a/2) / math.sqrt(2*math.pi) / q
        direction = h.w / h.w.norm()
        outer = torch.outer(direction, direction)
        identity = torch.eye(3, dtype=h.w.dtype)
        result = sample_reference_conditional(
            h, 16000, num_steps=steps, batch_size=2048,
            generator=torch.Generator().manual_seed(47))
        for draws, scale, base_variance in (
                (result.samples, sigma, variance),
                (result.initial_states, decay**steps / sigma, 1.)):
            expected_mean = -scale * direction * mills
            expected_cov = base_variance * identity - scale**2 * (a*mills + mills**2) * outer
            torch.testing.assert_close(draws.mean(0), expected_mean, atol=.025, rtol=0)
            torch.testing.assert_close(draws.T.cov(), expected_cov, atol=.04, rtol=0)
        self.assertTrue(bool((result.samples @ h.w <= h.C).all()))
        self.assertAlmostEqual(result.diagnostics['acceptance_rate'], q, delta=.015)

    def test_paired_selection_budget_tail_and_overshoot_without_h(self):
        h = GaussianH()
        starts = torch.arange(27., dtype=h.w.dtype).reshape(9, 3)
        portfolios = starts.new_tensor([-1., 1., 1., -2., 1., 1., -3., -4., 1.])
        terminals = portfolios[:, None] * h.w / h.w.square().sum()
        with patch('guided_sampling._normal', side_effect=[starts[:5], starts[5:]]), \
                patch('guided_sampling.generate_samples',
                      side_effect=[terminals[:5], terminals[5:]]) as sampler, \
                patch.object(h, 'log_prob', side_effect=AssertionError('H must not be used')):
            result = sample_reference_conditional(
                h, 3, num_steps=7, batch_size=5, max_proposals=9)
        torch.testing.assert_close(result.samples, terminals[[0, 3, 6]])
        torch.testing.assert_close(result.initial_states, starts[[0, 3, 6]])
        info = result.diagnostics
        self.assertEqual((info['proposals'], info['accepted'], info['returned']), (9, 4, 3))
        self.assertEqual((info['num_steps'], info['acceptance_rate']), (7, 4/9))
        self.assertEqual([call.args[1] for call in sampler.call_args_list], [5, 4])
        self.assertTrue(all(call.args[0] is h.reference for call in sampler.call_args_list))
        self.assertTrue(all(call.kwargs['num_steps'] == 7 for call in sampler.call_args_list))

    def test_seed_dtype_modes_and_frozen_reference(self):
        for dtype in (torch.float32, torch.float64):
            h = GaussianH().to(dtype=dtype)
            h.network[1].eval()
            modes = [m.training for m in h.modules()]
            before = {name: value.clone() for name, value in h.state_dict().items()}
            with patch.object(h, 'log_prob', side_effect=AssertionError('H must not be used')):
                results = [sample_reference_conditional(
                    h, 8, num_steps=4, batch_size=16,
                    generator=torch.Generator().manual_seed(91)) for _ in range(2)]
            for field in ('samples', 'initial_states'):
                draws = getattr(results[0], field)
                torch.testing.assert_close(draws, getattr(results[1], field), rtol=0, atol=0)
                self.assertEqual((draws.shape, draws.dtype), ((8, 3), dtype))
                self.assertFalse(draws.requires_grad)
            self.assertEqual(results[0].diagnostics, results[1].diagnostics)
            self.assertEqual(modes, [m.training for m in h.modules()])
            for name, value in h.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in h.parameters()))

    def test_validation_exhaustion_and_sampling_failure(self):
        h = GaussianH()
        modes = [m.training for m in h.modules()]
        with self.assertRaises(TypeError):
            sample_reference_conditional(None, 2)
        for kwargs in ({'num_samples': 0}, {'num_samples': True},
                       {'num_samples': 2, 'num_steps': 0},
                       {'num_samples': 2, 'num_steps': True},
                       {'num_samples': 2, 'batch_size': 0},
                       {'num_samples': 2, 'max_proposals': 1}):
            with self.assertRaises(ValueError):
                sample_reference_conditional(h, **kwargs)
        with patch('guided_sampling.generate_samples',
                   side_effect=lambda ref, n, **kw: h.w.new_zeros(n, 3)):
            with self.assertRaisesRegex(RuntimeError, 'exhausted 7 proposals: accepted 0/2'):
                sample_reference_conditional(h, 2, batch_size=4, max_proposals=7)
        with patch.object(h.reference, 'factor_score_network', side_effect=RuntimeError('score')):
            with self.assertRaisesRegex(RuntimeError, 'score'):
                sample_reference_conditional(h, 2, batch_size=4)
        self.assertEqual(modes, [m.training for m in h.modules()])
