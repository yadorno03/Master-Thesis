import importlib.util
import unittest
from unittest.mock import patch
import torch
from conditioning import ConditioningConfig
from author_h import DDPMHNetwork
from author_sampling import sample_author_ddpm
from author_guidance import sample_guided_author
from test_author_reference import small_reference


@unittest.skipUnless(importlib.util.find_spec('diffusion_factor_model'), 'Install requirements-dfm.txt')
class AuthorGuidanceTests(unittest.TestCase):
    def test_zero_scale_preserves_rng_across_uneven_batches(self):
        h = DDPMHNetwork(small_reference(), ConditioningConfig([.25]*4, 0), hidden_dims=(8,))
        expected_rng = torch.Generator().manual_seed(4)
        expected = torch.cat([sample_author_ddpm(h.reference, n, generator=expected_rng)
                              for n in (2, 2, 1)])
        actual_rng = torch.Generator().manual_seed(4)
        with patch.object(h, 'gradient', side_effect=AssertionError('must skip H')):
            actual = sample_guided_author(h, 5, batch_size=2, guidance_scale=0,
                initialization='gaussian', generator=actual_rng)
        torch.testing.assert_close(actual.samples, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(actual_rng.get_state(), expected_rng.get_state()))
        self.assertEqual(actual.initial_states.shape, (5, 4))

    def test_scale_zero_and_single_transition_variance_scaling(self):
        h = DDPMHNetwork(small_reference(3), ConditioningConfig([.25]*4, 0), hidden_dims=(8,))
        expected = sample_author_ddpm(h.reference, 3, generator=torch.Generator().manual_seed(4))
        with patch.object(h, 'gradient', side_effect=AssertionError('must skip H')):
            zero = sample_guided_author(h, 3, guidance_scale=0, initialization='gaussian',
                                        generator=torch.Generator().manual_seed(4))
        torch.testing.assert_close(zero.samples, expected, rtol=0, atol=0)
        for scale in (.5, 1., 2.):
            with patch.object(h, 'gradient', side_effect=lambda i,x: torch.ones_like(x)):
                actual = sample_guided_author(h, 3, guidance_scale=scale, initialization='gaussian',
                                              generator=torch.Generator().manual_seed(4))
            torch.testing.assert_close(actual.samples, expected + scale*h.reference.model.diffusion.posterior_variance[3])
        for bad in (-1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                sample_guided_author(h, 3, guidance_scale=bad)
        with patch.object(h, 'log_prob', side_effect=lambda i,x: x.new_full((len(x),1), -torch.inf)):
            with self.assertRaisesRegex(RuntimeError, 'exhausted 4 proposals'):
                sample_guided_author(h, 3, max_proposals=4)
