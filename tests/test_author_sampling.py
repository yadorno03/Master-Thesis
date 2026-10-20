import importlib.util
import unittest
import torch
from author_sampling import sample_author_ddpm
from test_author_reference import small_reference


@unittest.skipUnless(importlib.util.find_spec('diffusion_factor_model'), 'Install requirements-dfm.txt')
class AuthorSamplingTests(unittest.TestCase):
    def test_saved_endpoint_and_successive_batches_match_upstream_bitwise(self):
        ref = small_reference()
        torch.manual_seed(83)
        expected = [ref.model.diffusion.sample(batch_size=3, save_timesteps=[1]).flatten(1)
                    for _ in range(2)]
        expected_rng = torch.get_rng_state()
        torch.manual_seed(83)
        actual = [sample_author_ddpm(ref, 3) for _ in range(2)]
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        path = sample_author_ddpm(ref, 3, generator=torch.Generator().manual_seed(3), return_trajectory=True)
        end = sample_author_ddpm(ref, 3, generator=torch.Generator().manual_seed(3))
        self.assertEqual(path.shape, (3, len(ref.indexes)+1, 4))
        torch.testing.assert_close(path[:, -1], end, rtol=0, atol=0)
