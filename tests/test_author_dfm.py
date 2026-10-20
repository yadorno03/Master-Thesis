import importlib.util
import unittest
import torch
from author_dfm import AuthorDFMConfig, author_module, build_author_dfm


@unittest.skipUnless(importlib.util.find_spec('diffusion_factor_model'), 'Install requirements-dfm.txt')
class AuthorBackendTests(unittest.TestCase):
    def test_factory_matches_direct_upstream_construction(self):
        config = AuthorDFMConfig(image_size=(4, 4), model_dim=8, timesteps=8, save_index=1)
        upstream = author_module()
        torch.manual_seed(4)
        actual = build_author_dfm(config)
        torch.manual_seed(4)
        expected = upstream.GaussianDiffusion(upstream.Unet(dim=8, channels=1,
            filter_size=7, dim_mults=(1, 2)), image_size=(4, 4), latent_dim=16,
            timesteps=8, objective='pred_noise', beta_schedule='cosine', auto_normalize=False)
        for name, tensor in actual.state_dict().items():
            torch.testing.assert_close(tensor, expected.state_dict()[name], rtol=0, atol=0)
        x, noise, t = torch.randn(2, 1, 4, 4), torch.randn(2, 1, 4, 4), torch.tensor([0, 5])
        torch.testing.assert_close(actual.p_losses(x, t, noise=noise), expected.p_losses(x, t, noise=noise))
        self.assertFalse(config.metadata()['released_config_defaults'])


class AuthorConfigTests(unittest.TestCase):
    def test_defaults_and_invalid_geometry(self):
        c = AuthorDFMConfig()
        self.assertEqual((c.d, c.dim_mults, c.save_index), (2048, (1, 2, 4, 16), 20))
        self.assertTrue(c.metadata()['released_config_defaults'])
        for kwargs in ({'image_size': (33, 64)}, {'model_dim': 3}, {'save_index': 200}):
            with self.assertRaises(ValueError):
                AuthorDFMConfig(**kwargs)
