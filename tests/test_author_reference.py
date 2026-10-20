import importlib.util
import unittest
import torch
from author_dfm import AuthorDFMConfig, build_author_dfm
from author_data import ReturnTransform
from author_checkpoint import AuthorModel
from author_reference import FrozenAuthorReference


def small_reference(save_index=1):
    cfg = AuthorDFMConfig(image_size=(2, 2), model_dim=8, timesteps=4, save_index=save_index)
    prep = ReturnTransform.fit(torch.randn(8, 4), ['a','b','c','d'])
    return FrozenAuthorReference(AuthorModel(build_author_dfm(cfg), prep, cfg), take_ownership=True)


@unittest.skipUnless(importlib.util.find_spec('diffusion_factor_model'), 'Install requirements-dfm.txt')
class AuthorReferenceTests(unittest.TestCase):
    def test_each_step_matches_upstream_and_stays_frozen(self):
        ref = small_reference(0)
        x = ref.normal(3)
        for index in ref.indexes:
            torch.manual_seed(71)
            expected, _ = ref.model.diffusion.p_sample(x.reshape(3,1,2,2), index)
            torch.manual_seed(71)
            actual = ref.step(x, index)
            torch.testing.assert_close(actual, expected.flatten(1), rtol=0, atol=0)
        ref.train()
        self.assertTrue(all(not m.training for m in ref.modules()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in ref.parameters()))
        with self.assertRaises(ValueError):
            ref.step(x, -1)
