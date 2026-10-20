import importlib.util
from pathlib import Path
import tempfile
import unittest
import torch
from author_checkpoint import AuthorModel
from author_dfm import AuthorDFMConfig, build_author_dfm
from author_data import ReturnTransform


@unittest.skipUnless(importlib.util.find_spec('diffusion_factor_model'), 'Install requirements-dfm.txt')
class AuthorCheckpointTests(unittest.TestCase):
    def test_round_trip_and_no_overwrite(self):
        cfg = AuthorDFMConfig(image_size=(2, 2), model_dim=8, timesteps=4, save_index=1)
        prep = ReturnTransform.fit(torch.randn(8, 4), ['a', 'b', 'c', 'd'])
        model = AuthorModel(build_author_dfm(cfg), prep, cfg)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reference.pt'
            model.save_checkpoint(path, metadata={'seed': 4})
            loaded, metadata = AuthorModel.load_checkpoint(path)
            self.assertEqual(metadata, {'seed': 4})
            self.assertEqual(loaded.transform.assets, prep.assets)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
            with self.assertRaises(FileExistsError):
                model.save_checkpoint(path)
        bad = model.checkpoint_payload()
        bad['upstream_commit'] = 'unverified'
        with self.assertRaises(ValueError):
            AuthorModel.from_payload(bad)
