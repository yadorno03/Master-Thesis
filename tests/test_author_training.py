import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from return_preprocessing import NumpyStandardizer, ReturnTransform
import torch
from author_dfm import AuthorDFMConfig
from author_checkpoint import AuthorModel
from author_training import train_author_dfm


@unittest.skipUnless(importlib.util.find_spec('diffusion_factor_model'), 'Install requirements-dfm.txt')
class AuthorTrainingTests(unittest.TestCase):
    def test_actual_upstream_trainer_and_checkpoint(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            config = AuthorDFMConfig(image_size=(2, 2), model_dim=8, timesteps=4,
                                    save_index=1, epochs=1, batch_size=2)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'fit'
                model = train_author_dfm(torch.randn(5, 4), ['a','b','c','d'], path,
                                         config=config, amp=False)
                loaded, info = AuthorModel.load_checkpoint(path / 'dfm.pt')
                self.assertEqual(info['optimizer_updates'], 2)  # upstream drop_last
                self.assertEqual(info['checkpoint_selection_observations'], 0)
                self.assertFalse(json.loads((path/'training.json').read_text())['full_published_defaults'])
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(loaded.state_dict()[key], value)
        finally:
            torch.set_num_threads(previous)

    def test_shared_float64_transform_is_reused_and_checkpointed(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            assets = ['a', 'b', 'c', 'd']
            raw = np.random.default_rng(52).normal(size=(5, 4))
            shared = NumpyStandardizer.fit(raw, assets)
            config = AuthorDFMConfig(image_size=(2, 2), model_dim=8, timesteps=4,
                                     save_index=1, epochs=1, batch_size=2)
            original = shared.mean.copy()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'shared'
                with patch.object(ReturnTransform, 'fit', side_effect=AssertionError('must reuse fit')):
                    model = train_author_dfm(raw, assets, path, config=config, amp=False,
                                             transform=shared.tensor_transform)
                loaded, metadata = AuthorModel.load_checkpoint(path / 'dfm.pt')
                self.assertEqual(metadata['preprocessing_dtype'], 'torch.float64')
                self.assertEqual(next(model.diffusion.parameters()).dtype, torch.float32)
                self.assertIsNot(model.transform, shared.tensor_transform)
                np.testing.assert_array_equal(loaded.transform.mean.numpy(), original)
                np.testing.assert_array_equal(loaded.transform(raw).numpy(), shared.transform(raw))
                np.testing.assert_allclose(loaded.transform.inverse(loaded.transform(raw)).numpy(), raw)
                np.testing.assert_array_equal(shared.mean, original)
                with self.assertRaisesRegex(ValueError, 'Asset order'):
                    train_author_dfm(raw, assets[::-1], Path(directory)/'invalid', config=config,
                                     amp=False, transform=shared.tensor_transform)
                self.assertFalse((Path(directory)/'invalid').exists())
        finally:
            torch.set_num_threads(previous)
