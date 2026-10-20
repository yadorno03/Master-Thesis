import importlib.util
from pathlib import Path
import tempfile
import unittest
import torch
from author_h import DDPMHNetwork
from author_h_checkpoint import save_ddpm_h, load_ddpm_h
from conditioning import ConditioningConfig
from test_author_reference import small_reference


@unittest.skipUnless(importlib.util.find_spec('diffusion_factor_model'), 'Install requirements-dfm.txt')
class DDPMHCheckpointTests(unittest.TestCase):
    def test_bundle_restores_reference_event_and_probabilities(self):
        h = DDPMHNetwork(small_reference(), ConditioningConfig([.25]*4, .1), (8,))
        x = h.reference.normal(3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'h.pt'
            save_ddpm_h(h, path, metadata={'trajectories': 20})
            loaded, info = load_ddpm_h(path)
            torch.testing.assert_close(loaded(2,x), h(2,x), rtol=0, atol=0)
            torch.testing.assert_close(loaded.gradient(2,x), h.gradient(2,x), rtol=0, atol=0)
            self.assertEqual(loaded.event, h.event)
            self.assertEqual(loaded.reference.indexes, h.reference.indexes)
            self.assertEqual(info['trajectories'], 20)
            self.assertTrue(all(not p.requires_grad for p in loaded.reference.parameters()))
            with self.assertRaises(FileExistsError):
                save_ddpm_h(h, path)
