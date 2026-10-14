import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from guided_benchmark import benchmark_guidance
from h_model import HNetwork
from h_setup import train_h_from_checkpoint
import test_reference_checkpoint


class HSetupTests(unittest.TestCase):
    def test_training_bundle_reproducibility_and_benchmark_cost(self):
        options = dict(weights={'B': .8, 'C': .3, 'A': -.1}, threshold=.029,
                       num_steps=3, train_trajectories=32, held_out_trajectories=16,
                       epochs=2, batch_size=8, rollout_batch_size=8, hidden_dims=[4])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reference.pt'
            _, saved = test_reference_checkpoint.ReferenceCheckpointTests().fixture()
            torch.save(saved, path)
            original, rng = path.read_bytes(), torch.get_rng_state().clone()
            h, metadata = train_h_from_checkpoint(path, **options)
            duplicate, repeated = train_h_from_checkpoint(path, **options)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(metadata['source_checkpoint_sha256'], hashlib.sha256(original).hexdigest())
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertFalse(h.training)
            for key, value in h.state_dict().items():
                torch.testing.assert_close(value, duplicate.state_dict()[key], rtol=0, atol=0)
            for key, value in h.reference._score.state_dict().items():
                torch.testing.assert_close(value, saved['state_dict']['factor_score_network.' + key],
                                           rtol=0, atol=0)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in h.reference.parameters()))
            self.assertEqual(metadata['assets'], ['C', 'A', 'B'])
            self.assertEqual(metadata['preprocessing'], saved['preprocessing'])
            x = np.arange(12.).reshape(4, 3)
            raw_margin = (x * [.2, .3, .4] + [.01, -.02, .03]) @ [.3, -.1, .8] - .029
            np.testing.assert_allclose(x @ h.config.w - h.C, raw_margin, atol=1e-15)
            self.assertEqual(metadata['training_loss'], repeated['training_loss'])
            self.assertEqual(metadata['h_validation'], repeated['h_validation'])
            validation = h.sample_trajectories(16, 3, batch_size=8,
                generator=torch.Generator().manual_seed(metadata['seeds']['validation_paths']))
            expected = (validation.labels - metadata['training_event_rate']).square().mean().item()
            self.assertAlmostEqual(metadata['h_validation']['constant_loss'], expected)
            self.assertEqual(len(set(metadata['seeds'].values())), 4)
            cost = metadata['h_setup_cost']
            self.assertGreater(cost['total_seconds'], 0.)
            self.assertEqual(cost['total_seconds'], sum(cost[key] for key in
                ('initialization_seconds', 'label_generation_seconds', 'optimization_seconds')))
            bundle = Path(directory) / 'h.pt'
            h.save_checkpoint(bundle, metadata=metadata)
            restored, restored_metadata = HNetwork.load_checkpoint(bundle)
            self.assertEqual(restored_metadata, metadata)
            self.assertEqual(restored.config, h.config)
            self.assertEqual(restored.horizon, 1.1)
            report, samples = benchmark_guidance(restored, num_samples=4, num_steps=3,
                repeats=1, validation_samples=8, batch_size=8, max_proposals=1000,
                setup_seconds=restored_metadata['h_setup_cost']['total_seconds'])
            self.assertEqual(report['setup_seconds'], cost['total_seconds'])
            self.assertEqual(samples['guided_0_model'].shape, (4, 3))

    def test_invalid_event_and_single_class_labels_fail_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reference.pt'
            _, saved = test_reference_checkpoint.ReferenceCheckpointTests().fixture()
            torch.save(saved, path)
            for weights, threshold in (({'C': 1.}, 0.), ({'C': 0., 'A': 0., 'B': 0.}, 0.),
                                       ({'C': 1., 'A': 0., 'B': 0.}, float('nan')),
                                       ({'C': 1., 'A': 0., 'B': 0.}, 1e10)):
                with self.subTest(weights=weights, threshold=threshold), patch('h_setup.train_h') as train:
                    with self.assertRaises(ValueError):
                        train_h_from_checkpoint(path, weights=weights, threshold=threshold,
                            train_trajectories=8, num_steps=2)
                    train.assert_not_called()
