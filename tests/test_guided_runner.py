import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.run_guided_dfm import ROOT, parser, run, sha256
from h_model import HNetwork
import test_h_network
import test_reference_checkpoint


class GuidedRunnerTests(unittest.TestCase):
    def test_reference_training_cli_and_saved_bundle_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, weights = root / 'reference.pt', root / 'weights.json'
            _, saved = test_reference_checkpoint.ReferenceCheckpointTests().fixture()
            torch.save(saved, source)
            weights.write_text(json.dumps({'B': .8, 'C': .3, 'A': -.1}))
            digest = sha256(source)
            output = root / 'trained'
            options = ['--samples', '8', '--steps', '3', '--timing-repeats', '2',
                       '--timing-reference-samples', '12', '--batch-size', '8',
                       '--max-proposals', '1000', '--seed', '20261006']
            command = [sys.executable, str(ROOT / 'scripts/run_guided_dfm.py'),
                       '--reference-checkpoint', str(source), '--weights', str(weights),
                       '--threshold', '.029', '--output', str(output),
                       '--train-trajectories', '32', '--held-out-trajectories', '16',
                       '--epochs', '2', '--hidden-dims', '4', '--rollout-batch-size', '8',
                       *options]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = json.loads((output / 'run_summary.json').read_text())
            h, metadata = HNetwork.load_checkpoint(output / 'h_checkpoint.pt')
            self.assertEqual(summary['status'], 'complete')
            self.assertEqual(summary['original_metadata'], metadata)
            self.assertEqual(summary['source_checkpoint_sha256'], digest)
            self.assertEqual(metadata['source_checkpoint_sha256'], digest)
            self.assertEqual(sha256(source), digest)
            self.assertEqual(metadata['assets'], ['C', 'A', 'B'])
            self.assertEqual(metadata['preprocessing'], saved['preprocessing'])
            np.testing.assert_allclose(h.config.w, [.06, -.03, .32])
            self.assertAlmostEqual(h.C, 0.)
            self.assertEqual((h.reference.T, h.reference.t0), (1.3, .2))
            self.assertEqual(metadata['training_config']['num_steps'], 3)
            self.assertEqual(len(metadata['training_loss']), 2)
            self.assertIn('constant_loss', metadata['h_validation'])
            self.assertGreater(summary['h_setup_cost']['total_seconds'], 0.)
            timing = json.loads((output / 'timing_benchmark.json').read_text())
            self.assertEqual(timing['setup_seconds'], summary['h_setup_cost']['total_seconds'])
            benchmark_seeds = {20261006 + offset for offset in (0, 1, 2, 3)}
            benchmark_seeds.update(row['seed'] for row in timing['rows'])
            self.assertTrue(benchmark_seeds.isdisjoint(metadata['seeds'].values()))
            for name, expected in summary['artifacts_sha256'].items():
                self.assertEqual(sha256(output / name), expected)
            self.assertIn('h_checkpoint.pt', summary['artifacts_sha256'])
            reuse = parser().parse_args(['--h-checkpoint', str(output / 'h_checkpoint.pt'),
                                        '--output', str(root / 'reused'), *options])
            with patch('scripts.run_guided_dfm.train_h_from_checkpoint') as train:
                reused = run(reuse)
                train.assert_not_called()
            self.assertEqual(reused['event_model'], summary['event_model'])
            self.assertEqual(reused['h_setup_cost'], summary['h_setup_cost'])
            with np.load(output / 'timing_samples.npz') as first, np.load(
                    reuse.output / 'timing_samples.npz') as second:
                self.assertEqual(set(first.files), set(second.files))
                for name in first.files:
                    np.testing.assert_array_equal(first[name], second[name])

    def test_event_arguments_fail_before_creating_output(self):
        cases = [(['--h-checkpoint', 'missing.pt', '--weights', 'weights.json'], 'fixes the event'),
                 (['--h-checkpoint', 'missing.pt', '--threshold', '0'], 'fixes the event'),
                 (['--reference-checkpoint', 'missing.pt'], 'requires'),
                 (['--reference-checkpoint', 'missing.pt', '--threshold', '0'], 'requires'),
                 (['--reference-checkpoint', 'missing.pt', '--weights', 'weights.json'], 'requires')]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            for options, message in cases:
                with self.subTest(options=options):
                    args = parser().parse_args([*options, '--output', str(output)])
                    with self.assertRaisesRegex(ValueError, message):
                        run(args)
                    self.assertFalse(output.exists())

    def test_cli_persists_all_draws_event_and_setup_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h = test_h_network.HNetworkTests().fixture()
            checkpoint = root / 'h.pt'
            previous = None
            for index, cost in enumerate((None, {'total_seconds': 12., 'scope': 'labels + training'})):
                metadata = {} if cost is None else {'h_setup_cost': cost}
                h.save_checkpoint(checkpoint, metadata=metadata)
                digest = sha256(checkpoint)
                output = root / f'run{index}'
                command = [sys.executable, str(ROOT / 'scripts/run_guided_dfm.py'),
                           '--h-checkpoint', str(checkpoint), '--output', str(output),
                           '--samples', '8', '--steps', '3', '--timing-repeats', '2',
                           '--timing-reference-samples', '12', '--batch-size', '8']
                result = subprocess.run(command, cwd=root, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                summary = json.loads((output / 'run_summary.json').read_text())
                timing = json.loads((output / 'timing_benchmark.json').read_text())
                self.assertEqual(summary['status'], 'complete')
                self.assertEqual(summary['source_checkpoint_sha256'], digest)
                self.assertEqual(sha256(checkpoint), digest)
                self.assertEqual(summary['original_metadata'], metadata)
                self.assertEqual(summary['h_setup_cost'], cost)
                self.assertEqual(timing['setup_seconds'], None if cost is None else 12.)
                self.assertEqual(timing['config']['initialization'], 'h_rejection')
                self.assertEqual(summary['event_model'], {
                    'weights': list(h.config.w), 'threshold': h.C,
                    'T': h.reference.T, 't0': h.reference.t0})
                for name, expected in summary['artifacts_sha256'].items():
                    self.assertEqual(sha256(output / name), expected)
                with np.load(output / 'timing_samples.npz', allow_pickle=False) as archive:
                    samples = dict(archive)
                self.assertEqual(len(samples), 5)
                self.assertEqual(samples['validation_model'].shape, (12, h.reference.d))
                self.assertTrue(any(row['event_rate'] < 1. for row in timing['rows']
                                    if row['method'] == 'guided'))
                for row in timing['rows']:
                    values = samples[f"{row['method']}_{row['repeat']}_model"]
                    self.assertEqual(values.shape, (8, h.reference.d))
                    self.assertFalse(row['terminal_filtering'])
                    self.assertEqual(row['event_rate'], np.mean(values @ h.w.numpy() <= h.C))
                if previous is not None:
                    for name in samples:
                        np.testing.assert_array_equal(samples[name], previous[name])
                previous = samples

    def test_failed_run_is_recorded_and_existing_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'h.pt'
            test_h_network.HNetworkTests().fixture().save_checkpoint(checkpoint)
            args = parser().parse_args(['--h-checkpoint', str(checkpoint),
                                       '--output', str(root / 'run')])
            with patch('scripts.run_guided_dfm.benchmark_guidance', side_effect=RuntimeError('budget')):
                with self.assertRaisesRegex(RuntimeError, 'budget'):
                    run(args)
            saved = (args.output / 'run_summary.json').read_bytes()
            self.assertEqual(json.loads(saved)['status'], 'failed')
            self.assertEqual(json.loads(saved)['error'], 'budget')
            with patch('scripts.run_guided_dfm.HNetwork.load_checkpoint') as load:
                with self.assertRaises(FileExistsError):
                    run(args)
                load.assert_not_called()
            self.assertEqual((args.output / 'run_summary.json').read_bytes(), saved)
