import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.run_guided_dfm import ROOT, parser, run, sha256
import test_h_network


class GuidedRunnerTests(unittest.TestCase):
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
                self.assertEqual(timing['config']['scale'], 1.)
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
