import json
from functools import partial
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from guided_benchmark import benchmark_guidance
from test_conditional_sampling import BatchedGaussianH


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)

    def test_clock_scope_order_streams_counts_and_unfiltered_quality(self):
        h = BatchedGaussianH()
        calls, clock = [], [0.]

        def sampler(method, model, count, **kwargs):
            self.assertEqual(torch.get_num_threads(), 1)
            calls.append((method, count, kwargs))
            clock[0] += 7. if method == 'guided' else 11.
            values = model.w.new_zeros(count, 3)  # All guided draws violate C < 0.
            if method == 'rejection':
                values[:] = -model.w
            return SimpleNamespace(samples=values, diagnostics={'proposals': 3 * count})

        from guided_evaluation import conditional_diagnostics

        def metrics(*args, **kwargs):
            clock[0] += 1000.  # Must not enter either generation timer.
            return conditional_diagnostics(*args, **kwargs)

        threads = torch.get_num_threads()
        with patch('guided_benchmark.generate_conditional_samples', partial(sampler, 'guided')), \
             patch('guided_benchmark.sample_reference_conditional', partial(sampler, 'rejection')), \
             patch('guided_benchmark.perf_counter', side_effect=lambda: clock[0]), \
             patch('guided_benchmark.conditional_diagnostics', side_effect=metrics):
            report, samples = benchmark_guidance(h, num_samples=4, validation_samples=6,
                num_steps=3, repeats=2, batch_size=2, max_proposals=60, setup_seconds=23.)
        self.assertEqual(torch.get_num_threads(), threads)
        self.assertEqual([c[:2] for c in calls], [
            ('guided', 4), ('rejection', 4), ('rejection', 6),
            ('guided', 4), ('rejection', 4), ('rejection', 4), ('guided', 4)])
        self.assertEqual(len({c[2]['generator'].initial_seed() for c in calls}), 7)
        for _, _, kwargs in calls:
            self.assertEqual([kwargs[k] for k in ('num_steps', 'batch_size', 'max_proposals')], [3, 2, 60])
        self.assertEqual([r['seconds'] for r in report['rows']], [7., 11., 11., 7.])
        self.assertEqual([r['event_rate'] for r in report['rows']], [0., 1., 1., 0.])
        self.assertTrue(all(r['samples'] == 4 and not r['terminal_filtering'] for r in report['rows']))
        self.assertTrue(all(r['generation_diagnostics']['proposals'] == 12 for r in report['rows']))
        self.assertEqual(samples['validation_model'].shape, (6, 3))
        self.assertEqual(report['setup_seconds'], 23.)
        json.dumps(report, allow_nan=False)

    def test_real_sampling_unknown_setup_and_failure_restore_threads(self):
        h = BatchedGaussianH()
        threads = torch.get_num_threads()
        report, samples = benchmark_guidance(h, num_samples=4, validation_samples=4,
            repeats=1, num_steps=3, batch_size=8)
        self.assertIsNone(report['setup_seconds'])
        self.assertEqual(samples['guided_0_model'].shape, (4, 3))
        _, repeated = benchmark_guidance(h, num_samples=4, validation_samples=4,
            repeats=1, num_steps=3, batch_size=8)
        for key in samples:
            self.assertTrue((samples[key] == repeated[key]).all())
        with patch('guided_benchmark.sample_reference_conditional', side_effect=RuntimeError('budget')):
            with self.assertRaisesRegex(RuntimeError, 'budget'):
                benchmark_guidance(h, num_samples=4, validation_samples=4, num_steps=3)
        self.assertEqual(torch.get_num_threads(), threads)
        for kwargs in ({'setup_seconds': -1}, {'setup_seconds': float('nan')},
                       {'repeats': 0}, {'num_samples': 1}, {'max_proposals': 1}):
            with self.assertRaises(ValueError):
                benchmark_guidance(h, **kwargs)
