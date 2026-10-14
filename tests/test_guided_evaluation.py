import json
import unittest

import numpy as np

from guided_evaluation import conditional_diagnostics


class ConditionalDiagnosticsTests(unittest.TestCase):
    def test_known_distances_include_violations_and_event_boundary(self):
        report = conditional_diagnostics([[-2.], [2.]], [[-2.], [-1.]], [1.], -1.)
        expected = {
            'samples': 2, 'benchmark_samples': 2,
            'event_rate': .5, 'benchmark_event_rate': 1.,
            'portfolio_mean': 0., 'benchmark_portfolio_mean': -1.5,
            'portfolio_std': np.sqrt(8), 'benchmark_portfolio_std': np.sqrt(.5),
            'portfolio_wasserstein': 1.5, 'portfolio_ks_statistic': .5,
            'portfolio_mean_bias': 1.5, 'mean_vector_rmse': 1.5,
            'covariance_rmse': 7.5, 'sliced_wasserstein': 1.5,
            'portfolio_quantile_rmse': np.sqrt(np.mean(np.square([.03, .15, 1.5, 2.85, 2.97]))),
        }
        for key, value in expected.items():
            with self.subTest(metric=key):
                self.assertAlmostEqual(report[key], value)
        np.testing.assert_allclose(report['portfolio_quantiles'], [-1.96, -1.8, 0, 1.8, 1.96])
        np.testing.assert_allclose(report['benchmark_portfolio_quantiles'],
                                   [-1.99, -1.95, -1.5, -1.05, -1.01])
        self.assertFalse(report['terminal_filtering'])
        json.dumps(report, allow_nan=False)

    def test_dependence_error_despite_matching_marginals_and_portfolio(self):
        x = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        target = np.array([[-1., -1.], [-1., -1.], [1., 1.], [1., 1.]])
        report = conditional_diagnostics(x, target, [1., 0.], 0., seed=7)
        self.assertEqual(report['portfolio_wasserstein'], 0.)
        self.assertEqual(report['mean_vector_rmse'], 0.)
        self.assertAlmostEqual(report['covariance_rmse'], np.sqrt(8/9))
        self.assertGreater(report['sliced_wasserstein'], .1)
        self.assertEqual(report, conditional_diagnostics(x, target, [1., 0.], 0., seed=7))
        changed = conditional_diagnostics(x, target, [1., 0.], 0., seed=8)
        self.assertNotEqual(report['sliced_wasserstein'], changed['sliced_wasserstein'])
        np.testing.assert_array_equal(x, [[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])

    def test_identical_distributions_unequal_sizes_and_local_randomness(self):
        x = np.array([[-2., 1.], [0., -1.], [2., 0.]])
        before = np.random.get_state()
        report = conditional_diagnostics(x, x.copy(), [1., -1.], 0.)
        after = np.random.get_state()
        for left, right in zip(before, after):
            np.testing.assert_equal(left, right)
        for key in ('portfolio_wasserstein', 'portfolio_ks_statistic', 'portfolio_mean_bias',
                    'portfolio_quantile_rmse', 'mean_vector_rmse', 'covariance_rmse',
                    'sliced_wasserstein'):
            self.assertEqual(report[key], 0., key)
        unequal = conditional_diagnostics(x, x[:2], [1., -1.], 0.)
        self.assertEqual((unequal['samples'], unequal['benchmark_samples']), (3, 2))
        json.dumps(unequal, allow_nan=False)

    def test_invalid_inputs(self):
        x = [[-1.], [1.]]
        for bad in ([], [1., 2.], [[1.]], [[], []], [[np.nan], [0.]], [[np.inf], [0.]]):
            with self.subTest(matrix=bad), self.assertRaises(ValueError):
                conditional_diagnostics(bad, x, [1.], 0.)
            with self.subTest(benchmark=bad), self.assertRaises(ValueError):
                conditional_diagnostics(x, bad, [1.], 0.)
        for weights in ([0.], [1., 2.], [[1.]], [np.nan], [np.inf]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                conditional_diagnostics(x, x, weights, 0.)
        for threshold in (np.nan, np.inf, [0.]):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                conditional_diagnostics(x, x, [1.], threshold)
        for projections in (0, -1, True, 1.5):
            with self.subTest(projections=projections), self.assertRaises(ValueError):
                conditional_diagnostics(x, x, [1.], 0., projections=projections)
