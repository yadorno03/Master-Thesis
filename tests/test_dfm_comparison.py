"""Factor variance ratios, exact quantiles, and the figure and tables comparing the two DFMs."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import multivariate_normal, norm  # noqa: E402

from synthetic_benchmark.reporting import (  # noqa: E402
    DFM_LABELS, DISTANCE_LABELS, EXACT_LABEL, QUANTILE_LEVELS, compare, dfm_tables, exact_portfolio_quantiles,
    exact_projection_quantiles, factor_variance_ratios, factor_variance_summary, largest_factor_direction,
    plot_dfm_distributions,
)
from synthetic_conditional_truth import FactorPopulation  # noqa: E402


class FactorVarianceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        B = np.linalg.qr(rng.normal(size=(40, 4)))[0]
        # The last factor (variance 1 < 5 x mean residual variance) cannot be told apart from the noise.
        self.pop = FactorPopulation(B, rng.uniform(.5, 1.5, 40), np.ones(1), np.zeros((1, 4)),
                                    np.diag([60., 20., 8., 1.])[None], dict(population_seed=0))
        self.x = self.pop.sample(4000, rng)

    def test_identifiable_factors_and_exact_rescaling(self):
        ratios, residual = factor_variance_ratios(self.x, self.pop)
        self.assertEqual(len(ratios), 3)
        np.testing.assert_allclose(ratios, 1., atol=.1)
        self.assertAlmostEqual(residual, 1., delta=.05)
        # Halving the returns along the leading factor quarters its ratio and changes nothing else.
        leading = self.pop.B[:, 0]
        shrunk, same_residual = factor_variance_ratios(self.x - .5 * np.outer(self.x @ leading, leading), self.pop)
        np.testing.assert_allclose(shrunk[0], ratios[0] / 4)
        np.testing.assert_allclose(shrunk[1:], ratios[1:])
        self.assertAlmostEqual(same_residual, residual)
        # Inflating only the residual noise moves only the residual ratio.
        outside = np.eye(40) - self.pop.B @ self.pop.B.T
        noisier, larger = factor_variance_ratios(self.x + self.x @ outside, self.pop)
        np.testing.assert_allclose(noisier, ratios)
        self.assertAlmostEqual(larger, 4 * residual)

    def test_summary_and_single_component_guard(self):
        summary = factor_variance_summary(self.x, self.pop)
        ratios, residual = factor_variance_ratios(self.x, self.pop)
        self.assertEqual(summary, dict(identifiable_factors=3, leading_factor_variance_ratio=ratios[0],
                                       factor_variance_ratio_min=ratios.min(),
                                       factor_variance_ratio_max=ratios.max(), residual_variance_ratio=residual))
        mixture = FactorPopulation(self.pop.B, self.pop.D, np.array([.5, .5]), np.zeros((2, 4)),
                                   np.repeat(self.pop.S, 2, axis=0))
        with self.assertRaisesRegex(ValueError, 'single-component'):
            factor_variance_ratios(self.x, mixture)

    def test_exact_quantiles_along_the_largest_factor_and_the_portfolio(self):
        direction = largest_factor_direction(self.pop)
        self.assertAlmostEqual(np.linalg.norm(direction), 1.)
        self.assertGreaterEqual(direction @ self.pop.w, 0.)
        self.assertAlmostEqual(abs(direction @ self.pop.B[:, 0]), 1.)  # S is diagonal, largest variance first.
        mean, covariance = self.pop.moments()
        levels = np.array([.01, .3, .5, .9, .99])
        sd = np.sqrt(direction @ covariance @ direction)
        np.testing.assert_allclose(exact_projection_quantiles(self.pop, direction, levels),
                                   direction @ mean + sd * norm.ppf(levels), atol=1e-3 * sd)
        # Given the event, P(a'R <= y, w'R <= c) / P(w'R <= c) is a bivariate normal ratio.
        c = self.pop.threshold(.05)
        joint = multivariate_normal(
            [direction @ mean, self.pop.w @ mean],
            [[sd ** 2, direction @ covariance @ self.pop.w],
             [direction @ covariance @ self.pop.w, self.pop.w @ covariance @ self.pop.w]])
        conditional = exact_projection_quantiles(self.pop, direction, levels, c)
        np.testing.assert_allclose([joint.cdf([y, c]) / .05 for y in conditional], levels, atol=2e-3)
        np.testing.assert_allclose(self.pop.cdf(exact_portfolio_quantiles(self.pop, levels, c), c), levels,
                                   atol=1e-9)
        np.testing.assert_allclose(exact_portfolio_quantiles(self.pop, levels),
                                   [self.pop.threshold(q) for q in levels])

    def test_figure_draws_both_dfms_against_the_exact_law(self):
        rng = np.random.default_rng(5)
        generated = {'mlp': .7 * self.pop.sample(500, rng), 'gaussian_mlp': self.pop.sample(500, rng)}
        direction = largest_factor_direction(self.pop)
        with tempfile.TemporaryDirectory() as tmp, patch.object(plt, 'show'):
            path = plot_dfm_distributions(generated, self.pop, Path(tmp) / 'dfm_distributions.png')
            self.assertTrue(Path(path).is_file())
            figure = plt.gcf()
            try:
                portfolio, factor = figure.axes
                for axis, project, exact in (
                        (portfolio, self.pop.w, exact_portfolio_quantiles(self.pop, QUANTILE_LEVELS)),
                        (factor, direction, exact_projection_quantiles(self.pop, direction, QUANTILE_LEVELS))):
                    diagonal, *lines = axis.lines
                    self.assertEqual(diagonal.get_label(), EXACT_LABEL)
                    self.assertEqual([line.get_label() for line in lines],
                                     [DFM_LABELS['mlp'], DFM_LABELS['gaussian_mlp']])
                    for line, network in zip(lines, ('mlp', 'gaussian_mlp')):
                        np.testing.assert_allclose(line.get_xdata(), exact)
                        np.testing.assert_allclose(line.get_ydata(),
                                                   np.quantile(generated[network] @ project, QUANTILE_LEVELS))
                    self.assertEqual(len(axis.patches), 0)
                labels = [text.get_text() for text in figure.legends[0].get_texts()]
                self.assertEqual(labels, [EXACT_LABEL, DFM_LABELS['mlp'], DFM_LABELS['gaussian_mlp']])
            finally:
                plt.close(figure)

    def test_readable_tables_match_the_saved_metrics(self):
        rng = np.random.default_rng(6)
        truth = self.pop.sample(400, rng)
        generated = {'mlp': .7 * self.pop.sample(400, rng), 'gaussian_mlp': self.pop.sample(400, rng)}
        values, distances = dfm_tables(generated, truth, self.pop, noise_seed=8, replicates=20)
        self.assertEqual(list(values.columns), ['Exact', DFM_LABELS['mlp'], DFM_LABELS['gaussian_mlp']])
        self.assertEqual(values.loc['5% quantile of the portfolio return', 'Exact'],
                         f'{100 * self.pop.threshold(.05):.2f}%')
        ratios = factor_variance_ratios(generated['mlp'], self.pop)[0]
        self.assertEqual(values.loc['Factor variances relative to the exact ones (min–max)', DFM_LABELS['mlp']],
                         f'{ratios.min():.2f}–{ratios.max():.2f}')
        self.assertEqual(list(distances.columns)[2:], ['Noise level (95%, n=400)',
                                                       'DFM (Gaussian + MLP) within noise?'])
        # Same projections and definitions as metrics.csv, which compare() fills.
        for network in generated:
            saved = compare(generated[network], truth, self.pop)
            for key, metric in (('portfolio_ks', 'portfolio_population_ks'), ('sliced_w1', 'sliced_w1'),
                                ('covariance_error', 'covariance_error')):
                self.assertEqual(distances.loc[DISTANCE_LABELS[key], DFM_LABELS[network]], f'{saved[metric]:.3f}')
        self.assertEqual(distances.loc[DISTANCE_LABELS['covariance_error'], 'DFM (Gaussian + MLP) within noise?'],
                         'yes' if float(distances.iloc[2, 1]) <= float(distances.iloc[2, 2]) else 'no')
        with self.assertRaisesRegex(ValueError, 'size of the ground-truth sample'):
            dfm_tables({'mlp': generated['mlp'][:10], 'gaussian_mlp': generated['gaussian_mlp']}, truth, self.pop)


if __name__ == '__main__':
    unittest.main()
