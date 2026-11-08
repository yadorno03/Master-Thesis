"""Factor variance ratios and the figure comparing the two DFMs."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from synthetic_benchmark.reporting import (  # noqa: E402
    DFM_LABELS, factor_variance_ratios, factor_variance_summary, plot_dfm_comparison,
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

    def test_figure_draws_both_dfms_and_the_noise_reference(self):
        rng = np.random.default_rng(5)
        generated = {'mlp': .7 * self.pop.sample(500, rng), 'gaussian_mlp': self.pop.sample(500, rng)}
        truth = self.pop.sample(500, rng)
        with tempfile.TemporaryDirectory() as tmp, patch.object(plt, 'show'):
            path = plot_dfm_comparison(generated, truth, self.pop, Path(tmp) / 'dfm_comparison.png')
            self.assertTrue(Path(path).is_file())
            figure = plt.gcf()
            try:
                variance, qq, tail = figure.axes
                expected = [factor_variance_ratios(values, self.pop)[0]
                            for values in (generated['mlp'], generated['gaussian_mlp'], truth)]
                np.testing.assert_allclose([bar.get_height() for bar in variance.patches], np.concatenate(expected))
                frequencies = [np.mean(values @ self.pop.w <= self.pop.threshold(q)) / q
                               for values in (generated['mlp'], generated['gaussian_mlp'], truth)
                               for q in (.1, .05, .01)]
                np.testing.assert_allclose([bar.get_height() for bar in tail.patches], frequencies)
                self.assertEqual([line.get_label() for line in qq.lines[:2]], [DFM_LABELS['mlp'],
                                                                                DFM_LABELS['gaussian_mlp']])
                labels = [text.get_text() for text in figure.legends[0].get_texts()]
                self.assertEqual(labels, ['Target', DFM_LABELS['mlp'], DFM_LABELS['gaussian_mlp'],
                                          'Independent truth draw (noise)'])
            finally:
                plt.close(figure)


if __name__ == '__main__':
    unittest.main()
