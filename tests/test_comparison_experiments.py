import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import norm

from comparison_experiments import (gaussian_tail, gaussian_tail_draws, paper_population,
    read_french_zip, load_market_returns, filtered_guidance)


class ComparisonTests(unittest.TestCase):
    def test_gaussian_tail_matches_independent_rejection(self):
        mean, cov = paper_population(4, 2)
        w = np.array([.4, -.1, .5, .2])
        c = float(w @ mean + np.sqrt(w @ cov @ w) * norm.ppf(.2))
        cm, cc = gaussian_tail(mean, cov, w, c)
        # Independently generated unconditional draws and direct rejection.
        x = np.random.default_rng(191).multivariate_normal(mean, cov, 500000)
        rejected = x[x @ w <= c]
        exact = gaussian_tail_draws(mean, cov, w, c, 100000, 192)
        for samples in (rejected, exact):
            self.assertTrue(np.all(samples @ w <= c + 1e-14))
            np.testing.assert_allclose(samples.mean(0), cm, atol=.004, rtol=0)
            np.testing.assert_allclose(np.cov(samples, rowvar=False), cc, atol=.002, rtol=.02)
        # Gaussian lower-half analytic special case, and rare-tail stability.
        m, s = gaussian_tail(np.zeros(2), np.eye(2), np.array([1., 0.]), 0.)
        np.testing.assert_allclose(m, [-np.sqrt(2/np.pi), 0])
        np.testing.assert_allclose(s, np.diag([1-2/np.pi, 1]))
        m, s = gaussian_tail(np.zeros(2), np.eye(2), np.array([1., 0.]), -10.)
        self.assertTrue(np.isfinite(m).all() and np.linalg.eigvalsh(s).min() > 0)

    def test_french_parser_table_units_missing_and_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'returns.zip'
            lines = ['Average Value Weighted Returns -- Daily', ',A,B']
            import pandas as pd
            for date in pd.date_range('2019-11-01', '2020-03-01', freq='B'):
                value = '-99.99' if str(date.date()) == '2019-11-04' else '1.5'
                lines.append(f'{date:%Y%m%d},{value},-2.0')
            lines.extend(['', 'Average Equal Weighted Returns -- Daily', ',A,B', '20191101,99,99'])
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('data.csv', '\n'.join(lines))
            frame = read_french_zip(path)
            self.assertEqual(frame.iloc[0, 0], .015)
            self.assertTrue(np.isnan(frame.loc['2019-11-04', 'A']))
            train, test, metadata = load_market_returns(path)
            self.assertEqual(metadata['dropped_rows'], 1)
            self.assertLess(train.index.max(), test.index.min())
            self.assertTrue((test.B == -.02).all())
            self.assertEqual(len(metadata['sha256']), 64)

    def test_filter_counts_overshoot_and_has_global_initial_budget(self):
        h = SimpleNamespace(w=torch.tensor([1.]), C=0.)
        observed_budgets = []
        def starts(h, n, **kwargs):
            observed_budgets.append(kwargs['max_proposals'])
            return torch.zeros(n, 1), {'proposals': 2*n}
        def sample(h, n, **kwargs):
            return SimpleNamespace(samples=torch.tensor([[-1.], [1.], [-2.], [1.]]))
        with patch('comparison_experiments.sample_conditional_initial', side_effect=starts), \
             patch('comparison_experiments.generate_conditional_samples', side_effect=sample):
            values, info = filtered_guidance(h, 3, steps=2, batch_size=4,
                                            max_proposals=20, generator=None)
            self.assertEqual(values.shape, (3, 1))
            self.assertEqual(observed_budgets, [20, 12])
            self.assertEqual(info['terminal_proposals'], 8)
            self.assertEqual(info['terminal_discarded'], 5)
            self.assertEqual(info['initial_proposals'], 16)
            self.assertEqual(info['initial_discarded'], 8)
            with self.assertRaisesRegex(RuntimeError, 'initial-proposal budget'):
                filtered_guidance(h, 3, steps=2, batch_size=4, max_proposals=10, generator=None)

    def test_filter_budget_does_not_return_incomplete_batch(self):
        h = SimpleNamespace(w=torch.tensor([1.]), C=0.)
        with patch('comparison_experiments.sample_conditional_initial',
                   return_value=(torch.zeros(4,1), {'proposals': 4})), \
             patch('comparison_experiments.generate_conditional_samples',
                   return_value=SimpleNamespace(samples=torch.ones(4,1))):
            with self.assertRaisesRegex(RuntimeError, 'terminal proposals'):
                filtered_guidance(h, 3, steps=2, batch_size=4, max_proposals=4, generator=None)


if __name__ == '__main__':
    unittest.main()
