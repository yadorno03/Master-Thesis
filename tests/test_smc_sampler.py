"""Sequential Monte Carlo correction of the guided sampler."""
import unittest

import numpy as np
import torch
from scipy.stats import ks_2samp

import test_h_network
from guided_sampling import sample_guided_smc, systematic_resample
from sampling import generate_samples
from synthetic_benchmark.conditional import H_VARIANTS, SMC_VARIANTS, VARIANTS, sample_h_guided, sample_h_smc
from guided_sampling import DFMHGuidance
from synthetic_benchmark.reporting import MODELS, SMC_ROWS, readable_smc, smc_table


class SMCSamplerTests(unittest.TestCase):
    def setUp(self):
        self.h = test_h_network.HNetworkTests().fixture(torch.float64)

    def test_rows_follow_the_rejection_law_whatever_h_is(self):
        h = self.h
        w = h.w.numpy()
        unguided = generate_samples(h.reference, 400_000, num_steps=8,
                                    generator=torch.Generator().manual_seed(3)).numpy() @ w
        event = unguided <= h.C
        q, rejection = event.mean(), unguided[event]
        diagnostics = {}
        rows = sample_h_smc(h, n_samples=60_000, steps=8, seed=11, batch_size=20_000, diagnostics=diagnostics)
        self.assertEqual(rows.shape, (60_000, 4))
        portfolio = rows @ w
        self.assertTrue((portfolio <= h.C + 1e-12).all())
        # The guided rows alone are off; the SMC rows are not.
        guided = sample_h_guided(h, DFMHGuidance(h), n_samples=60_000, batch_size=20_000, steps=8, seed=7,
                                 final_step='exact') @ w
        self.assertGreater(ks_2samp(guided, rejection).statistic, .1)
        self.assertLess(ks_2samp(portfolio, rejection).statistic, .02)
        # Rows that share a path are not independent: about four standard errors of the SMC mean.
        self.assertAlmostEqual(portfolio.mean(), rejection.mean(), delta=.02)
        self.assertLess(abs(diagnostics['event_probability'] - q), .03 * q)
        self.assertEqual(diagnostics['resampling_count'], len(diagnostics['resampling_steps']) + 1)
        self.assertTrue(all(0 <= step < 7 for step in diagnostics['resampling_steps']))
        for name in ('lowest_effective_sample_fraction', 'final_effective_sample_fraction',
                     'distinct_final_paths_fraction', 'distinct_starts_fraction'):
            self.assertTrue(0 < diagnostics[name] <= 1, name)
        self.assertLessEqual(diagnostics['lowest_effective_sample_fraction'],
                             diagnostics['final_effective_sample_fraction'])
        self.assertEqual((diagnostics['seed'], diagnostics['final_step'], diagnostics['particles']),
                         (11, 'exact', 60_000))

    def test_threshold_controls_the_resampling_times_and_seeds_reproduce(self):
        h = self.h
        never, _ = sample_guided_smc(h, 500, 6, generator=torch.Generator().manual_seed(2), ess_threshold=0.)
        _, diagnostics = sample_guided_smc(h, 500, 6, generator=torch.Generator().manual_seed(2), ess_threshold=0.)
        self.assertEqual((diagnostics['resampling_steps'], diagnostics['resampling_count']), ([], 1))
        again, _ = sample_guided_smc(h, 500, 6, generator=torch.Generator().manual_seed(2), ess_threshold=0.)
        torch.testing.assert_close(never, again, rtol=0, atol=0)
        _, always = sample_guided_smc(h, 500, 6, generator=torch.Generator().manual_seed(2), ess_threshold=1.)
        # The first step has equal weights; every later one is resampled.
        self.assertEqual(always['resampling_steps'], [1, 2, 3, 4])
        one_step, single = sample_guided_smc(h, 200, 1, generator=torch.Generator().manual_seed(2))
        self.assertEqual((one_step.shape, single['resampling_steps']), ((200, 4), []))
        self.assertTrue(bool((one_step @ h.w <= h.C + 1e-12).all()))
        for kwargs in (dict(ess_threshold=1.5), dict(ess_threshold=True), dict(ess_threshold=-.1)):
            with self.assertRaisesRegex(ValueError, 'ess_threshold'):
                sample_guided_smc(h, 8, 4, **kwargs)
        with self.assertRaisesRegex(ValueError, 'num_steps'):
            sample_guided_smc(h, 8, 0)

    def test_systematic_resampling_counts(self):
        generator = torch.Generator().manual_seed(1)
        uniform = systematic_resample(torch.full((7,), 1 / 7, dtype=torch.float64), generator)
        self.assertEqual(uniform.tolist(), list(range(7)))
        weights = torch.tensor([.5, .26, .24, 0.], dtype=torch.float64)
        counts = torch.bincount(systematic_resample(weights.repeat_interleave(1), generator), minlength=4)
        self.assertEqual(int(counts.sum()), 4)
        for count, weight in zip(counts.tolist(), weights.tolist()):
            self.assertIn(count, (int(np.floor(4 * weight)), int(np.floor(4 * weight)) + 1))
        self.assertEqual(int(counts[3]), 0)
        # Unnormalized weights give the same law.
        mean = torch.stack([torch.bincount(systematic_resample(3 * weights, generator), minlength=4)
                            for _ in range(4_000)]).double().mean(0)
        torch.testing.assert_close(mean, 4 * weights, atol=.05, rtol=0)
        for invalid in (torch.tensor([]), torch.tensor([[.5, .5]]), torch.tensor([.5, -.5]),
                        torch.tensor([.5, float('nan')])):
            with self.assertRaisesRegex(ValueError, 'weights'):
                systematic_resample(invalid)

    def test_source_names_and_table(self):
        # The sampled models are the guided sampler of each H model and the SMC sampler of Refined CDG-ML.
        sources = {H_VARIANTS[key][3] for key in VARIANTS} | {f'{key}_smc' for key in SMC_VARIANTS}
        self.assertEqual(sources, set(MODELS))
        _, diagnostics = sample_guided_smc(self.h, 300, 6, generator=torch.Generator().manual_seed(2))
        table = smc_table({'refined_smc': diagnostics}, event_frequency=.07)
        self.assertEqual((list(table.index), list(table.columns)),
                         (list(SMC_ROWS), ['DFM rejection', 'Refined CDG-ML + SMC']))
        self.assertEqual(table.loc['DFM event frequency', 'DFM rejection'], .07)
        note, readable = readable_smc(table)
        self.assertIn('Sequential Monte Carlo', note)
        self.assertEqual(readable.loc['Resampling times', 'DFM rejection'], '—')
        self.assertEqual(readable.loc['Resampling times', 'Refined CDG-ML + SMC'],
                         str(diagnostics['resampling_count']))


if __name__ == '__main__':
    unittest.main()
