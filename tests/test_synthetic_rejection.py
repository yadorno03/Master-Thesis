"""Benchmark fitting failures and conditional evaluation with tiny frozen samplers."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance

from DFM import UnconditionalDFM
from reference import FrozenDFMReference
from synthetic_conditional_truth import make_population
from synthetic_benchmark.configuration import Config, ConditionalAssessmentConfig, Streams
from synthetic_benchmark.unconditional import fit_dfm
from synthetic_benchmark.conditional import sample_dfm_rejection
from synthetic_benchmark.reporting import (
    conditional_comparisons, conditional_moment_table, conditional_summary, plot_conditional_distributions,
    conditional_tables, exact_portfolio_quantiles, exact_projection_quantiles, format_return,
    largest_factor_direction, matched_orders, noise_levels, portfolio_values, sample_distances,
    DISTANCE_LABELS, EXACT_LABEL, MODELS, QUANTILE_LEVELS, SOURCE_LABELS, SOURCES,
)


class ToyReference(torch.nn.Module):
    d, T, t0 = 3, 1.5, .1

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False)


class DFMFitFailureTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(d=3, k=3, dfm_epochs=1, width=4, batch=4, T=1.5, t0=.1)
        self.train = np.random.default_rng(31).normal(size=(4, 3))
        self.validation = np.random.default_rng(32).normal(size=(3, 3))

    def test_invalid_epochs_fail_before_model_initialization(self):
        for epochs in (0, -1, 1.5, True, False, '1'):
            with self.subTest(epochs=epochs), \
                 patch('synthetic_benchmark.unconditional.UnconditionalDFM') as model:
                with self.assertRaisesRegex(ValueError, 'dfm_epochs must be a positive integer'):
                    fit_dfm(self.train, self.validation, replace(self.config, dfm_epochs=epochs), Streams(41))
                model.assert_not_called()

    def test_nonfinite_training_loss_is_explicit(self):
        train = self.train.copy()
        train[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite DFM training loss at epoch 1'):
            fit_dfm(train, self.validation, self.config, Streams(41))

    def test_nonfinite_validation_loss_is_explicit(self):
        validation = self.validation.copy()
        validation[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite DFM validation loss at epoch 1'):
            fit_dfm(self.train, validation, self.config, Streams(41))

    def test_finite_data_still_returns_selected_frozen_model(self):
        model, history, selected = fit_dfm(self.train, self.validation, self.config, Streams(41))
        self.assertEqual(selected, 1)
        self.assertEqual(len(history), 1)
        self.assertTrue(np.isfinite([history[0]['train'], history[0]['validation']]).all())
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))


class RejectionTests(unittest.TestCase):
    def setUp(self):
        self.reference = ToyReference()
        self.kwargs = dict(steps=7, batch_size=4, seed=73, max_proposals=12)
        self.w = np.array([1., 0., 0.])

    def test_order_counts_excess_and_sampler_contract(self):
        batches = [torch.tensor([[1., 2., 3.], [-1., 4., 5.], [2., 6., 7.], [-2., 8., 9.]]),
                   torch.tensor([[-3., 1., 2.], [-4., 3., 4.], [1., 5., 6.], [-5., 7., 8.]])]
        with patch('synthetic_benchmark.conditional.generate_samples', side_effect=batches) as sampler:
            x, d = sample_dfm_rejection(self.reference, self.w, 0., 3, **self.kwargs)
        np.testing.assert_array_equal(x, np.array([batches[0][1], batches[0][3], batches[1][0]]))
        self.assertTrue(np.all(portfolio_values(x, self.w) <= 0))
        self.assertEqual((d['total_proposals'], d['total_event_hits'], d['retained_count']), (8, 5, 3))
        self.assertEqual(d['observed_acceptance_fraction'], 5/8)
        self.assertNotEqual(d['observed_acceptance_fraction'], len(x)/8)
        self.assertEqual(d['shortfall'], 0)
        self.assertEqual(d['status'], 'complete')
        self.assertTrue(d['complete'])
        self.assertGreaterEqual(d['sampling_seconds'], 0.)
        generators = []
        for call in sampler.call_args_list:
            self.assertIs(call.args[0], self.reference)
            self.assertEqual(call.args[1], 4)
            self.assertEqual(set(call.kwargs), {'num_steps', 'generator'})
            self.assertEqual(call.kwargs['num_steps'], 7)
            generators.append(call.kwargs['generator'])
        self.assertIs(generators[0], generators[1])
        self.assertEqual(generators[0].initial_seed(), 73)
        self.assertEqual((d['horizon'], d['stopping_time']), (1.5, .1))
        self.assertEqual(d['initialization'], 'N(0,I)')
        self.assertEqual(d['dtype'], 'torch.float32')

    def test_cap_partial_empty_and_boundary(self):
        for hit in (False, True):
            def proposals(reference, n, **kwargs):
                # Boundary rows must be accepted exactly; no tolerance on positives.
                return torch.tensor([[0., 10., 20.] if hit else [1e-12, 10., 20.]]).repeat(n, 1)
            with patch('synthetic_benchmark.conditional.generate_samples', side_effect=proposals) as sampler:
                x, d = sample_dfm_rejection(self.reference, self.w, 0., 10,
                                            **dict(self.kwargs, max_proposals=6))
            self.assertEqual([c.args[1] for c in sampler.call_args_list], [4, 2])
            self.assertEqual(d['total_proposals'], 6)
            self.assertEqual(len(x), 6 if hit else 0)
            self.assertEqual(x.shape[1], 3)
            self.assertEqual(d['shortfall'], 4 if hit else 10)
            self.assertEqual(d['status'], 'partial' if hit else 'empty')
            self.assertFalse(d['complete'])

    def test_nonfinite_output_and_integration_failure(self):
        with patch('synthetic_benchmark.conditional.generate_samples', return_value=torch.full((4, 3), float('nan'))):
            x, d = sample_dfm_rejection(self.reference, self.w, 0., 3, **self.kwargs)
        self.assertEqual(d['status'], 'nonfinite_output')
        self.assertEqual(d['total_proposals'], 4)
        self.assertEqual(x.shape, (0, 3))
        with patch('synthetic_benchmark.conditional.generate_samples', side_effect=ValueError('Sampling produced nonfinite values')):
            x, d = sample_dfm_rejection(self.reference, self.w, 0., 3, **self.kwargs)
        self.assertEqual(d['status'], 'nonfinite_sampler')
        self.assertEqual(d['failed_batch_attempted'], 4)
        self.assertEqual(d['total_proposals'], 0)
        self.assertIsNone(d['observed_acceptance_fraction'])

    def test_invalid_arguments(self):
        for name, value in (('steps', 0), ('steps', True), ('batch_size', 0), ('seed', -1),
                            ('seed', 2**63), ('max_proposals', 1.5)):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                sample_dfm_rejection(self.reference, self.w, 0., 3, **dict(self.kwargs, **{name: value}))
        for w, threshold, n in (([0., 0., 0.], 0., 3), ([1., 0.], 0., 3),
                                 ([1., np.nan, 0.], 0., 3), (self.w, np.nan, 3), (self.w, 0., 0),
                                 (self.w, True, 3), (self.w, 'bad', 3), ([1j, 0., 0.], 0., 3)):
            with self.assertRaises(ValueError):
                sample_dfm_rejection(self.reference, w, threshold, n, **self.kwargs)
        self.reference.anchor.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'frozen'):
            sample_dfm_rejection(self.reference, self.w, 0., 3, **self.kwargs)

    def test_real_sampler_reproducibility_frozen_parameters_and_rng_isolation(self):
        with torch.random.fork_rng():
            torch.manual_seed(19)
            reference = FrozenDFMReference(UnconditionalDFM(3, 3, T=1.5, t0=.1, hidden_dims=[4, 4]))
        before = {k: v.clone() for k, v in reference.state_dict().items()}
        rng_before = torch.get_rng_state().clone()
        np_before = np.random.get_state()
        streams_before = Streams(42).seeds
        args = dict(steps=3, batch_size=8, seed=77, max_proposals=32)
        a, da = sample_dfm_rejection(reference, self.w, 0., 5, **args)
        b, db = sample_dfm_rejection(reference, self.w, 0., 5, **args)
        c, dc = sample_dfm_rejection(reference, self.w, 0., 5, **dict(args, seed=78))
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))
        self.assertEqual({k: v for k, v in da.items() if k != 'sampling_seconds'},
                         {k: v for k, v in db.items() if k != 'sampling_seconds'})
        torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
        np.testing.assert_array_equal(np.random.get_state()[1], np_before[1])
        self.assertEqual(np.random.get_state()[2:], np_before[2:])
        self.assertEqual(Streams(42).seeds, streams_before)
        for k, v in reference.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in reference.parameters()))
        self.assertEqual(da['status'], 'complete')
        self.assertEqual(dc['status'], 'complete')


class ConditionalMetricTests(unittest.TestCase):
    def samples(self):
        rng = np.random.default_rng(9)
        return {name: rng.normal(size=(n, 3)) for name, n in
                [('refined_exact', 3), ('cdg_ml_exact', 20), ('unguided_exact', 20), ('rejection', 12),
                 ('truth', 20), ('truth_reference', 20), ('rejection_reference', 9)]}

    def test_value_independent_matching_and_shared_unit_projections(self):
        x = self.samples()
        original = {k: v.copy() for k, v in x.items()}
        kwargs = dict(projection_seed=93, matching_seed=94, projections=5)
        with patch('synthetic_benchmark.reporting.compare', side_effect=AssertionError('unconditional compare')):
            table, directions = conditional_comparisons(x, [1., 0., 0.], **kwargs)
        pd.testing.assert_frame_equal(table, conditional_comparisons(x, [1., 0., 0.], **kwargs)[0])
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.)
        for name in x:
            np.testing.assert_array_equal(x[name], original[name])
        for r in table.itertuples():
            if r.row_type == 'comparison':
                self.assertEqual(r.n_used, min(r.n_left, r.n_right))
        self.assertEqual(set(zip(table.left, table.right)) - {('truth', 'truth_reference'),
                                                              ('rejection', 'rejection_reference')},
                         {(m, r) for m in MODELS for r in ('rejection', 'truth')} | {('rejection', 'truth')})
        # Reconstruct the documented seeded permutation rule to check W1.
        names = (*MODELS, 'rejection', 'truth', 'truth_reference', 'rejection_reference')
        orders = {k: np.random.default_rng(s).permutation(len(x[k]))
                  for k, s in zip(names, np.random.SeedSequence(94).spawn(len(names)))}
        for r in table.itertuples():
            if r.n_used:
                target = wasserstein_distance(x[r.left][orders[r.left][:r.n_used], 0],
                                              x[r.right][orders[r.right][:r.n_used], 0])
                self.assertAlmostEqual(r.portfolio_w1, target)
        # Every comparison size has both reference types, with no silent downsizing.
        for n in (3, 12, 20):
            rows = table[(table.row_type == 'MC reference') & (table.target_n == n)]
            self.assertEqual(len(rows), 2)
        short = table[(table.left == 'rejection') & (table.row_type == 'MC reference') & (table.target_n == 12)].iloc[0]
        self.assertEqual(short.status, 'reference_shortfall')
        self.assertEqual(short.n_used, 0)
        self.assertTrue(np.isnan(short.portfolio_w1))
        shifted = {k: v + np.array([10., 0., 0.]) for k, v in x.items()}
        # No population methods/targets enter these metrics; shifting both sides preserves distances.
        other, other_dirs = conditional_comparisons(shifted, [1., 0., 0.], **kwargs)
        np.testing.assert_allclose(table.portfolio_w1, other.portfolio_w1, equal_nan=True, atol=1e-14)
        np.testing.assert_array_equal(directions, other_dirs)

    def test_empty_singleton_and_input_validation(self):
        x = self.samples()
        x['refined_exact'] = x['refined_exact'][:0]
        x['rejection_reference'] = x['rejection_reference'][:1]
        table, _ = conditional_comparisons(x, [1., 0., 0.], projection_seed=1, matching_seed=2)
        empty = table[table.left == 'refined_exact']
        self.assertTrue((empty.status == 'empty').all())
        self.assertTrue(empty.portfolio_w1.isna().all())
        one = {k: v[:1] for k, v in self.samples().items()}
        table, _ = conditional_comparisons(one, [1., 0., 0.], projection_seed=1, matching_seed=2)
        self.assertTrue((table.status == 'singleton').all())
        self.assertTrue(np.isfinite(table.portfolio_w1).all())
        x['truth'][0, 0] = np.nan
        with self.assertRaises(ValueError):
            conditional_comparisons(x, [1., 0., 0.], projection_seed=1, matching_seed=2)

    def five_sources(self):
        pop = make_population(4, 'gaussian', 3)
        threshold = float(pop.threshold(.2))
        rng = np.random.default_rng(3)
        x = {name: pop.sample(n, rng, c=threshold) for name, n in
             (('truth', 40), ('truth_reference', 40), ('rejection', 30), ('rejection_reference', 30),
              ('refined_exact', 40))}
        x['cdg_ml_exact'] = pop.sample(40, rng)  # Unconditional draws: many event violations.
        # Rows moved onto C along w and stored in float32, as the exact final step leaves a path
        # that is still far above the threshold.
        inside = pop.sample(40, rng, c=threshold)
        x['unguided_exact'] = (inside + np.outer(threshold - inside @ pop.w, pop.w) / (pop.w @ pop.w)
                               ).astype(np.float32)
        return pop, threshold, x

    def test_summary_diagonal_entries_are_monte_carlo_levels(self):
        pop, threshold, x = self.five_sources()
        table, _ = conditional_comparisons(x, pop.w, projection_seed=5, matching_seed=6, projections=4)
        moments = conditional_moment_table(x, pop, threshold)
        summary = conditional_summary(x, table, moments, threshold, pop.w, near_band=.05)
        self.assertEqual(list(summary.columns), [SOURCE_LABELS[s] for s in SOURCES])
        indexed = table.set_index(['left', 'right', 'n_used'])
        truth, rejection = SOURCE_LABELS['truth'], SOURCE_LABELS['rejection']
        self.assertEqual(summary.loc['Sliced W1 vs ground truth', truth],
                         indexed.loc[('truth', 'truth_reference', 40), 'sliced_w1'])
        self.assertEqual(summary.loc['Sliced W1 vs DFM rejection', rejection],
                         indexed.loc[('rejection', 'rejection_reference', 30), 'sliced_w1'])
        self.assertEqual(summary.loc['Sliced W1 vs DFM rejection', truth],
                         indexed.loc[('rejection', 'truth', 30), 'sliced_w1'])
        unguided, cdg_ml = SOURCE_LABELS['unguided_exact'], SOURCE_LABELS['cdg_ml_exact']
        self.assertEqual(summary.loc['Event rate', truth], 1.)
        self.assertLess(summary.loc['Event rate', cdg_ml], 1.)
        # Rows on C count as inside the event despite float32 rounding.
        self.assertGreater(np.max(x['unguided_exact'].astype(np.float64) @ pop.w), threshold)
        self.assertEqual(summary.loc['Event rate', unguided], 1.)
        self.assertEqual(summary.loc['Rows on the threshold', unguided], 1.)
        self.assertEqual(summary.loc['Rows on the threshold', truth], 0.)
        p = x['truth'] @ pop.w
        self.assertEqual(summary.loc['Rows just below the threshold', truth], np.mean(p > threshold - .05))
        self.assertNotIn('Rows just below the threshold',
                         conditional_summary(x, table, moments, threshold, pop.w).index)
        mean, covariance = pop.moments(c=threshold)
        row = moments.set_index('source').loc['truth']
        self.assertAlmostEqual(row.mean_error, np.linalg.norm(x['truth'].mean(0) - mean)
                               / np.sqrt(np.trace(pop.moments()[1])))
        self.assertAlmostEqual(row.covariance_error, np.linalg.norm(np.cov(x['truth'], rowvar=False) - covariance)
                               / np.linalg.norm(covariance))

    def test_distribution_figure_draws_only_shown_sources_against_the_exact_law(self):
        pop, threshold, x = self.five_sources()
        shown = ('rejection', 'refined_exact')
        original_sources = set(x)
        with tempfile.TemporaryDirectory() as tmp, patch.object(plt, 'show'):
            path = plot_conditional_distributions(x, pop, threshold, Path(tmp), sources=('truth', *shown))
            self.assertEqual(path.name, 'conditional_distributions.png')
            self.assertTrue(path.is_file())
            fig = plt.gcf()
            try:
                cdf, portfolio_qq, factor_qq = fig.axes
                exact = cdf.lines[0]
                self.assertEqual(exact.get_label(), EXACT_LABEL)
                np.testing.assert_allclose(exact.get_ydata(), pop.cdf(exact.get_xdata(), c=threshold))
                for line, source in zip(cdf.lines[1:3], shown):
                    self.assertEqual(line.get_label(), SOURCE_LABELS[source])
                    self.assertEqual(line.get_drawstyle(), 'steps-post')
                    np.testing.assert_allclose(line.get_xdata(), np.sort(portfolio_values(x[source], pop.w)))
                direction = largest_factor_direction(pop)
                for axis, project, exact_quantiles in (
                        (portfolio_qq, pop.w, exact_portfolio_quantiles(pop, QUANTILE_LEVELS, c=threshold)),
                        (factor_qq, direction,
                         exact_projection_quantiles(pop, direction, QUANTILE_LEVELS, c=threshold))):
                    diagonal, *lines = axis.lines
                    self.assertEqual(diagonal.get_label(), EXACT_LABEL)
                    self.assertEqual([line.get_label() for line in lines], [SOURCE_LABELS[s] for s in shown])
                    for line, source in zip(lines, shown):
                        np.testing.assert_allclose(line.get_xdata(), exact_quantiles)
                        np.testing.assert_allclose(line.get_ydata(), np.quantile(x[source] @ project, QUANTILE_LEVELS))
                for axis in fig.axes:
                    self.assertEqual(len(axis.patches), 0)  # Distributions only, no bars.
                legend = {text.get_text() for text in fig.legends[0].get_texts()}
                self.assertEqual(legend, {EXACT_LABEL, 'Event threshold C', *(SOURCE_LABELS[s] for s in shown)})
                self.assertEqual(set(x), original_sources)
            finally:
                plt.close(fig)
        for sources in (('truth',), ('rejection', 'rejection'), ('rejection', 'unknown')):
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                plot_conditional_distributions(x, pop, threshold, '.', sources=sources)

    def test_readable_tables_use_matched_rows_and_noise_levels(self):
        pop, threshold, x = self.five_sources()
        table, directions = conditional_comparisons(x, pop.w, projection_seed=5, matching_seed=6, projections=4)
        values, distances = conditional_tables(x, pop, threshold, directions=directions, matching_seed=6,
                                               noise_seed=7, replicates=20)
        self.assertEqual(list(values.columns), ['Exact', 'DFM rejection', 'Refined CDG-ML'])
        mean, covariance = pop.moments(c=threshold)
        self.assertEqual(values.loc['Expected shortfall: mean return given the event', 'Exact'],
                         format_return(pop.w @ mean, True))
        self.assertEqual(values.loc['Volatility given the event', 'Refined CDG-ML'],
                         format_return((x['refined_exact'] @ pop.w).std(ddof=1), True))
        self.assertEqual(values.loc['Rows in the event', 'Refined CDG-ML'], '100.0%')
        self.assertEqual(values.loc['Rows generated', 'DFM rejection'], '30')
        # Every source uses the 30 rows of the smallest sample, so one noise level applies.
        noise_column, verdict_column = 'Noise level (95%, n=30)', 'Refined CDG-ML within noise?'
        self.assertEqual(list(distances.columns), ['DFM rejection', 'Refined CDG-ML', noise_column, verdict_column])
        self.assertEqual(list(distances.index), list(DISTANCE_LABELS.values()))
        # The matched rows are those of conditional_comparisons.
        row = table[(table.left == 'rejection') & (table.right == 'truth')].iloc[0]
        self.assertEqual(row.n_used, 30)
        self.assertEqual(distances.loc[DISTANCE_LABELS['sliced_w1'], 'DFM rejection'], f'{row.sliced_w1:.3f}')
        orders = matched_orders({name: len(x[name]) for name in ('refined_exact', 'truth')}, 6)
        refined = sample_distances(x['refined_exact'][orders['refined_exact'][:30]], x['truth'][orders['truth'][:30]],
                                   pop, threshold, directions, covariance)
        noise = noise_levels(pop, 30, threshold, directions, covariance, replicates=20, seed=7)
        for key, label in DISTANCE_LABELS.items():
            self.assertEqual(distances.loc[label, 'Refined CDG-ML'], f'{refined[key]:.3f}')
            self.assertEqual(distances.loc[label, noise_column], f'{noise[key]:.3f}')
            self.assertEqual(distances.loc[label, verdict_column], 'yes' if refined[key] <= noise[key] else 'no')
        with self.assertRaises(ValueError):
            conditional_tables(x, pop, threshold, directions=directions, matching_seed=6, noise_seed=7,
                               verdict='cdg_ml_exact')
        with self.assertRaises(ValueError):
            noise_levels(pop, 30, threshold, directions, covariance, replicates=0, seed=7)

    def test_settings_validate_independent_seeds(self):
        settings = ConditionalAssessmentConfig()
        for field, value in [('rejection_n', 0), ('rejection_reference_max_proposals', -1),
                             ('projections', True), ('matching_seed', -1),
                             ('truth_reference_seed', settings.rejection_seed),
                             ('noise_seed', settings.matching_seed), ('noise_replicates', 0)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(settings, **{field: value})


if __name__ == '__main__':
    unittest.main()
