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
from synthetic_benchmark.conditional import (
    sample_dfm_rejection, assess_conditional_samples, filter_conditional_samples,
    save_conditional_samples,
)
from synthetic_benchmark.reporting import (
    conditional_comparisons, conditional_sample_summary, extend_conditional_summary,
    plot_conditional_ecdf, plot_h_refinement_comparison, portfolio_values,
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
                [('guided', 20), ('filtered', 3), ('rejection', 12), ('truth', 20),
                 ('truth_reference', 20), ('rejection_reference', 9)]}

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
        # Reconstruct the documented seeded permutation rule to check W1.
        names = ('guided', 'filtered', 'rejection', 'truth', 'truth_reference', 'rejection_reference')
        orders = {k: np.random.default_rng(s).permutation(len(x[k]))
                  for k, s in zip(names, np.random.SeedSequence(94).spawn(6))}
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
        x['filtered'] = x['filtered'][:0]
        x['rejection_reference'] = x['rejection_reference'][:1]
        table, _ = conditional_comparisons(x, [1., 0., 0.], projection_seed=1, matching_seed=2)
        empty = table[table.left == 'filtered']
        self.assertTrue((empty.status == 'empty').all())
        self.assertTrue(empty.portfolio_w1.isna().all())
        one = {k: v[:1] for k, v in self.samples().items()}
        table, _ = conditional_comparisons(one, [1., 0., 0.], projection_seed=1, matching_seed=2)
        self.assertTrue((table.status == 'singleton').all())
        self.assertTrue(np.isfinite(table.portfolio_w1).all())
        x['truth'][0, 0] = np.nan
        with self.assertRaises(ValueError):
            conditional_comparisons(x, [1., 0., 0.], projection_seed=1, matching_seed=2)

    def test_filter_uses_identical_float64_event_convention(self):
        pop = make_population(3, 'gaussian', 3)
        guided = np.array([[-1., -1., -1.], [0., 0., 0.], [1e-9, 0., 0.]], dtype=np.float32)
        truth = np.full((3, 3), -1.)
        p, _, mask, filtered, _ = filter_conditional_samples(guided, truth, pop, 0., 3, 3)
        np.testing.assert_array_equal(p, portfolio_values(guided, pop.w))
        np.testing.assert_array_equal(mask, [True, True, False])
        np.testing.assert_array_equal(filtered, guided[:2])

    def test_assessment_independent_truth_caps_and_strict_artifacts(self):
        pop = make_population(3, 'gaussian', 3)
        guided, truth = np.ones((5, 3)), -np.ones((5, 3))
        settings = ConditionalAssessmentConfig(rejection_n=3, rejection_batch_size=4,
                    rejection_max_proposals=8, rejection_reference_max_proposals=4, projections=5)
        reference = ToyReference()
        seeds = []
        def proposals(ref, n, *, num_steps, generator):
            seeds.append(generator.initial_seed())
            # Complete primary; independent reference remains empty.
            return torch.full((n, 3), -1. if generator.initial_seed() == settings.rejection_seed else 1.)
        before = torch.get_rng_state().clone()
        with patch('synthetic_benchmark.conditional.generate_samples', side_effect=proposals), \
             patch.object(pop, 'sample', wraps=pop.sample) as exact:
            result = assess_conditional_samples(reference, pop, 0., guided, truth, guided[:0],
                                                steps=2, settings=settings, existing_seeds=[41001, 41002])
        self.assertEqual(seeds, [41003, 41004])
        self.assertEqual(exact.call_args.kwargs, {'c': 0.})
        np.testing.assert_array_equal(result['samples']['truth_reference'],
                                     pop.sample(5, np.random.default_rng(41005), c=0.))
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        self.assertIs(result['samples']['guided'], guided)
        with self.assertRaisesRegex(ValueError, 'differ'):
            assess_conditional_samples(reference, pop, 0., guided, truth, guided[:0],
                                       steps=2, settings=settings, existing_seeds=[41003])
        summary = conditional_sample_summary(guided @ pop.w, np.array([]), truth @ pop.w, 0., 5)
        summary = extend_conditional_summary(summary, result['samples'], pop.w, 0., result['rejection_diagnostics'])
        with tempfile.TemporaryDirectory() as tmp:
            out = save_conditional_samples(Path(tmp), {'case': 'gaussian'}, pop, 0., guided, truth,
                guided[:0], np.zeros(5, dtype=bool), summary, n_samples=5, steps=2, batch_size=4,
                guided_seed=41001, truth_seed=41002, assessment=result)
            self.assertTrue((out / 'conditional_metrics.csv').exists())
            strict = lambda p: json.loads(p.read_text(), parse_constant=lambda s: self.fail('Nonstandard JSON ' + s))
            saved = strict(out / 'rejection_diagnostics.json')
            self.assertEqual(saved['rejection_reference']['status'], 'empty')
            self.assertEqual(saved['rejection_reference']['shortfall'], 3)
            rows = strict(out / 'conditional_metrics.json')
            self.assertTrue(any(r['portfolio_w1'] is None for r in rows))
            self.assertEqual(strict(out / 'sampling.json')['assessment_config'], result['assessment_config'])
            with np.load(out / 'conditional_samples.npz') as z:
                for key in ('guided_samples', 'true_conditional_samples', 'filtered_guided_samples',
                            'guided_event_mask', 'rejection_samples', 'independent_rejection_samples',
                            'independent_true_conditional_samples', 'projection_directions'):
                    self.assertIn(key, z.files)
            with patch.object(plt, 'show'):
                fig = plot_conditional_ecdf(result['samples'], pop, 0., 'gaussian', result['comparison_metrics'],
                                            out, result['rejection_diagnostics'])
            ax = fig.axes[0]
            self.assertLess(ax.get_xlim()[0], -1.)
            self.assertGreater(ax.get_xlim()[1], 1.)  # Raw guided violations visible.
            self.assertTrue(all(line.get_drawstyle() == 'steps-post' for line in ax.lines[:3]))
            self.assertTrue((out / 'conditional_portfolio_ecdf.png').exists())
            caption = fig.axes[1].texts[0].get_text()
            for label in ('H-guided / truth', 'H-guided filtered / truth',
                          'H-guided / rejection', 'H-guided filtered / rejection'):
                self.assertIn(label, caption)
            from synthetic_benchmark.reporting import plot_conditional_comparison
            with patch.object(plt, 'show'):
                plot_conditional_comparison(guided, truth, guided[:0], pop, 0., 'gaussian', out)
            comparison = plt.gcf()
            self.assertEqual(len(comparison.axes), 4)
            for panel in comparison.axes:
                labels = panel.get_legend_handles_labels()[1]
                self.assertIn('H-guided DFM', labels)
                self.assertTrue(any('filtered' in label and 'n=0' in label for label in labels))
            plt.close(comparison)
            plt.close(fig)

    def test_h_comparison_uses_requested_cdf_and_refined_filtered_joint_samples(self):
        pop = make_population(3, 'gaussian', 3)
        raw_samples = np.array([[-4., -2., -3.], [-2., -1., -2.], [8., 7., 9.]])
        refined_samples = np.array([[-3., -2., -1.], [-1., -2., -1.], [4., 5., 6.]])
        truth = np.array([[-2.5, -1.5, -2.], [-1.5, -1., -1.], [-.5, -1., -.5]])
        rejection = np.array([[-3., -1., -2.], [-1., -.5, -1.]])
        raw = dict(population=pop, threshold=0., guided_samples=raw_samples,
                   filtered_guided_samples=raw_samples[:2])
        refined = dict(population=pop, threshold=0., guided_samples=refined_samples,
                       filtered_guided_samples=refined_samples[:2],
                       true_conditional_samples=truth, rejection_samples=rejection,
                       rejection_diagnostics={'rejection': {'status': 'complete'}})
        with tempfile.TemporaryDirectory() as tmp, patch.object(plt, 'show'):
            path = plot_h_refinement_comparison(raw, refined, Path(tmp))
            self.assertTrue(path.is_file())
            fig = plt.gcf()
            try:
                self.assertEqual(len(fig.axes), 3)
                cdf, qq, projection = fig.axes
                curve_labels = [line.get_label() for line in cdf.lines]
                expected_labels = [
                    'Raw H (no refinements), unfiltered (n=3)',
                    'Raw H (no refinements), event filtered (n=2)',
                    'Refined H (5 refinements), unfiltered (n=3)',
                    'Refined H (5 refinements), event filtered (n=2)',
                    'DFM rejection sampling (n=2)', 'Exact conditional CDF', 'Event threshold C',
                ]
                self.assertEqual(curve_labels, expected_labels)
                # Each empirical curve uses its own complete source draw; the
                # unfiltered event violations remain visible on the CDF axis.
                for line, samples in zip(cdf.lines, (raw_samples, raw_samples[:2], refined_samples,
                                                     refined_samples[:2], rejection)):
                    self.assertEqual(line.get_drawstyle(), 'steps-post')
                    np.testing.assert_allclose(line.get_xdata()[1:-1],
                                               np.sort(portfolio_values(samples, pop.w)))
                exact = cdf.lines[5]
                np.testing.assert_allclose(exact.get_ydata(), pop.cdf(exact.get_xdata(), c=0.))
                self.assertGreater(cdf.get_xlim()[1], max(raw_samples @ pop.w))

                # Q–Q: event-filtered Raw H, event-filtered Refined H, then DFM rejection.
                self.assertEqual(len(qq.collections), 3)
                q = np.linspace(.005, .995, 100)
                for collection, samples in zip(qq.collections,
                                               (raw_samples[:2], refined_samples[:2], rejection)):
                    expected_qq = np.column_stack((np.quantile(truth @ pop.w, q),
                                                    np.quantile(samples @ pop.w, q)))
                    np.testing.assert_allclose(collection.get_offsets(), expected_qq)
                self.assertEqual(len(qq.lines), 1)
                np.testing.assert_allclose(qq.lines[0].get_xdata(), qq.lines[0].get_ydata())
                self.assertEqual(len(projection.collections), 2)
                np.testing.assert_allclose(projection.collections[0].get_offsets(), truth @ pop.B[:, :2])
                np.testing.assert_allclose(projection.collections[1].get_offsets(),
                                           refined_samples[:2] @ pop.B[:, :2])
                self.assertEqual([c.get_label() for c in projection.collections],
                                 ['True conditional', 'Refined H-guided DFM (event filtered)'])
            finally:
                plt.close(fig)

    def test_failed_numerical_reference_not_used_for_metrics(self):
        pop = make_population(3, 'gaussian', 3)
        settings = ConditionalAssessmentConfig(rejection_n=10, rejection_batch_size=4,
                    rejection_max_proposals=12, rejection_reference_max_proposals=4, projections=2)
        batches = [torch.full((4, 3), -1.), torch.full((4, 3), float('nan')),
                   torch.full((4, 3), -1.)]
        x = -np.ones((3, 3))
        with patch('synthetic_benchmark.conditional.generate_samples', side_effect=batches):
            result = assess_conditional_samples(ToyReference(), pop, 0., x, x.copy(), x.copy(), steps=2, settings=settings)
        self.assertEqual(len(result['samples']['rejection']), 4)  # Preserve evidence.
        self.assertEqual(result['rejection_diagnostics']['rejection']['status'], 'nonfinite_output')
        row = result['comparison_metrics'].iloc[0]
        self.assertEqual(row.n_used, 0)
        self.assertTrue(np.isnan(row.portfolio_w1))
        self.assertEqual(row.rejection_status, 'nonfinite_output')
        self.assertEqual(row.n_right_saved, 4)

    def test_unavailable_truth_reference_has_no_dependent_substitution(self):
        pop = make_population(3, 'gaussian', 3)
        settings = ConditionalAssessmentConfig(rejection_n=2, rejection_batch_size=2,
                    rejection_max_proposals=2, rejection_reference_max_proposals=2, projections=2)
        x = -np.ones((3, 3))
        with patch('synthetic_benchmark.conditional.generate_samples', return_value=torch.full((2, 3), -1.)), \
             patch.object(pop, 'sample', side_effect=FloatingPointError('nonfinite truth sample')):
            result = assess_conditional_samples(ToyReference(), pop, 0., x, x.copy(), x.copy(), steps=2, settings=settings)
        self.assertEqual(result['samples']['truth_reference'].shape, (0, 3))
        self.assertEqual(result['truth_reference_diagnostics']['status'], 'unavailable_nonfinite')
        self.assertEqual(result['truth_reference_diagnostics']['shortfall'], 3)
        reference_rows = result['comparison_metrics'].query("row_type == 'MC reference' and left == 'truth'")
        self.assertTrue(reference_rows.portfolio_w1.isna().all())
        self.assertTrue((reference_rows.status == 'reference_shortfall').all())

    def test_settings_validate_independent_seeds(self):
        settings = ConditionalAssessmentConfig()
        for field, value in [('rejection_n', 0), ('rejection_reference_max_proposals', -1),
                             ('projections', True), ('matching_seed', -1),
                             ('truth_reference_seed', settings.rejection_seed)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(settings, **{field: value})


if __name__ == '__main__':
    unittest.main()
