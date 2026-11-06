"""Tiny end-to-end execution of the target-only ablation; never full-scale training."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from synthetic_benchmark.configuration import Config, ConditionalAssessmentConfig, Streams
from synthetic_benchmark.conditional import load_frozen_dfm
from synthetic_benchmark.soft_target_experiment import (
    aggregate_repetitions, build_residual_reference, check_distinct_streams, common_size_comparison,
    conditional_moment_row, continuation_setting, data_seed_sets, default_arms, distribution_row,
    evaluate_residuals, load_reference_samples, load_soft_target_results, run_soft_target_experiment,
)
from synthetic_benchmark.unconditional import (
    fit_and_save_dfm, initialize_unconditional_run, prepare_unconditional_case,
)
from synthetic_conditional_truth import make_population

TINY = Config(d=12, k=4, n_train=96, n_val=48, n_eval=48, dfm_epochs=2, steps=12, width=8,
              batch=32, lr=1e-3, device='cpu')
ASSESSMENT = ConditionalAssessmentConfig(
    rejection_n=24, rejection_batch_size=64, rejection_max_proposals=2000,
    rejection_reference_max_proposals=2000, rejection_seed=62003, rejection_reference_seed=62004,
    truth_reference_seed=62005, projection_seed=62006, matching_seed=62007, projections=4)
SETTINGS = dict(n_train=96, n_validation=48, n_test=48, epochs=2, batch_size=32, path_batch_size=32,
                branch_batch_size=64, tail=0.3, widths=(8, 8), n_samples=48, sample_batch_size=32,
                assessment_settings=ASSESSMENT, residual_states=4, residual_branches=6,
                residual_remaining=(1.4, 2.7), subset_repeats=3)


def make_frozen_run(root):
    """A real (barely trained) unconditional run: DFM checkpoint with population provenance."""
    run = Path(root) / 'source'
    initialize_unconditional_run(run, TINY)
    folder, streams, population = prepare_unconditional_case(run, TINY, 'gaussian', 42)
    train = population.sample(TINY.n_train, streams.np('train'))
    val = population.sample(TINY.n_val, streams.np('validation'))
    fit_and_save_dfm(train, val, TINY, streams, population, 'gaussian', folder)
    return run


class MetricHelperTests(unittest.TestCase):
    def setUp(self):
        self.population = make_population(12, 'gaussian', k=4)
        self.threshold = self.population.threshold(.3)

    def test_helpers_report_explicit_empty_singleton_and_ok_statuses(self):
        truth = self.population.sample(200, np.random.default_rng(1), c=self.threshold)
        values = truth @ self.population.w
        for samples, status in ((truth[:0], 'empty'), (truth[:1], 'singleton'), (truth, 'ok')):
            row = conditional_moment_row(samples, self.population, self.threshold)
            self.assertEqual((row['n'], row['status']), (len(samples), status))
            self.assertEqual(np.isfinite(row['mean_error']), status == 'ok')
            row = distribution_row(samples @ self.population.w, self.population, self.threshold,
                                   values, values)
            self.assertEqual(row['status'], status)
            self.assertEqual(np.isfinite(row['ks_exact']), status == 'ok')
        self.assertLess(distribution_row(values, self.population, self.threshold, values, values)['ks_exact'], .2)

    def test_common_size_subsets_are_value_independent_and_never_upsample(self):
        rng = np.random.default_rng(2)
        arms = {'a': rng.normal(size=50) - 5, 'b': rng.normal(size=20) - 5, 'empty': np.empty(0),
                'one': np.array([-5.])}
        refs = dict(rejection=rng.normal(size=100) - 5, truth=rng.normal(size=10) - 5)
        table = common_size_comparison(arms, self.population, self.threshold, refs, seed=7, repeats=4)
        used = dict(zip(table.arm, table.n_used))
        self.assertEqual(used, {'a': 20, 'b': 20, 'empty': 0, 'one': 0})
        self.assertEqual(dict(zip(table.arm, table.n_available)), {k: len(v) for k, v in arms.items()})
        self.assertEqual(dict(zip(table.arm, table.status)),
                         {'a': 'ok', 'b': 'ok', 'empty': 'insufficient', 'one': 'insufficient'})
        # the 10-row truth reference is too small for n=20: explicit NaN, no substitution
        self.assertTrue(table.w1_truth_mean.isna().all())
        self.assertTrue(table.w1_rejection_mean.loc[table.status == 'ok'].notna().all())
        again = common_size_comparison(arms, self.population, self.threshold, refs, seed=7, repeats=4)
        pd.testing.assert_frame_equal(table, again)
        shifted = {k: v + 1000 * (np.arange(len(v)) == 0) for k, v in arms.items()}  # one outlier
        self.assertEqual(common_size_comparison(shifted, self.population, self.threshold, refs,
                                                seed=7, repeats=4).n_used.tolist(), [20, 20, 0, 0])
        none = common_size_comparison({'x': np.empty(0)}, self.population, self.threshold, refs, seed=1)
        self.assertEqual(none.status.tolist(), ['insufficient'])

    def test_arm_definitions_streams_and_aggregation(self):
        arms = default_arms()
        self.assertEqual([a['mode'] for a in arms.values()], ['off', 'reuse_final', 'reuse_final', 'continuation'])
        self.assertEqual(continuation_setting(arms), (0.1, 8))
        arms['x'] = dict(label='x', mode='continuation', window=.5, branch_count=2)
        with self.assertRaisesRegex(ValueError, 'one continuation'):
            continuation_setting(arms)
        self.assertIsNone(continuation_setting({'a': arms['existing_refined']}))
        with self.assertRaisesRegex(ValueError, 'reuse the same stream'):
            check_distinct_streams(dict(a=5, b=5 + 2**32))
        base, branch = data_seed_sets(1)
        check_distinct_streams({**{f'b{k}': v for k, v in base.items()}, **{f'r{k}': v for k, v in branch.items()},
                                **{f'o{k}': v for k, v in data_seed_sets(0)[0].items()}})
        table = pd.DataFrame(dict(arm=['a'] * 4, data_offset=[0, 0, 1, 1], init_seed=[1, 2, 1, 2],
                                  m=[1., 3., 10., 14.]))
        row = aggregate_repetitions(table, ['m']).iloc[0]
        self.assertEqual((row.n_runs, row.m_mean), (4, 7.))
        self.assertAlmostEqual(row.m_sd_init_seed, np.mean([np.sqrt(2), np.sqrt(8)]))
        self.assertAlmostEqual(row.m_sd_data_offset, np.std([2., 12.], ddof=1))


class TinyExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.source = make_frozen_run(cls.tmp.name)
        cls.dfm_bytes = (cls.source / 'gaussian_seed42_primary' / 'dfm.pt').read_bytes()
        cls.output = Path(cls.tmp.name) / 'ablation'
        cls.arms = default_arms(branch_count=3)
        # dt = 7.995/12 = 0.666: windows select the last 2, 4 and 2 preterminal rows
        cls.arms['reuse_0.1']['window'] = cls.arms['continuation_0.1_M3']['window'] = 1.4
        cls.arms['reuse_0.5']['window'] = 2.7
        cls.result = run_soft_target_experiment(
            cls.source, cls.output, arms=cls.arms, init_seeds=(2027, 2028), data_offsets=(0,),
            **SETTINGS)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_outputs_provenance_and_unchanged_frozen_dfm(self):
        output = self.output
        manifest = json.loads((output / 'soft_target_comparison.json').read_text())
        self.assertFalse((output / 'h_comparison.json').exists())
        self.assertEqual(len(manifest['repetitions']), 2)
        summary = manifest['repetitions'][0]['target_summary']
        self.assertGreater(summary['continuation']['mean_standard_error'], 0)
        self.assertEqual(summary['continuation']['rows'], [10, 11])
        self.assertIn('fraction_between_0p01_0p99', summary['last_step_probability'])
        self.assertEqual(set(manifest['repetitions'][0]['arms']), set(self.arms))
        self.assertEqual((self.source / 'gaussian_seed42_primary' / 'dfm.pt').read_bytes(), self.dfm_bytes)
        self.assertEqual(manifest['frozen_dfm']['sha256'],
                         json.loads((output / 'config.json').read_text())['dfm']['sha256'])
        table = self.result['table']
        self.assertEqual(len(table), 8)
        # identical initial weights across arms, per repetition; different init seeds differ
        digests = table.groupby('init_seed').initial_weights_sha256.nunique()
        self.assertTrue((digests == 1).all())
        self.assertEqual(table.initial_weights_sha256.nunique(), 2)
        for name, arm in manifest['repetitions'][0]['arms'].items():
            training = json.loads((Path(arm['h_checkpoint']).parent / 'training.json').read_text())
            construction = training.get('target_construction')
            if self.arms[name]['mode'] == 'off':
                self.assertIsNone(construction)
                self.assertNotIn('soft_target_mode', training['training_options'])
                continue
            self.assertEqual(construction['mode'], self.arms[name]['mode'])
            self.assertEqual(training['training_options']['soft_target_mode'], construction['mode'])
            self.assertNotIn('branch_count', training['training_options'])
            self.assertEqual(construction['time_indices'][-1], 11)
            self.assertEqual(len(construction['time_indices']), 4 if name == 'reuse_0.5' else 2)
            self.assertEqual(len(construction['time_indices']), len(construction['remaining_times']))
            self.assertEqual(construction['original_steps'], 12)
            self.assertEqual(construction['construction_identity'], arm['construction_identity'])
            self.assertEqual(set(construction['target_arrays']), {'train', 'validation', 'test'})
            self.assertIn('last_probabilities', construction['target_arrays']['train'])
            self.assertEqual('soft_targets' in construction['target_arrays']['train'], True)  # shared collection
            self.assertEqual(len(construction['dfm']['sha256']), 64)
            self.assertEqual(construction['branch_count'], 3 if construction['mode'] == 'continuation' else None)
        # shared base splits: identical event rate in the common test split across arms
        rates = {json.loads((Path(a['h_checkpoint']).parent / 'test_metrics.json').read_text())['test_event_rate']
                 for a in manifest['repetitions'][0]['arms'].values()}
        self.assertEqual(len(rates), 1)
        self.assertTrue((output / 'data_offset0' / 'target_cache' / 'identity.json').is_file())

    def test_metrics_include_required_separate_evaluations(self):
        table = self.result['table']
        for column in ('event_rate', 'n_filtered', 'filtered_fraction', 'init_proposals', 'init_acceptance_rate',
                       'sampling_seconds', 'event_enrichment', 'dfm_proposal_acceptance', 'filtered_ks_exact',
                       'filtered_w1_rejection', 'filtered_w1_truth', 'unfiltered_ks_exact',
                       'filtered_sliced_w1_rejection', 'filtered_covariance_trace_ratio',
                       'score_test_binary_brier', 'score_test_binary_bce', 'score_test_configured_bce',
                       'optimization_seconds', 'selected_epoch', 'collection_seconds',
                       'target_construction_seconds', 'peak_memory_mb', 'test_last_probability_rmse'):
            self.assertIn(column, table)
        continuation = table[table['mode'] == 'continuation']
        self.assertTrue((continuation.target_construction_seconds > 0).all())
        self.assertTrue((table[table['mode'] != 'continuation'].target_construction_seconds == 0).all())
        # common binary scores use the same test labels for every arm: one noise floor
        self.assertTrue((table.score_test_binary_brier > 0).all())
        self.assertTrue(((table.n_filtered <= table.n_requested)).all())
        residuals = self.result['residuals']
        self.assertEqual(set(residuals.arm), set(self.arms))
        self.assertEqual(sorted(residuals.row.unique()), [8, 10, 11])
        last = residuals[residuals.row == 11]
        self.assertTrue((last.mean_reference_se == 0).all())
        late = residuals[residuals.row < 11]
        self.assertTrue((late.mean_reference_se > 0).all())
        self.assertTrue((late.rmse_noise_corrected <= late.rmse + 1e-12).all())
        common = self.result['common']
        self.assertEqual(set(common.arm), set(self.arms))
        summary = self.result['summary']
        self.assertEqual(set(summary.arm), set(self.arms))
        self.assertTrue(summary.filtered_ks_exact_sd_init_seed.notna().all())
        self.assertTrue(summary.filtered_ks_exact_sd_data_offset.isna().all())   # one data offset

    def test_reload_for_notebook_and_reference_reuse_verification(self):
        loaded = load_soft_target_results(self.output)
        self.assertEqual(set(loaded['portfolios']), set(self.arms))
        self.assertEqual(len(loaded['truth']), SETTINGS['n_samples'])
        manifest = loaded['manifest']
        first = manifest['repetitions'][0]['arms']['existing_refined']
        samples = Path(first['samples'])
        _, _, population, _, reference = load_frozen_dfm(
            self.source / 'gaussian_seed42_primary' / 'dfm.pt', 'gaussian', 'cpu')
        options = dict(dfm_checkpoint=self.source / 'gaussian_seed42_primary' / 'dfm.pt',
                       population=population, threshold=loaded['threshold'], count=48, steps=12,
                       settings=ASSESSMENT)
        reused = load_reference_samples(samples, **options)
        np.testing.assert_array_equal(reused['samples']['truth'], first_truth := np.load(samples / 'conditional_samples.npz')['true_conditional_samples'])
        self.assertEqual(reused['source'], str(samples.resolve()))
        for change in (dict(threshold=loaded['threshold'] + 1e-3), dict(count=47), dict(steps=13),
                       dict(settings=replace(ASSESSMENT, projections=5))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                load_reference_samples(samples, **dict(options, **change))
        other = Path(self.tmp.name) / 'other_dfm.pt'
        other.write_bytes(b'different')
        with self.assertRaisesRegex(ValueError, 'different frozen DFM'):
            load_reference_samples(samples, **dict(options, dfm_checkpoint=other))

    def test_comparison_figure_handles_complete_and_missing_arms(self):
        from synthetic_benchmark.reporting import plot_soft_target_comparison
        loaded = load_soft_target_results(self.output)
        figures = Path(self.tmp.name) / 'figures'
        figures.mkdir()
        arguments = (loaded['truth'], loaded['rejection'], loaded['population'], loaded['threshold'],
                     loaded['table'], loaded['residuals'], loaded['curves'], figures)
        path = plot_soft_target_comparison(loaded['portfolios'], *arguments, show=False)
        self.assertEqual(path, figures / 'soft_target_comparison.png')
        self.assertGreater(path.stat().st_size, 10_000)
        # an arm without draws (failed sampling) and an empty filtered set stay visible
        partial = dict(loaded['portfolios'])
        partial.pop('reuse_0.5')
        partial['reuse_0.1'] = dict(unfiltered=partial['reuse_0.1']['unfiltered'], filtered=np.empty(0))
        path.unlink()
        plot_soft_target_comparison(partial, *arguments, show=False)
        self.assertTrue(path.is_file())

    def test_cost_estimate_trains_nothing_and_creates_no_output(self):
        out = Path(self.tmp.name) / 'estimate_only'
        estimate = run_soft_target_experiment(self.source, out, arms=self.arms, estimate_only=True,
                                              **dict(SETTINGS, n_samples=48))
        self.assertFalse(out.exists())
        for key in ('base_collection_seconds', 'continuation_target_seconds', 'training_seconds_per_arm',
                    'guided_sampling_seconds_per_arm', 'rejection_seconds_at_proposal_caps'):
            self.assertGreater(estimate[key], 0)
        # window selects rows 10 and 11: (12-1-10) = 1 update, times 3 branches
        self.assertEqual(estimate['continuation_state_steps_per_path'], 3)
        self.assertEqual(estimate['continuation_final_evaluations_per_path'], 3)

    def test_existing_directories_and_unsafe_outputs_are_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            run_soft_target_experiment(self.source, self.output, arms=self.arms, **SETTINGS)
        for unsafe in (self.source, self.source / 'new', self.source.parent):
            with self.subTest(unsafe=unsafe), self.assertRaisesRegex(ValueError, 'fresh directory'):
                run_soft_target_experiment(self.source, unsafe, arms=self.arms, **SETTINGS)

    def test_failed_initialization_is_explicit_and_other_arms_continue(self):
        out = Path(self.tmp.name) / 'failing'
        real = __import__('synthetic_benchmark.soft_target_experiment', fromlist=['x']).sample_h_guided
        calls = []

        def flaky(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError('Conditional initialization exhausted 10 proposals')
            return real(*args, **kwargs)
        arms = {k: v for k, v in self.arms.items() if k in ('existing_refined', 'reuse_0.1')}
        with patch('synthetic_benchmark.soft_target_experiment.sample_h_guided', side_effect=flaky):
            result = run_soft_target_experiment(self.source, out, arms=arms, **SETTINGS)
        table = result['table'].set_index('arm')
        self.assertEqual(table.loc['existing_refined', 'status'], 'complete')
        self.assertEqual(table.loc['reuse_0.1', 'status'], 'initialization_exhausted')
        self.assertTrue(pd.isna(table.loc['reuse_0.1', 'event_rate']))
        self.assertTrue(pd.isna(table.loc['reuse_0.1', 'filtered_ks_exact']))
        manifest = json.loads((out / 'soft_target_comparison.json').read_text())
        failed = manifest['repetitions'][0]['arms']['reuse_0.1']
        self.assertEqual(failed['sample_status'], 'initialization_exhausted')
        self.assertIn('exhausted', failed['sample_error'])
        self.assertEqual(set(result['portfolios']), {'existing_refined'})

    def test_target_cache_reuse_requires_matching_construction(self):
        out = Path(self.tmp.name) / 'cached'
        cache = self.output / 'data_offset0' / 'target_cache'
        with patch('synthetic_benchmark.soft_target_experiment.sample_h_training_data',
                   side_effect=AssertionError('resampled instead of loading the verified cache')):
            result = run_soft_target_experiment(self.source, out, arms=self.arms, target_cache=cache,
                                                init_seeds=(2027,), **SETTINGS)
        self.assertEqual(len(result['table']), 4)
        stale = Path(self.tmp.name) / 'stale'
        with self.assertRaisesRegex(ValueError, 'Stale target cache'):
            run_soft_target_experiment(self.source, stale, arms=self.arms, target_cache=cache,
                                       init_seeds=(2027,), **dict(SETTINGS, n_train=64))

    def test_residual_reference_streams_do_not_touch_training_streams(self):
        _, _, population, _, reference = load_frozen_dfm(
            self.source / 'gaussian_seed42_primary' / 'dfm.pt', 'gaussian', 'cpu')
        from synthetic_benchmark.conditional import initialize_h
        _, model, _ = initialize_h(reference, population, .3, (8, 8), 2027, log_remaining_time=True)
        global_state = torch.get_rng_state().clone()
        kwargs = dict(steps=12, remaining=(1.4, 2.7), n_states=4, branches=5, path_seed=9, branch_seed=10)
        first = build_residual_reference(model, **kwargs)
        again = build_residual_reference(model, **kwargs)
        self.assertTrue(torch.equal(torch.get_rng_state(), global_state))
        for j in first['rows']:
            torch.testing.assert_close(first['targets'][j], again['targets'][j], rtol=0, atol=0)
        table = evaluate_residuals(model, first)
        self.assertEqual(len(table), len(first['rows']))
        self.assertTrue((table.n_states == 4).all())


if __name__ == '__main__':
    unittest.main()
