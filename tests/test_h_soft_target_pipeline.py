"""Seeds, option routing, legacy metadata and cache provenance for soft targets."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from conditioning import ConditioningConfig
from h_model import HNetwork
from synthetic_benchmark.configuration import validate_branch_seeds
from synthetic_benchmark.conditional import (
    _check_raw_h, _normalized_training_options, _raw_h_options, fit_and_evaluate_h,
    sample_h_training_data, split_h_options,
)
from synthetic_benchmark.soft_targets import (
    construction_identity, load_target_cache, save_target_cache, target_construction,
)
import test_h_network

BASE = dict(train=71, validation=72, test=73, shuffle=74)
BRANCH = dict(train=171, validation=172, test=173)
COUNTS = (('train', 10), ('validation', 6), ('test', 6))


def collect(model, **kwargs):
    options = dict(steps=8, seeds=BASE, path_batch_size=4, device='cpu')
    options.update(kwargs)
    return sample_h_training_data(model, COUNTS, **options)


def window_of(model, rows=3):
    return (rows + .2) * model.horizon / 8


class BranchSeedTests(unittest.TestCase):
    def test_branch_seeds_validated_against_base_and_each_other(self):
        self.assertEqual(validate_branch_seeds(dict(BRANCH, extra=1), BASE), BRANCH)
        for bad in (dict(BRANCH, train=BASE['shuffle']),
                    dict(BRANCH, validation=BASE['train'] + 2**32),
                    dict(BRANCH, test=BRANCH['train'] + 2**32),
                    dict(BRANCH, test=True), dict(BRANCH, test=-1), dict(BRANCH, test=2**63),
                    {'train': 1, 'validation': 2}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_branch_seeds(bad, BASE)
        with self.assertRaises(ValueError):
            validate_branch_seeds([1, 2, 3], BASE)
        with self.assertRaises(ValueError):
            validate_branch_seeds(BRANCH, BASE, names=('train', 'train'))

    def test_collection_rejects_bad_branch_settings_before_sampling(self):
        model = test_h_network.HNetworkTests().fixture()
        window = window_of(model)
        with patch('h_model.generate_samples') as sampler:
            for kwargs in (
                    dict(soft_target_mode='continuation', soft_target_window=window, branch_count=2),
                    dict(soft_target_mode='continuation', soft_target_window=window, branch_count=2,
                         branch_seeds=dict(BRANCH, train=BASE['train'])),
                    dict(soft_target_mode='continuation', soft_target_window=window, branch_count=2,
                         branch_seeds={'train': 1}),
                    dict(branch_count=2, branch_seeds=BRANCH),
                    dict(soft_target_mode='reuse_final', soft_target_window=window, branch_count=2),
                    dict(soft_target_mode='nope')):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    collect(model, **kwargs)
            sampler.assert_not_called()

    def test_branch_settings_leave_base_splits_unchanged_and_split_by_parent(self):
        model = test_h_network.HNetworkTests().fixture()
        window = window_of(model)
        plain = collect(model)
        timings = {}
        soft = collect(model, soft_target_mode='continuation', soft_target_window=window,
                       branch_count=3, branch_seeds=BRANCH, branch_batch_size=5, diagnostics=timings)
        other = collect(model, soft_target_mode='continuation', soft_target_window=window,
                        branch_count=6, branch_seeds=BRANCH, branch_batch_size=7)
        for split, count in COUNTS:
            for name in ('features', 'labels', 'last_probabilities', 'last_states'):
                for candidate in (soft, other):
                    torch.testing.assert_close(getattr(candidate[split], name),
                                               getattr(plain[split], name), rtol=0, atol=0)
            self.assertEqual(len(soft[split].soft_targets), count)       # one row per parent path
            self.assertEqual(timings[split]['branch_state_steps'],       # rows 5,6 -> 2+1 updates
                             count * 3 * (2 + 1))
            self.assertGreaterEqual(timings[split]['collection_seconds'], timings[split]['branch_seconds'])
        # distinct named streams: the splits' targets are not copies of each other
        self.assertFalse(torch.equal(soft['train'].soft_targets[:6], soft['validation'].soft_targets))


class OptionRoutingTests(unittest.TestCase):
    def test_sampler_only_keys_never_reach_train_h(self):
        training, sampler = split_h_options(dict(
            loss='bce', soft_target_mode='continuation', soft_target_window=.3,
            branch_count=4, branch_seeds=BRANCH, branch_batch_size=8))
        self.assertEqual(set(training), {'loss', 'soft_target_mode', 'soft_target_window'})
        self.assertEqual(set(sampler), {'branch_count', 'branch_seeds', 'branch_batch_size'})
        self.assertEqual(split_h_options(None), ({}, {}))
        model = test_h_network.HNetworkTests().fixture()
        window = window_of(model)
        data = collect(model, soft_target_mode='continuation', soft_target_window=window,
                       branch_count=2, branch_seeds=BRANCH)
        construction = target_construction(
            model, mode='continuation', window=window, steps=8, split_counts=COUNTS, stream_seeds=BASE,
            path_batch_size=4, dfm_checkpoint=__file__, branch_count=2, branch_seeds=BRANCH)
        frozen = {k: v.clone() for k, v in model.reference.state_dict().items()}
        options = dict(loss='bce', use_last_step_targets=True, soft_target_mode='continuation',
                       soft_target_window=window, branch_count=2, branch_seeds=BRANCH,
                       branch_batch_size=None)
        selection, history, _, test = fit_and_evaluate_h(
            model, model.reference, data, frozen, epochs=2, batch_size=4, lr=1e-3, seeds=BASE,
            device='cpu', training_options=options, target_construction=construction)
        self.assertEqual(len(history), 2)
        self.assertNotIn('branch_count', selection['training_options'])
        self.assertEqual(selection['target_construction'], construction)
        self.assertIn('configured_scores', test)
        self.assertIn('binary_brier', selection['validation_scores'])
        for kwargs in (dict(target_construction=dict(construction, window=window * 2)),
                       dict(target_construction=dict(construction, mode='reuse_final')),
                       dict(target_construction=dict(construction, branch_count=9))):
            with self.subTest(kwargs=kwargs), patch('synthetic_benchmark.conditional.train_h') as trainer:
                with self.assertRaises(ValueError):
                    fit_and_evaluate_h(model, model.reference, data, frozen, epochs=1, batch_size=4,
                                       lr=1e-3, seeds=BASE, device='cpu', training_options=options, **kwargs)
                trainer.assert_not_called()
        # a different window than the collected rows is rejected by training itself
        with self.assertRaisesRegex(ValueError, 'do not match'):
            fit_and_evaluate_h(model, model.reference, data, frozen, epochs=1, batch_size=4, lr=1e-3,
                               seeds=BASE, device='cpu',
                               training_options=dict(options, soft_target_window=window_of(model, 4)))

    def test_legacy_runs_add_no_new_scores_or_metadata(self):
        model = test_h_network.HNetworkTests().fixture()
        data = collect(model)
        frozen = {k: v.clone() for k, v in model.reference.state_dict().items()}
        selection, _, _, test = fit_and_evaluate_h(
            model, model.reference, data, frozen, epochs=1, batch_size=4, lr=1e-3, seeds=BASE,
            device='cpu', training_options=_raw_h_options())
        self.assertEqual(selection['training_options'], _raw_h_options())
        self.assertNotIn('target_construction', selection)
        self.assertNotIn('configured_scores', test)

    def test_old_raw_h_metadata_still_validates_and_new_modes_do_not(self):
        model = test_h_network.HNetworkTests().fixture()
        training = dict(case='gaussian', seed=1, tail=.05, threshold=.1, n_train=1, n_validation=1,
                        n_test=1, epochs=1, lr=1e-3, batch_size=1, path_batch_size=1, steps=8,
                        stream_seeds=BASE, initialization_seed=3, reference_checkpoint='x')
        refined = dict(h_model=model)
        legacy = dict(training, training_options=_raw_h_options())
        _check_raw_h(model, legacy, refined, training)
        explicit = dict(training, training_options=dict(_raw_h_options(), soft_target_mode='off',
                                                        soft_target_window=None))
        _check_raw_h(model, explicit, refined, training)
        self.assertEqual(_normalized_training_options(_raw_h_options())['soft_target_mode'], 'off')
        self.assertEqual(_raw_h_options(), dict(
            normalize_inputs=False, loss='brier', late_time_fraction=0., last_step_fraction=0.,
            use_last_step_targets=False, times_per_trajectory=None))
        for mode in ('reuse_final', 'continuation'):
            bad = dict(training, training_options=dict(_raw_h_options(), soft_target_mode=mode,
                                                       soft_target_window=.1))
            with self.assertRaisesRegex(ValueError, 'five refinements'):
                _check_raw_h(model, bad, refined, training)


class CacheProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dfm = Path(self.tmp.name) / 'dfm.pt'
        self.dfm.write_bytes(b'frozen dfm bytes')
        self.model = test_h_network.HNetworkTests().fixture()
        self.window = window_of(self.model)
        self.options = dict(mode='continuation', window=self.window, steps=8, split_counts=COUNTS,
                            stream_seeds=BASE, path_batch_size=4, dfm_checkpoint=self.dfm,
                            branch_count=3, branch_seeds=BRANCH, branch_batch_size=5)
        self.data = collect(self.model, soft_target_mode='continuation', soft_target_window=self.window,
                            branch_count=3, branch_seeds=BRANCH, branch_batch_size=5)

    def construction(self, model=None, **changes):
        return target_construction(model or self.model, **dict(self.options, **changes))

    def test_construction_records_resolved_rows_and_roundtrips_cache(self):
        record = self.construction()
        self.assertEqual(record['time_indices'], [5, 6, 7])
        dt = self.model.horizon / 8
        for remaining, rows in zip(record['remaining_times'], (3, 2, 1)):
            self.assertAlmostEqual(remaining, rows * dt)
        for key in ('version', 'step_size', 'original_steps', 'branch_count', 'branch_seeds',
                    'branch_batch_size', 'stream_seeds', 'split_counts', 'path_batch_size', 'dtype',
                    'device', 'event', 'dfm'):
            self.assertIn(key, record)
        self.assertEqual(len(record['dfm']['sha256']), 64)
        self.assertEqual(self.construction(mode='reuse_final', branch_count=None)['branch_seeds'], None)
        cache = Path(self.tmp.name) / 'cache'
        save_target_cache(cache, record, self.data)
        with self.assertRaises(FileExistsError):
            save_target_cache(cache, record, self.data)
        restored = load_target_cache(cache, record, self.model)
        for split in self.data:
            for name in ('features', 'labels', 'last_probabilities', 'soft_targets',
                         'soft_target_standard_errors', 'soft_target_indices'):
                torch.testing.assert_close(getattr(restored[split], name), getattr(self.data[split], name),
                                           rtol=0, atol=0)

    def test_any_change_of_target_inputs_invalidates_the_cache(self):
        record = self.construction()
        cache = Path(self.tmp.name) / 'cache'
        save_target_cache(cache, record, self.data)
        other_event = HNetwork(self.model.reference, ConditioningConfig(self.model.config.w,
                                                                       self.model.C + .05))
        other_weights = HNetwork(self.model.reference, ConditioningConfig(
            tuple(reversed(self.model.config.w)), self.model.C))
        changes = {
            'window': dict(window=window_of(self.model, 4)),
            'mode': dict(mode='reuse_final', branch_count=None, branch_seeds=None, branch_batch_size=None),
            'grid': dict(steps=16),
            'branch count': dict(branch_count=4),
            'branch seeds': dict(branch_seeds=dict(BRANCH, test=999)),
            'branch batching': dict(branch_batch_size=6),
            'base seeds': dict(stream_seeds=dict(BASE, train=1)),
            'split counts': dict(split_counts=(('train', 11), ('validation', 6), ('test', 6))),
            'base batching': dict(path_batch_size=5),
        }
        for name, change in changes.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, 'Stale target cache'):
                load_target_cache(cache, self.construction(**change), self.model)
        for name, model in (('event threshold', other_event), ('event weights', other_weights)):
            with self.subTest(name), self.assertRaisesRegex(ValueError, 'Stale target cache'):
                load_target_cache(cache, self.construction(model), self.model)
        self.dfm.write_bytes(b'a different frozen dfm')
        with self.assertRaisesRegex(ValueError, 'Stale target cache.*dfm'):
            load_target_cache(cache, self.construction(), self.model)
        self.assertNotEqual(construction_identity(record), construction_identity(self.construction()))
        # a renamed checkpoint path with identical bytes is the same DFM
        self.dfm.write_bytes(b'frozen dfm bytes')
        moved = Path(self.tmp.name) / 'copy.pt'
        moved.write_bytes(b'frozen dfm bytes')
        load_target_cache(cache, self.construction(dfm_checkpoint=moved), self.model)

    def test_tampered_arrays_and_construction_version_are_rejected(self):
        record = self.construction()
        cache = Path(self.tmp.name) / 'cache'
        save_target_cache(cache, record, self.data)
        with self.assertRaisesRegex(ValueError, 'Stale target cache'):
            load_target_cache(cache, dict(record, version='h-soft-targets-v0'), self.model)
        tampered = torch.load(cache / 'splits.pt', weights_only=True)
        tampered['train']['soft_targets'] = tampered['train']['soft_targets'] * .5
        tampered['train']['soft_targets'][:, -1] = tampered['train']['last_probabilities']
        torch.save(tampered, cache / 'splits.pt')
        with self.assertRaisesRegex(ValueError, 'digests'):
            load_target_cache(cache, record, self.model)


if __name__ == '__main__':
    unittest.main()
