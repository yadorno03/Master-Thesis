"""Reject overlapping random streams before they can contaminate H evaluation."""
from dataclasses import replace
from unittest.mock import patch
import unittest

import numpy as np
import torch

from synthetic_benchmark.configuration import (
    ConditionalAssessmentConfig, Config, validate_h_seeds,
)
from synthetic_benchmark.conditional import run_conditional_benchmark, sample_h_training_data
from test_conditional_initial import GaussianH


class SyntheticSeedTests(unittest.TestCase):
    def setUp(self):
        self.seeds = dict(train=71, validation=72, test=73, shuffle=74)

    def test_validation_copies_and_normalizes_requested_seeds(self):
        seeds = dict(self.seeds, train=np.int64(71), unused=71)
        actual = validate_h_seeds(seeds)
        self.assertEqual(actual, self.seeds)
        self.assertIsNot(actual, seeds)
        self.assertIs(type(actual['train']), int)
        actual['train'] = 99
        self.assertEqual(seeds['train'], 71)
        self.assertEqual(validate_h_seeds({'test': 73}, names=('test',)), {'test': 73})

    def test_missing_invalid_and_reused_seeds_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'mapping'):
            validate_h_seeds([71, 72, 73, 74])
        with self.assertRaisesRegex(ValueError, 'Missing H stream seeds: test'):
            validate_h_seeds({key: value for key, value in self.seeds.items() if key != 'test'})
        for value in (True, False, -1, 2**63, 71., '71', np.nan):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'integer'):
                validate_h_seeds(dict(self.seeds, test=value))
        for name in ('validation', 'test', 'shuffle'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'distinct seeds'):
                validate_h_seeds(dict(self.seeds, **{name: self.seeds['train']}))
        with self.assertRaisesRegex(ValueError, 'names'):
            validate_h_seeds(self.seeds, names=('train', 'train'))

    def test_cpu_aliases_are_rejected_for_h_and_assessment_streams(self):
        # These are valid, distinct manual_seed inputs but repeat CPU normal draws.
        a = torch.randn(64, generator=torch.Generator().manual_seed(71))
        b = torch.randn(64, generator=torch.Generator().manual_seed(71 + 2**32))
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'lower 32 bits'):
            validate_h_seeds(dict(self.seeds, validation=71 + 2**32))
        settings = ConditionalAssessmentConfig()
        with self.assertRaisesRegex(ValueError, 'lower 32 bits'):
            replace(settings, rejection_reference_seed=settings.rejection_seed + 2**32)

    def test_assessment_rejects_aliases_of_existing_streams_before_loading(self):
        settings = ConditionalAssessmentConfig()
        config = Config(d=3, k=3)
        with patch('synthetic_benchmark.conditional.load_frozen_dfm') as load:
            for guided_seed in (settings.rejection_seed, settings.rejection_seed + 2**32):
                with self.subTest(guided_seed=guided_seed), self.assertRaisesRegex(ValueError, 'differ'):
                    run_conditional_benchmark(config, 'unused-output', guided_seed=guided_seed,
                                              assessment_settings=settings)
            load.assert_not_called()

    def test_duplicate_trajectory_splits_are_rejected_before_sampling(self):
        model = GaussianH()
        # Reproduce the exact leakage rather than testing only integer inequality.
        repeated = [model.sample_trajectories(
            64, num_steps=8, generator=torch.Generator().manual_seed(71), batch_size=32,
        ) for _ in range(2)]
        torch.testing.assert_close(repeated[0].features, repeated[1].features, rtol=0, atol=0)
        torch.testing.assert_close(repeated[0].labels, repeated[1].labels, rtol=0, atol=0)
        with patch.object(model, 'sample_trajectories', wraps=model.sample_trajectories) as sampler:
            for second_seed in (71, 71 + 2**32):
                with self.subTest(second_seed=second_seed), self.assertRaisesRegex(ValueError, 'distinct seeds'):
                    sample_h_training_data(
                        model, (('train', 64), ('validation', 64)), steps=8,
                        seeds={'train': 71, 'validation': second_seed},
                        path_batch_size=32, device='cpu',
                    )
            sampler.assert_not_called()

    def test_distinct_splits_remain_reproducible_and_support_iterable_counts(self):
        model = GaussianH()
        kwargs = dict(steps=8, seeds=self.seeds, path_batch_size=32, device='cpu')
        counts = (('train', 64), ('validation', 64), ('test', 64))
        first = sample_h_training_data(model, iter(counts), **kwargs)
        repeated = sample_h_training_data(model, counts, **kwargs)
        for name in first:
            torch.testing.assert_close(first[name].features, repeated[name].features, rtol=0, atol=0)
            torch.testing.assert_close(first[name].labels, repeated[name].labels, rtol=0, atol=0)
        self.assertFalse(torch.equal(first['train'].features, first['validation'].features))
        self.assertFalse(torch.equal(first['train'].features, first['test'].features))

    def test_runner_rejects_bad_seeds_before_loading_a_checkpoint(self):
        config = Config(d=3, k=3)
        invalid = [dict(self.seeds, validation=71),
                   dict(self.seeds, test=71 + 2**32),
                   {key: value for key, value in self.seeds.items() if key != 'shuffle'}]
        with patch('synthetic_benchmark.conditional.load_frozen_dfm') as load:
            for seeds in invalid:
                with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                    run_conditional_benchmark(config, 'unused-output', train_seeds=seeds)
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
