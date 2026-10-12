from dataclasses import FrozenInstanceError
import unittest

import torch

from conditioning import ConditioningConfig, portfolio_event_in_model_coordinates


class PortfolioEventTests(unittest.TestCase):
    def test_configuration_owns_immutable_values(self):
        for weights in ([0.25, -0.5], torch.tensor([0.25, -0.5], requires_grad=True)):
            threshold = torch.tensor(-0.125, requires_grad=True)
            event = ConditioningConfig(weights, threshold)
            with torch.no_grad():
                weights[0] = 7.
                threshold.fill_(9.)
            self.assertEqual(event.w, (0.25, -0.5))
            self.assertEqual(event.C, -0.125)
            with self.assertRaises(FrozenInstanceError):
                event.C = 0.
            with self.assertRaises(FrozenInstanceError):
                event.w = (1., 0.)
        self.assertEqual(ConditioningConfig((0, 2), 0).w, (0., 2.))

    def test_invalid_event_inputs(self):
        invalid_weights = ([], 1., [[1., 2.]], [0., 0.], [float('nan')],
                           [float('inf')], [True, False], [1j])
        invalid_thresholds = ([0.], float('nan'), float('inf'), -float('inf'), True, 1j)
        for weights in invalid_weights:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                ConditioningConfig(weights, 0.)
        for threshold in invalid_thresholds:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                ConditioningConfig([1., 0.], threshold)

    def test_conversion_preserves_portfolio_margin_and_event(self):
        for dtype in (torch.float32, torch.float64):
            # Binary fractions make the threshold boundary exactly representable.
            weights = torch.tensor([0.5, -1., 0.], dtype=dtype)
            mean = torch.tensor([0.25, -0.125, 2.], dtype=dtype)
            std = torch.tensor([0.5, 2., 0.25], dtype=dtype)
            x = torch.tensor([[0., 0., 8.], [0., 0.125, -4.],
                              [0., 0.25, 1.]], dtype=dtype)
            originals = [value.clone() for value in (weights, mean, std)]
            event = portfolio_event_in_model_coordinates(weights, 0., mean, std)
            self.assertEqual(event.w, (0.25, -2., 0.))
            self.assertEqual(event.C, -0.25)
            raw_margin = (mean + std*x) @ weights
            model_margin = x @ torch.tensor(event.w, dtype=dtype) - event.C
            torch.testing.assert_close(model_margin, raw_margin, rtol=0, atol=0)
            self.assertEqual((raw_margin <= 0).tolist(), [False, True, True])
            self.assertTrue(torch.equal(raw_margin <= 0, model_margin <= 0))
            for actual, original in zip((weights, mean, std), originals):
                torch.testing.assert_close(actual, original, rtol=0, atol=0)

    def test_conversion_accepts_sequences_and_preserves_precision(self):
        event = portfolio_event_in_model_coordinates(
            [0.123456789012345, -2.], -0.3, (0., 0.), [1., 1.])
        self.assertEqual(event, ConditioningConfig([0.123456789012345, -2.], -0.3))

    def test_invalid_preprocessing_and_converted_event(self):
        invalid = (1., [1.], [[1., 1.]], [float('nan'), 1.],
                   [1., float('inf')], [True, False], [1j, 1.])
        for value in invalid:
            for mean, std in ((value, [1., 1.]), ([0., 0.], value)):
                with self.subTest(mean=mean, std=std), self.assertRaises(ValueError):
                    portfolio_event_in_model_coordinates([1., -1.], 0., mean, std)
        for std in ([0., 1.], [-1., 1.]):
            with self.assertRaises(ValueError):
                portfolio_event_in_model_coordinates([1., -1.], 0., [0., 0.], std)
        for weights, mean, std in (([1e308], [0.], [2.]),
                                   ([1e308], [2.], [1.]),
                                   ([1e-300], [0.], [1e-300])):
            with self.assertRaises(ValueError):
                portfolio_event_in_model_coordinates(weights, 0., mean, std)
