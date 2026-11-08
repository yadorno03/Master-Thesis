from dataclasses import FrozenInstanceError
import unittest

import torch

from conditioning import ConditioningConfig


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
