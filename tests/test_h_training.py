from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from h_model import train_h, terminal_label_loss
import test_h_network


class HTrainingTests(unittest.TestCase):
    def fixture(self, dtype=torch.float64):
        model = test_h_network.HNetworkTests().fixture(dtype)
        data = model.sample_trajectories(5, 3, generator=torch.Generator().manual_seed(38))
        return model, data

    def test_training_reduces_loss_preserves_reference_and_restores_modes(self):
        for dtype in (torch.float32, torch.float64):
            model, data = self.fixture(dtype)
            model.eval()
            model.network[0].train()
            modes = [m.training for m in model.modules()]
            before = {key: value.clone() for key, value in model.conditioning.reference.state_dict().items()}
            def loss():
                probabilities = torch.stack([model(t, data.features[:, j])
                                             for j, t in enumerate(data.times)], dim=1)
                return terminal_label_loss(probabilities, data.labels, data.weights).item()
            initial = loss()
            history = train_h(model, data, epochs=30, batch_size=5, lr=.03,
                              generator=torch.Generator().manual_seed(11))
            self.assertEqual(len(history), 30)
            self.assertLess(loss(), initial)
            self.assertEqual([m.training for m in model.modules()], modes)
            for key, value in model.conditioning.reference.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            self.assertTrue(all(not p.requires_grad and p.grad is None
                                for p in model.conditioning.reference.parameters()))

    def test_whole_trajectory_shuffle_partial_batch_weighting_and_reproducibility(self):
        model, data = self.fixture()
        batches = []
        def record(probabilities, labels, weights):
            value = terminal_label_loss(probabilities, labels, weights)
            batches.append((value.item(), labels.clone(), probabilities.shape))
            return value
        with patch('h_model.terminal_label_loss', side_effect=record):
            history = train_h(model, data, epochs=1, batch_size=2,
                              generator=torch.Generator().manual_seed(6))
        self.assertEqual([shape for _, _, shape in batches], [(2, 3, 1), (2, 3, 1), (1, 3, 1)])
        self.assertAlmostEqual(history[0], sum(value*len(labels) for value, labels, _ in batches)/5)
        order = torch.randperm(5, generator=torch.Generator().manual_seed(6))
        torch.testing.assert_close(torch.cat([labels for _, labels, _ in batches]), data.labels[order])
        duplicate, same_data = self.fixture()
        repeat = train_h(duplicate, same_data, epochs=1, batch_size=2,
                         generator=torch.Generator().manual_seed(6))
        self.assertEqual(history, repeat)
        for left, right in zip(model.parameters(), duplicate.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_valid_nonuniform_grid_matches_direct_time_integral(self):
        model, data = self.fixture()
        data = replace(data, times=data.times.new_tensor([0., .1, .8])*model.horizon,
                       weights=data.weights.new_tensor([.1, .7, .2]))
        losses = [(model(t, data.features[:, j])-data.labels).square().mean().item()
                  for j, t in enumerate(data.times)]
        expected = sum(weight*loss for weight, loss in zip((.1, .7, .2), losses))
        actual = train_h(model, data, epochs=1, batch_size=5,
                         generator=torch.Generator().manual_seed(16))
        self.assertAlmostEqual(actual[0], expected)

    def test_invalid_trajectory_contract_fails_before_optimizer_creation(self):
        model, data = self.fixture()
        bad = [replace(data, features=value) for value in
               (data.features[:0], data.features[:, :0], data.features[:, :, :2], data.features[0])]
        for field in ('times', 'features', 'labels', 'weights'):
            value = getattr(data, field)
            bad.extend(replace(data, **{field: v}) for v in
                       (value.float(), value*float('nan'), value.clone().requires_grad_()))
        bad.extend([replace(data, labels=data.labels+.5), replace(data, labels=data.labels[:, 0]),
                    replace(data, weights=data.weights[:-1]), replace(data, weights=-data.weights),
                    replace(data, times=data.times+.1), replace(data, times=data.times.flip(0)),
                    replace(data, times=data.times.new_tensor([0., .4, model.horizon])),
                    replace(data, weights=data.weights*2)])
        with patch('h_model.torch.optim.Adam') as optimizer:
            for invalid in bad:
                with self.assertRaises(ValueError):
                    train_h(model, invalid, epochs=1)
            optimizer.assert_not_called()

    def test_invalid_training_settings_and_unfrozen_reference(self):
        model, data = self.fixture()
        for kwargs in ({'epochs': 0}, {'epochs': True}, {'batch_size': 0}, {'batch_size': 1.5},
                       {'lr': 0.}, {'lr': float('nan')}, {'lr': float('inf')}):
            with self.assertRaises(ValueError):
                train_h(model, data, **kwargs)
        parameter = next(model.conditioning.reference.parameters())
        for unfreeze in (False, True):
            parameter.requires_grad_(unfreeze)
            parameter.grad = None if unfreeze else torch.ones_like(parameter)
            with self.assertRaisesRegex(ValueError, 'remain frozen'):
                train_h(model, data, epochs=1)

    def test_nonfinite_loss_restores_mixed_modes_on_failure(self):
        model, data = self.fixture()
        model.eval()
        model.network[0].train()
        modes = [m.training for m in model.modules()]
        with torch.no_grad():
            model.network[-1].bias.fill_(float('nan'))
        with self.assertRaisesRegex(ValueError, 'Nonfinite H training loss'):
            train_h(model, data, epochs=1)
        self.assertEqual([m.training for m in model.modules()], modes)
