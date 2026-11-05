from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from h_model import TerminalLabelData, evaluate_h, train_h
import test_h_network


class HEvaluationTests(unittest.TestCase):
    def fixture(self, dtype=torch.float64):
        model = test_h_network.HNetworkTests().fixture(dtype)
        data = TerminalLabelData(
            torch.tensor([0., .1, .8], dtype=dtype)*model.horizon,
            torch.arange(45, dtype=dtype).reshape(5, 3, 3)/20-1,
            torch.tensor([[0.], [0.], [0.], [0.], [1.]], dtype=dtype),
            torch.tensor([.1, .7, .2], dtype=dtype))
        return model, data

    def test_analytic_nonuniform_losses_and_batch_size_invariance(self):
        for dtype in (torch.float32, torch.float64):
            model, data = self.fixture(dtype)
            expected = 0.
            for i in range(5):
                for j, weight in enumerate((.1, .7, .2)):
                    y = data.features[i, j]
                    a = .2*y[0]-.3*y[1]+.7*y[2]+.4*data.times[j]/model.horizon+.1
                    probability = torch.sigmoid(1.3*torch.tanh(a)-.2).item()
                    expected += weight*(probability-float(data.labels[i, 0]))**2/5
            for batch_size in (1, 2, 5, 8):
                metrics = evaluate_h(model, data, constant_probability=.75, batch_size=batch_size)
                self.assertEqual(set(metrics), {'h_loss', 'constant_loss'})
                self.assertAlmostEqual(metrics['h_loss'], expected, places=6)
                self.assertAlmostEqual(metrics['constant_loss'], (4*.75**2+.25**2)/5, places=6)
                self.assertTrue(all(isinstance(value, float) for value in metrics.values()))

    def test_supplied_training_baseline_is_not_refitted_on_heldout_labels(self):
        model, data = self.fixture()
        training_labels = torch.tensor([1., 1., 1., 0.])
        baseline = float(training_labels.mean())  # .75, unlike held-out mean .2
        metrics = evaluate_h(model, data, constant_probability=baseline, batch_size=2)
        self.assertAlmostEqual(metrics['constant_loss'], .4625)
        self.assertNotAlmostEqual(metrics['constant_loss'], .2*.8)
        for probability, expected in ((0., .2), (1., .8)):
            metrics = evaluate_h(model, data, constant_probability=probability)
            self.assertAlmostEqual(metrics['constant_loss'], expected)

    def test_evaluation_disables_autograd_without_mutating_model_data_or_rng(self):
        model, data = self.fixture()
        model.train()
        model.network[0].eval()
        for parameter in model.network.parameters():
            parameter.grad = torch.full_like(parameter, .125)
        modes = [module.training for module in model.modules()]
        state = {name: value.clone() for name, value in model.state_dict().items()}
        gradients = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
        tensors = {name: value.clone() if value is not None else None for name, value in vars(data).items()}
        rng = torch.random.get_rng_state().clone()
        seen = []
        handle = model.register_forward_hook(lambda module, inputs, output:
            seen.append((torch.is_grad_enabled(), output.requires_grad, output.grad_fn)))
        try:
            with torch.enable_grad():
                evaluate_h(model, data, constant_probability=.75, batch_size=2)
                self.assertTrue(torch.is_grad_enabled())
        finally:
            handle.remove()
        self.assertEqual(seen, [(False, False, None)]*3)
        self.assertEqual([module.training for module in model.modules()], modes)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        for parameter, gradient in zip(model.parameters(), gradients):
            if gradient is None:
                self.assertIsNone(parameter.grad)
            else:
                torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
        for name, value in vars(data).items():
            torch.testing.assert_close(value, tensors[name], rtol=0, atol=0)

    def test_invalid_requests_fail_before_prediction(self):
        model, data = self.fixture()
        invalid_data = [replace(data, features=data.features[:0]),
                        replace(data, labels=data.labels+.5), replace(data, weights=-data.weights),
                        replace(data, features=data.features.clone().requires_grad_())]
        with patch.object(model, 'forward') as forward:
            for probability in (-.1, 1.1, float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    evaluate_h(model, data, constant_probability=probability)
            for batch_size in (0, -1, True, 1.5):
                with self.assertRaises(ValueError):
                    evaluate_h(model, data, constant_probability=.5, batch_size=batch_size)
            for invalid in invalid_data:
                with self.assertRaises(ValueError):
                    evaluate_h(model, invalid, constant_probability=.5)
            forward.assert_not_called()

    def test_train_then_evaluate_independent_unguided_paths(self):
        model, _ = self.fixture()
        training = model.sample_trajectories(7, 4, generator=torch.Generator().manual_seed(51))
        heldout = model.sample_trajectories(5, 4, generator=torch.Generator().manual_seed(83))
        baseline = float(training.labels.mean())
        train_h(model, training, epochs=2, batch_size=3, generator=torch.Generator().manual_seed(4))
        metrics = evaluate_h(model, heldout, constant_probability=baseline, batch_size=2)
        probabilities = torch.stack([model(t, heldout.features[:, j])
                                     for j, t in enumerate(heldout.times)], dim=1).detach()
        self.assertAlmostEqual(metrics['h_loss'], float((probabilities-heldout.labels[:, None]).square().mean()))
        self.assertAlmostEqual(metrics['constant_loss'], float((baseline-heldout.labels).square().mean()))
