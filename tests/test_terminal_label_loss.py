import unittest

import torch

from h_model import TerminalLabelData, terminal_label_loss


class TerminalLabelLossTests(unittest.TestCase):
    def test_nonuniform_time_integral_and_gradient(self):
        for dtype in (torch.float32, torch.float64):
            probabilities = torch.tensor([[[0.25], [0.5]], [[0.75], [0.25]]],
                                         dtype=dtype, requires_grad=True)
            data = TerminalLabelData(
                times=torch.tensor([0., 0.25], dtype=dtype),
                features=torch.zeros(2, 2, 3, dtype=dtype),
                labels=torch.tensor([[0.], [1.]], dtype=dtype),
                weights=torch.tensor([0.25, 0.75], dtype=dtype))
            loss = terminal_label_loss(probabilities, data.labels, data.weights)
            # Each path integrates its squared error; then the two paths average.
            expected = ((0.25**2/4 + 0.5**2*3/4)
                        + (0.25**2/4 + 0.75**2*3/4))/2
            self.assertEqual(loss.dtype, dtype)
            self.assertEqual(loss.ndim, 0)
            self.assertEqual(loss.item(), expected)
            loss.backward()
            gradient = torch.tensor([[[0.0625], [0.375]], [[-0.0625], [-0.5625]]],
                                    dtype=dtype)
            torch.testing.assert_close(probabilities.grad, gradient, rtol=0, atol=0)

    def test_time_refinement_and_trajectory_duplication_preserve_loss(self):
        probabilities = torch.tensor([[[0.2], [0.8]], [[0.6], [0.3]]], dtype=torch.float64)
        labels = torch.tensor([[0.], [1.]], dtype=probabilities.dtype)
        weights = torch.tensor([0.3, 0.7], dtype=probabilities.dtype)
        expected = terminal_label_loss(probabilities, labels, weights)
        refined = probabilities.repeat_interleave(2, dim=1)
        refined_weights = weights.repeat_interleave(2)/2
        torch.testing.assert_close(terminal_label_loss(refined, labels, refined_weights), expected)
        repeated = terminal_label_loss(probabilities.repeat(3, 1, 1), labels.repeat(3, 1), weights)
        torch.testing.assert_close(repeated, expected)

    def test_partial_batches_aggregate_by_trajectory_count(self):
        probabilities = torch.tensor([[[0.1], [0.2]], [[0.8], [0.9]],
                                      [[0.4], [0.6]]], dtype=torch.float64)
        labels = torch.tensor([[0.], [1.], [1.]], dtype=probabilities.dtype)
        weights = torch.tensor([0.25, 0.75], dtype=probabilities.dtype)
        full = terminal_label_loss(probabilities, labels, weights)
        first = terminal_label_loss(probabilities[:2], labels[:2], weights)
        last = terminal_label_loss(probabilities[2:], labels[2:], weights)
        torch.testing.assert_close((2*first + last)/3, full)
        self.assertGreater(abs(float((first + last)/2 - full)), 0.01)

    def test_single_time_perfect_predictions_and_constant_baseline(self):
        labels = torch.tensor([[0.], [0.], [0.], [1.]], dtype=torch.float64)
        weights = torch.ones(1, dtype=labels.dtype)
        perfect = labels[:, None].clone().requires_grad_()
        loss = terminal_label_loss(perfect, labels, weights)
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        torch.testing.assert_close(perfect.grad, torch.zeros_like(perfect))
        constant = torch.full_like(perfect, 0.25)
        self.assertEqual(terminal_label_loss(constant, labels, weights).item(), 0.1875)

    def test_shape_mismatches_fail_instead_of_broadcasting(self):
        p, labels, weights = torch.zeros(2, 3, 1), torch.zeros(2, 1), torch.ones(3)/3
        invalid = ((p.squeeze(-1), labels, weights), (p[..., 0, 0], labels, weights),
                   (p.expand(-1, -1, 2), labels, weights), (p, labels[:, 0], weights),
                   (p, labels[:1], weights), (p, labels[:, None], weights),
                   (p, labels, weights[:2]), (p, labels, weights[:, None]),
                   (torch.tensor(0.), labels, weights))
        for probabilities, targets, time_weights in invalid:
            with self.subTest(shapes=(probabilities.shape, targets.shape, time_weights.shape)):
                with self.assertRaisesRegex(ValueError, 'Expected probabilities'):
                    terminal_label_loss(probabilities, targets, time_weights)
