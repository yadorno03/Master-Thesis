import unittest
from unittest.mock import patch

import torch

from DFM import UnconditionalDFM, train_dfm


class DFMTests(unittest.TestCase):
    def test_forward_ou_identity_and_transition(self):
        model = UnconditionalDFM(3, 1)
        x = torch.tensor([[1., -2., 3.]], dtype=torch.float64)
        noise = torch.ones_like(x)
        torch.testing.assert_close(model.q_sample(x, torch.zeros(1, 1), noise), x)
        t = torch.tensor([[0.7]], dtype=torch.float64)
        expected = torch.exp(-t / 2) * x + torch.sqrt(1 - torch.exp(-t)) * noise
        torch.testing.assert_close(model.q_sample(x, t, noise), expected)

    def test_dsm_is_unweighted_mean_of_squared_euclidean_errors(self):
        model = UnconditionalDFM(3, 1, t0=0.1, T=0.9).double()
        x = torch.zeros(2, 3, dtype=torch.float64)
        draws = torch.tensor([[0.25], [0.75]], dtype=x.dtype)
        noise = torch.tensor([[1., 2., 3.], [2., 0., -1.]], dtype=x.dtype)
        with patch('DFM.torch.rand', return_value=draws), \
                patch('DFM.torch.randn_like', return_value=noise), \
                patch.object(model.factor_score_network, 'forward', return_value=x):
            actual = model.loss_fn(x)
        times = torch.tensor([[0.3], [0.7]], dtype=x.dtype)
        expected = (noise.square() / (1 - torch.exp(-times))).sum(1).mean()
        torch.testing.assert_close(actual, expected)

    def test_training_preserves_precision_partial_batches_and_projection(self):
        for dtype in (torch.float32, torch.float64):
            torch.manual_seed(19)
            losses = []
            original = UnconditionalDFM.loss_fn
            def record(model, batch):
                loss = original(model, batch)
                losses.append((len(batch), loss.item()))
                return loss
            with patch.object(UnconditionalDFM, 'loss_fn', record):
                model, history = train_dfm(torch.randn(5, 3), 2, epochs=1,
                                          batch_size=3, t0=0.1, hidden_dims=[4], dtype=dtype)
            self.assertEqual([size for size, _ in losses], [3, 2])
            self.assertAlmostEqual(history[0], sum(n * loss for n, loss in losses) / 5)
            self.assertTrue(all(p.dtype == dtype and torch.isfinite(p).all()
                                for p in model.parameters()))
            loadings = model.factor_score_network.V
            torch.testing.assert_close(loadings.T @ loadings, torch.eye(2, dtype=dtype))
