import unittest

import torch

from h_model import HNetwork, terminal_label_loss
import test_conditioning


class HNetworkTests(unittest.TestCase):
    def fixture(self, dtype=torch.float64):
        with torch.random.fork_rng():
            torch.manual_seed(27)
            conditioning = test_conditioning.ConditioningTests().fixture(dtype)
            model = HNetwork(conditioning, hidden_dims=(1,))
        with torch.no_grad():
            model.network[0].weight.copy_(torch.tensor([[.2, -.3, .7, .4]], dtype=dtype))
            model.network[0].bias.fill_(.1)
            model.network[2].weight.fill_(1.3)
            model.network[2].bias.fill_(-.2)
        return model

    def test_probability_time_normalization_and_log_gradient(self):
        for dtype in (torch.float32, torch.float64):
            model = self.fixture(dtype)
            y = torch.tensor([[.2, -.1, .4], [-.3, .5, -.2]], dtype=dtype, requires_grad=True)
            t = torch.tensor([.2, .7], dtype=dtype, requires_grad=True)
            a = y @ y.new_tensor([.2, -.3, .7]) + .4*t/model.horizon + .1
            logits = 1.3*torch.tanh(a)-.2
            expected = torch.sigmoid(logits)[:, None]
            self.assertEqual(model(t, y).shape, (2, 1))
            torch.testing.assert_close(model(t, y), expected)
            torch.testing.assert_close(model(t[:, None], y), expected)
            torch.testing.assert_close(model(.2, y), model(torch.full_like(t, .2), y))
            torch.testing.assert_close(model.log_prob(t, y), expected.log())
            dy, dt = torch.autograd.grad(model.log_prob(t, y).sum(), (y, t))
            slope = (1-expected[:, 0])*1.3*(1-torch.tanh(a).square())
            torch.testing.assert_close(dy, slope[:, None]*y.new_tensor([.2, -.3, .7]))
            torch.testing.assert_close(dt, slope*.4/model.horizon)
            self.assertTrue(bool((dy[:, -1].abs() > 0).all()))  # full portfolio U participates

    def test_log_probability_survives_sigmoid_underflow(self):
        for dtype in (torch.float32, torch.float64):
            model = self.fixture(dtype)
            with torch.no_grad():
                model.network[-1].bias.fill_(-1000.)
            y = torch.zeros(2, 3, dtype=dtype, requires_grad=True)
            self.assertTrue(bool((model(.3, y) == 0).all()))
            log_probability = model.log_prob(.3, y)
            gradient, = torch.autograd.grad(log_probability.sum(), y)
            self.assertTrue(bool(torch.isfinite(log_probability).all()))
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_loss_updates_only_h_and_reference_stays_frozen(self):
        model = self.fixture()
        ref = model.conditioning.reference
        before = {name: value.clone() for name, value in ref.state_dict().items()}
        initial = model.network[-1].bias.detach().clone()
        model.train()
        self.assertFalse(ref.training)
        y = torch.zeros(2, 3, dtype=torch.float64)
        loss = terminal_label_loss(model(.3, y)[:, None], torch.ones(2, 1, dtype=y.dtype),
                                   torch.ones(1, dtype=y.dtype))
        optimizer = torch.optim.SGD(model.network.parameters(), lr=.1)
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(model.network[-1].bias, initial))
        for name, value in ref.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in ref.parameters()))

    def test_invalid_architecture_times_and_features(self):
        model = self.fixture()
        with self.assertRaises(TypeError):
            HNetwork(None)
        for widths in ((), (0,), (-1,), (True,), (2.5,)):
            with self.assertRaises(ValueError):
                HNetwork(model.conditioning, hidden_dims=widths)
        y = torch.zeros(2, 3, dtype=torch.float64)
        for t, features in [(-.1, y), (model.horizon+.1, y), ([.1, .2, .3], y),
                            (.1, y.float()), (.1, y[:, :2]), (.1, y[:0]), (.1, y*float('nan'))]:
            for method in (model, model.log_prob):
                with self.assertRaises(ValueError):
                    method(t, features)
