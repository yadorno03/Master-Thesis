"""Probability targets, scaling chain rule, and new training/checkpoint contract."""
from pathlib import Path
import tempfile
import unittest

import torch

from h_model import HNetwork, h_time_weights, train_h, diagnose_h_terminal
import test_h_network


class HLateTimeTests(unittest.TestCase):
    def fixture(self):
        original = test_h_network.HNetworkTests().fixture()
        model = HNetwork(original.reference, original.config, (12, 12), log_remaining_time=True)
        data = model.sample_trajectories(64, 8, generator=torch.Generator().manual_seed(113))
        return model, data

    def test_standardization_chain_rule_and_roundtrip(self):
        model, data = self.fixture()
        model.fit_normalization(data)
        for j in (0, 3, 7):
            inputs = model.network_inputs(data.times[j], data.features[:, j])
            torch.testing.assert_close(inputs[:, :3].mean(0), torch.zeros(3, dtype=inputs.dtype), atol=1e-12, rtol=0)
            torch.testing.assert_close(inputs[:, :3].std(0, unbiased=False), torch.ones(3, dtype=inputs.dtype))
        y = data.features[:4, -1].clone().requires_grad_()
        t = data.times[-1]
        g = torch.autograd.grad(model.log_prob(t, y).sum(), y)[0]
        manual_inputs = torch.cat(((y-model.normalization_mean[-1])/model.normalization_scale[-1],
                                  torch.full((4, 1), t/model.horizon, dtype=y.dtype),
                                  torch.full((4, 1), torch.log((model.horizon-t)/model.horizon), dtype=y.dtype)), 1)
        manual_g = torch.autograd.grad(torch.nn.functional.logsigmoid(model.network(manual_inputs)).sum(), y)[0]
        torch.testing.assert_close(g, manual_g)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'h.pt'
            model.save_checkpoint(path)
            restored, _ = HNetwork.load_checkpoint(path)
            torch.testing.assert_close(restored(t, y), model(t, y), rtol=0, atol=0)
            restored_g = torch.autograd.grad(restored.log_prob(t, y).sum(), y)[0]
            torch.testing.assert_close(restored_g, g, rtol=0, atol=0)

    def test_exact_target_matches_final_euler_monte_carlo_and_gradient(self):
        model, data = self.fixture()
        dt = model.horizon/8
        x = data.last_states[:3].clone().requires_grad_()
        p = model.exact_last_step_probability(x, dt)
        mean = x + dt*(.5*x+model.reference.score(x, model.reference.t0+dt))
        rng = torch.Generator().manual_seed(19)
        noise = torch.randn((100000, 3, model.reference.d), generator=rng, dtype=x.dtype)
        empirical = (((mean.detach()[None]+dt**.5*noise) @ model.w) <= model.C).double().mean(0)
        torch.testing.assert_close(p[:, 0], empirical, atol=.006, rtol=0)
        grad = torch.autograd.grad(model.exact_last_step_probability(x, dt, log=True).sum(), x)[0]
        eps = 1e-6
        step = torch.zeros_like(x); step[:, 0] = eps
        difference = (model.exact_last_step_probability(x+step, dt, log=True)
                      - model.exact_last_step_probability(x-step, dt, log=True))/(2*eps)
        torch.testing.assert_close(grad[:, 0], difference[:, 0], atol=1e-6, rtol=1e-5)

    def test_new_training_fits_only_training_statistics_and_freezes_reference(self):
        model, data = self.fixture()
        validation = model.sample_trajectories(32, 8, generator=torch.Generator().manual_seed(114))
        before = {k: v.clone() for k, v in model.reference.state_dict().items()}
        q = h_time_weights(model, data, .4, .1)
        self.assertAlmostEqual(float(q.sum()), 1.)
        self.assertGreater(float(q[-1]), float(data.weights[-1]))
        selection = {}
        history = train_h(model, data, epochs=3, batch_size=16, validation_data=validation,
            selection=selection, normalize_inputs=True, loss='bce', late_time_fraction=.4,
            last_step_fraction=.1, times_per_trajectory=4, use_last_step_targets=True,
            generator=torch.Generator().manual_seed(115))
        self.assertEqual(len(history), 3)
        self.assertEqual(selection['best_loss'], min(selection['validation_history']))
        torch.testing.assert_close(model.normalization_mean, data.features.mean(0), rtol=0, atol=0)
        for k, v in model.reference.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.reference.parameters()))
        diagnostic = diagnose_h_terminal(model, validation)
        self.assertGreaterEqual(diagnostic['last_probability_rmse'], 0)
        self.assertGreaterEqual(len(diagnostic['late_rows']), 2)

    def test_invalid_new_options_rejected(self):
        model, data = self.fixture()
        for kwargs in ({'loss': 'focal'}, {'late_time_fraction': 1.}, {'last_step_fraction': -.1},
                       {'times_per_trajectory': 0}):
            with self.assertRaises(ValueError):
                train_h(model, data, **kwargs)
