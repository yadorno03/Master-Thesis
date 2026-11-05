"""Observation-only initialization and explicit time-weighting contracts."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from DFM import UnconditionalDFM
from synthetic_benchmark.configuration import Config, Streams
from synthetic_benchmark.unconditional import (
    dfm_training_times, dfm_training_loss, initialize_dfm_from_observations, fit_dfm,
)


class SyntheticTrainingTests(unittest.TestCase):
    def test_importance_weights_recover_uniform_integral(self):
        cfg = Config(dfm_time_sampling='log_importance')
        # Midpoint quadrature for u in (0,1), with t=t0*(T/t0)**u.
        u = (torch.arange(100000, dtype=torch.float64) + .5) / 100000
        t = cfg.t0 * torch.exp(np.log(cfg.T / cfg.t0) * u)
        weight = t * np.log(cfg.T / cfg.t0) / (cfg.T - cfg.t0)
        self.assertAlmostEqual(float(weight.mean()), 1., places=8)
        self.assertAlmostEqual(float((weight * t).mean()), (cfg.T + cfg.t0) / 2, places=7)

    def test_time_proposals_and_losses(self):
        cfg = Config(d=3, k=3, T=2., t0=.01, width=4)
        model = UnconditionalDFM(3, 3, T=2., t0=.01, hidden_dims=[4])
        x = torch.randn(20, 3); z = torch.randn(20, 3)
        for mode in ('uniform', 'log_importance', 'log_balanced'):
            config = replace(cfg, dfm_time_sampling=mode)
            times = dfm_training_times(20, config, torch.Generator().manual_seed(33))
            self.assertTrue(((times >= cfg.t0) & (times <= cfg.T)).all())
            repeated = dfm_training_times(20, config, torch.Generator().manual_seed(33))
            torch.testing.assert_close(times, repeated)
            h = -torch.expm1(-times)
            error = (model.factor_score_network(model.q_sample(x, times, z), times)
                     + z / h.sqrt()).square().sum(-1)
            if mode == 'log_importance':
                error *= times[:, 0] * np.log(cfg.T / cfg.t0) / (cfg.T - cfg.t0)
            elif mode == 'log_balanced':
                error *= h[:, 0]
            torch.testing.assert_close(dfm_training_loss(model, x, times, z, config), error.mean())

    def test_pca_initialization_uses_centered_observation_covariance(self):
        rng = np.random.default_rng(8)
        x = torch.tensor(rng.normal(size=(100, 8)) @ np.diag(np.arange(1, 9)), dtype=torch.float64)
        a = UnconditionalDFM(8, 3, sigma_max=2., hidden_dims=[4]).double()
        b = UnconditionalDFM(8, 3, sigma_max=2., hidden_dims=[4]).double()
        initialize_dfm_from_observations(a, x)
        initialize_dfm_from_observations(b, x + 17.)
        V = a.factor_score_network.V
        torch.testing.assert_close(V.T @ V, torch.eye(3, dtype=torch.float64))
        # Sign-invariant comparison; centering is only for initialization.
        torch.testing.assert_close(V @ V.T, b.factor_score_network.V @ b.factor_score_network.V.T)
        torch.testing.assert_close(a.factor_score_network.c_raw, b.factor_score_network.c_raw)

    def test_short_tuned_fit_selects_validation_checkpoint_and_unfreezes_loadings(self):
        cfg = Config(d=8, k=3, width=4, batch=16, dfm_epochs=3,
                     dfm_initialization='pca', dfm_time_sampling='log_balanced',
                     dfm_loading_lr_scale=.1, dfm_loading_warmup_epochs=100)
        x = np.random.default_rng(12).normal(size=(32, 8))
        val = np.random.default_rng(13).normal(size=(20, 8))
        loading_updates = []
        original_step = torch.optim.Adam.step
        def record_step(optimizer, *args, **kwargs):
            loading = optimizer.param_groups[0]['params'][0]
            loading_updates.append(loading.requires_grad and loading.grad is not None)
            return original_step(optimizer, *args, **kwargs)
        with patch.object(torch.optim.Adam, 'step', record_step):
            model, history, selected = fit_dfm(x, val, cfg, Streams(3))
        self.assertEqual(loading_updates, [False, False, False, False, True, True])
        self.assertEqual(selected, 1 + np.argmin([h['validation'] for h in history]))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
        torch.testing.assert_close(model.factor_score_network.V.T @ model.factor_score_network.V,
                                   torch.eye(3), atol=1e-6, rtol=1e-6)

    def test_invalid_settings(self):
        for kwargs in ({'dfm_time_sampling': 'unknown'}, {'dfm_initialization': 'truth'},
                       {'dfm_loading_lr_scale': 0}, {'dfm_loading_warmup_epochs': -1}):
            with self.assertRaises(ValueError):
                Config(**kwargs)


if __name__ == '__main__':
    unittest.main()
