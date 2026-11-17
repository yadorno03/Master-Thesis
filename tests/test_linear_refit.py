"""Linear refit of g (refit_linear_part), DFMs that train only their MLP, and their checkpoints."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import math
import tempfile
import unittest

import numpy as np
import torch

from DFM import UnconditionalDFM
from NeuralNets import make_score_network
from conditioning import ConditioningConfig
from h_model import HNetwork
from reference import FrozenDFMReference
from synthetic_benchmark.configuration import Config, Streams
from synthetic_benchmark.unconditional import fit_dfm, refit_linear_part


def heavy_rows(n=320, d=10, seed=4):
    rng = np.random.default_rng(seed)
    return rng.standard_t(4, size=(n, 1)) * rng.uniform(.5, 1.5, d) + rng.standard_t(5, size=(n, d))


class LinearRefitTests(unittest.TestCase):
    def test_unfitted_refit_changes_no_score_and_older_states_load(self):
        x = torch.randn(5, 6, dtype=torch.float64)
        t = torch.tensor([[.005], [.05], [.5], [3.], [8.]], dtype=torch.float64)
        for kind in ('mlp', 'gaussian_mlp', 'regime_mlp'):
            plain = make_score_network(kind, 6, 2, 2.0, [8], regimes=(2, 2)).double()
            refit = make_score_network(kind, 6, 2, 2.0, [8], regimes=(2, 2), refit_knots=4).double()
            # A state saved without a refit lacks only the refit buffers.
            missing = refit.load_state_dict(plain.state_dict(), strict=False)
            self.assertEqual(set(missing.missing_keys), {'refit_log_times', 'refit_gamma'})
            self.assertFalse(hasattr(plain, 'refit_gamma'))
            torch.testing.assert_close(refit(x, t), plain(x, t))
        for knots in (1, -1):
            with self.assertRaises(ValueError):
                make_score_network('mlp', 6, 2, refit_knots=knots)

    def test_refit_term_is_piecewise_linear_in_log_time(self):
        network = make_score_network('mlp', 6, 2, refit_knots=3).double()
        network.refit_log_times.copy_(torch.tensor([-2., 0., 2.]))
        t = torch.exp(torch.tensor([[-3.], [-2.], [-1.], [0.], [1.5], [5.]], dtype=torch.float64))
        weights = torch.tensor([[1, 0, 0], [1, 0, 0], [.5, .5, 0], [0, 1, 0], [0, .25, .75], [0, 0, 1]],
                               dtype=torch.float64)
        torch.testing.assert_close(network.refit_hats(t), weights)
        network.refit_gamma.copy_(torch.stack([value * torch.eye(2) for value in (1., 2., 4.)]))
        u = torch.tensor([[1., -2.]], dtype=torch.float64).expand(6, 2)
        torch.testing.assert_close(network.linear_refit(u, t),
                                   (weights @ torch.tensor([1., 2., 4.], dtype=torch.float64))[:, None] * u)

    def test_refit_recovers_the_factor_scale_of_a_gaussian_factor_model(self):
        # Rows of N(0, V L V^T + diag(c)); the Gaussian part is given half the factor covariance.
        d, k = 6, 2
        generator = torch.Generator().manual_seed(0)
        V = torch.linalg.qr(torch.randn(d, k, generator=generator, dtype=torch.float64))[0]
        L, c = torch.tensor([4., 1.5], dtype=torch.float64), torch.full((d,), .3, dtype=torch.float64)
        covariance = V @ torch.diag(L) @ V.T + torch.diag(c)
        rows = torch.randn(4000, d, generator=generator, dtype=torch.float64) @ torch.linalg.cholesky(covariance).T
        model = UnconditionalDFM(d, k, sigma_max=2.0, T=8., t0=.005, hidden_dims=[8], score_network='gaussian_mlp',
                                 refit_knots=8).double()
        score = model.factor_score_network
        with torch.no_grad():
            score.V.copy_(V)
            score.c_raw.copy_(torch.logit(c / 2.0))
            score.prior_chol.copy_(torch.diag((L / 2).sqrt()))

        def error():
            t = torch.tensor([.01, .1, 1.], dtype=torch.float64).repeat_interleave(2000)[:, None]
            alpha, h = torch.exp(-t / 2), -torch.expm1(-t)
            x = alpha * rows[:2000].repeat(3, 1) + h.sqrt() * torch.randn(6000, d, generator=generator,
                                                                         dtype=torch.float64)
            exact = torch.stack([-torch.linalg.solve(a * a * covariance + (1 - a * a) * torch.eye(d, dtype=x.dtype), y)
                                 for a, y in zip(alpha[:, 0], x)])
            with torch.no_grad():
                return float((score(x, t) - exact).square().sum(-1).mean())

        before = error()
        cfg = SimpleNamespace(t0=.005, T=8., dfm_time_sampling='log_balanced', device='cpu')
        fitted = refit_linear_part(model, rows, cfg, torch.Generator().manual_seed(1), draws=8)
        torch.testing.assert_close(score.refit_log_times, torch.linspace(math.log(.005), math.log(8.), 8,
                                                                         dtype=torch.float64))
        self.assertLess(fitted['objective_after'], fitted['objective_before'])
        self.assertLess(error(), .1 * before)

    def test_mlp_only_training_refit_and_h_checkpoint(self):
        rows = heavy_rows()
        cfg = replace(Config.preset('smoke'), d=10, k=3, n_train=256, n_val=64, dfm_epochs=3,
                      dfm_initialization='pca', dfm_standardization='per_asset', dfm_time_sampling='log_balanced',
                      dfm_regimes=(2, 2), dfm_loading_warmup_epochs=1, dfm_loading_lr_scale=.1,
                      dfm_train_analytic=False, dfm_refit_knots=4)
        for network in ('mlp', 'regime_mlp'):
            short = fit_dfm(rows[:256], rows[256:], replace(cfg, dfm_network=network, dfm_epochs=1),
                            Streams(3))[0].factor_score_network
            model = fit_dfm(rows[:256], rows[256:], replace(cfg, dfm_network=network), Streams(3))[0]
            score = model.factor_score_network
            # Only the MLP trains: every other parameter keeps its initial value.
            initial = dict(short.named_parameters())
            for name, parameter in score.named_parameters():
                if not name.startswith('g_zeta.'):
                    torch.testing.assert_close(parameter, initial[name], msg=name)
            self.assertFalse(torch.equal(score.g_zeta[0].weight, initial['g_zeta.0.weight']))
            # The refit is fitted on [log t0, log T].
            torch.testing.assert_close(score.refit_log_times, torch.linspace(math.log(cfg.t0), math.log(cfg.T), 4))
            self.assertTrue(torch.isfinite(score.refit_gamma).all() and score.refit_gamma.abs().sum() > 0)
        # The H checkpoint restores the refit.
        reference = FrozenDFMReference(model)
        h = HNetwork(reference, ConditioningConfig(w=[.1] * 10, C=-.5), hidden_dims=(8,))
        with tempfile.TemporaryDirectory() as directory:
            h.save_checkpoint(Path(directory) / 'h.pt')
            restored, _ = HNetwork.load_checkpoint(Path(directory) / 'h.pt')
        self.assertEqual(restored.reference._score.refit_knots, 4)
        x = torch.randn(4, 10)
        torch.testing.assert_close(restored.reference.score(x, .3), reference.score(x, .3))

    def test_configuration_checks(self):
        for knots in (1, -2, True, 2.5):
            with self.assertRaises(ValueError):
                replace(Config.preset('smoke'), dfm_refit_knots=knots)
        with self.assertRaises(ValueError):
            replace(Config.preset('smoke'), dfm_train_analytic='no')


if __name__ == '__main__':
    unittest.main()
