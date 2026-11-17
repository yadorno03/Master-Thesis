"""DFM (Regimes + MLP): exact mixture score, the one-regime case, precision, initialization and training."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from NeuralNets import (
    REGIME_PARAMETERS, GaussianMLPScoreNetwork, RegimeMLPScoreNetwork, make_score_network, score_network_kind,
)
from conditioning import ConditioningConfig
from h_model import HNetwork
from reference import FrozenDFMReference
from synthetic_benchmark.configuration import Config, Streams
from synthetic_benchmark.unconditional import fit_dfm


def random_network(d=30, k=4, regimes=(3, 2), seed=0):
    torch.manual_seed(seed)
    network = RegimeMLPScoreNetwork(d, k, 2.0, [16, 16], regimes=regimes).double()
    with torch.no_grad():
        network.prior_mean.normal_()
        network.prior_chol.copy_(torch.randn(k, k, dtype=torch.float64).tril() + 2 * torch.eye(k))
        network.regime_logits.normal_()
    return network


def sample_regimes(n, V, c, S, factor_scales, residual_scales, table, seed):
    """Rows of the scale mixture X = V F + eps with the given scales and probability table."""
    rng = np.random.default_rng(seed)
    j, l = np.unravel_index(rng.choice(table.size, size=n, p=table.flatten()), table.shape)
    factors = factor_scales[j, None] * (rng.normal(size=(n, len(S))) @ np.linalg.cholesky(S).T)
    return factors @ V.T + residual_scales[l, None] * np.sqrt(c) * rng.normal(size=(n, len(c)))


class RegimeScoreNetworkTests(unittest.TestCase):
    times = torch.tensor([[.005], [.1], [.7], [2.], [8.]], dtype=torch.float64)

    def test_score_is_the_gradient_of_the_mixture_log_density(self):
        network = random_network()
        x = torch.randn(5, 30, dtype=torch.float64, requires_grad=True)
        gradient, = torch.autograd.grad(network.log_density(x, self.times).sum(), x)
        torch.testing.assert_close(network(x, self.times), gradient, rtol=1e-9, atol=1e-10)
        # The density is that of the dense mixture, and the scales have mean square 1 under the table.
        a, b, pi = network.regime_scales()
        torch.testing.assert_close(pi.sum(1) @ a ** 2, torch.tensor(1., dtype=torch.float64))
        torch.testing.assert_close(pi.sum(0) @ b ** 2, torch.tensor(1., dtype=torch.float64))
        c = network.c_raw.sigmoid() * network.sigma_max
        L = network.prior_chol.tril()
        for row, time in zip(x, self.times):
            alpha, h = torch.exp(-time / 2), -torch.expm1(-time)
            terms = [pi[j, l].log() + torch.distributions.MultivariateNormal(
                alpha * network.V @ network.prior_mean,
                alpha ** 2 * a[j] ** 2 * network.V @ L @ L.T @ network.V.T + torch.diag(h + alpha ** 2 * b[l] ** 2 * c)
            ).log_prob(row) for j in range(3) for l in range(2)]
            torch.testing.assert_close(network.log_density(row[None], time[None])[0],
                                       torch.logsumexp(torch.stack(terms), 0), rtol=1e-10, atol=1e-9)

    def test_one_regime_is_gaussian_mlp(self):
        network = random_network(regimes=(1, 1))
        torch.manual_seed(1)
        gaussian = GaussianMLPScoreNetwork(30, 4, 2.0, [16, 16]).double()
        with torch.no_grad():
            for layer in (network.g_zeta[0], network.g_zeta[2], network.g_zeta[4]):
                layer.weight.normal_()
            state = {name: value.clone() for name, value in network.state_dict().items()
                     if name not in REGIME_PARAMETERS}
            # The level input, between the projection and the time, is zero with a single residual scale.
            weight = state['g_zeta.0.weight']
            state['g_zeta.0.weight'] = torch.cat((weight[:, :4], weight[:, 5:]), 1)
            gaussian.load_state_dict(state)
        x = torch.randn(5, 30, dtype=torch.float64)
        torch.testing.assert_close(network(x, self.times), gaussian(x, self.times), rtol=1e-9, atol=1e-10)

    def test_float32_score_is_stable_at_market_dimensions(self):
        network = random_network(d=500, k=8, regimes=(4, 3))
        with torch.no_grad():
            network.prior_chol.mul_(3.)
        a, b, pi = (value.detach().numpy() for value in network.regime_scales())
        c = (network.c_raw.sigmoid() * network.sigma_max).detach().numpy()
        L = network.prior_chol.detach().tril().numpy()
        rows = sample_regimes(64, network.V.detach().numpy(), c, L @ L.T, a, b, pi, 3)
        x0 = torch.tensor(rows) + network.V.detach() @ network.prior_mean.detach()
        time = torch.exp(torch.linspace(np.log(.005), np.log(8.), 64, dtype=torch.float64))[:, None]
        x = torch.exp(-time / 2) * x0 + torch.sqrt(-torch.expm1(-time)) * torch.randn(64, 500, dtype=torch.float64)
        with torch.no_grad():
            exact = network(x, time)
            single = network.float()(x.float(), time.float())
        self.assertTrue(torch.isfinite(single).all())
        error = (single.double() - exact).norm(dim=1) / exact.norm(dim=1)
        self.assertLess(float(error.max()), 1e-3)

    def test_initialization_recovers_simulated_regimes_and_keeps_the_covariance(self):
        d, k = 40, 3
        rng = np.random.default_rng(0)
        V = np.linalg.qr(rng.normal(size=(d, k)))[0]
        c, S = rng.uniform(.3, .7, d), np.diag([30., 12., 5.])
        factor_scales, residual_scales = np.array([.6, 2.]), np.array([.7, 1.6])
        table = np.array([[.55, .1], [.05, .3]])
        factor_scales /= np.sqrt(table.sum(1) @ factor_scales ** 2)
        residual_scales /= np.sqrt(table.sum(0) @ residual_scales ** 2)
        x = torch.tensor(sample_regimes(6000, V, c, S, factor_scales, residual_scales, table, 1))
        network = RegimeMLPScoreNetwork(d, k, 2.0, [8], regimes=(2, 2)).double()
        with torch.no_grad():
            network.V.copy_(torch.tensor(V))
            network.c_raw.copy_(torch.logit(torch.tensor(c) / 2.0))
        network.initialize_prior(x)
        info = network.initialize_regimes(x)
        self.assertGreater(info['log_likelihood'], info['initial_log_likelihood'])
        a, b, pi = (value.detach().numpy() for value in network.regime_scales())
        np.testing.assert_allclose(a, factor_scales, rtol=.08)
        np.testing.assert_allclose(b, residual_scales, rtol=.03)
        np.testing.assert_allclose(pi, table, atol=.03)
        # The fitted covariance V S V' + diag(c) stays within sampling error of the sample covariance.
        self.assertLess(info['factor_covariance_error'], .05)
        self.assertLess(info['variance_error'], .05)
        L = network.prior_chol.detach().tril().numpy()
        fitted = V @ L @ L.T @ V.T + np.diag((network.c_raw.sigmoid() * 2.0).detach().numpy())
        sample = np.cov(x.numpy(), rowvar=False)
        self.assertLess(np.linalg.norm(fitted - sample) / np.linalg.norm(sample), .08)

    def test_registry_checkpoint_and_training(self):
        self.assertEqual(score_network_kind(make_score_network('regime_mlp', 6, 2, regimes=(2, 2))), 'regime_mlp')
        self.assertEqual(make_score_network('regime_mlp', 6, 2).regimes, (4, 3))
        with self.assertRaises(ValueError):
            RegimeMLPScoreNetwork(6, 2, regimes=(0, 2))
        cfg = replace(Config.preset('smoke'), d=10, k=3, n_train=256, n_val=64, dfm_epochs=4,
                      dfm_initialization='pca', dfm_standardization='per_asset', dfm_time_sampling='log_balanced',
                      dfm_network='regime_mlp', dfm_regimes=(2, 2), dfm_loading_warmup_epochs=2,
                      dfm_loading_lr_scale=.1, dfm_validation_draws=3, dfm_ema_decay=.9)
        rng = np.random.default_rng(4)
        rows = (rng.standard_t(4, size=(320, 1)) * rng.uniform(.5, 1.5, 10) + rng.standard_t(5, size=(320, 10)))
        model, history, selected = fit_dfm(rows[:256], rows[256:], cfg, Streams(3))
        score = model.factor_score_network
        self.assertIs(type(score), RegimeMLPScoreNetwork)
        self.assertEqual((score.regimes, len(history), set(history[0])),
                         ((2, 2), 4, {'epoch', 'train', 'validation', 'validation_factor'}))
        self.assertTrue(all(0 < row['validation_factor'] < row['validation'] for row in history))
        self.assertEqual(history[selected - 1]['validation'], min(row['validation'] for row in history))
        # The averaged loadings that are returned are orthonormal.
        torch.testing.assert_close(score.V.T @ score.V, torch.eye(3), atol=1e-5, rtol=0)
        # With the analytic part frozen, the regimes keep their maximum-likelihood values; trained ones move
        # after the warm-up.
        frozen = fit_dfm(rows[:256], rows[256:], replace(cfg, dfm_train_analytic=False, dfm_ema_decay=0.),
                         Streams(3))[0].factor_score_network
        trained = fit_dfm(rows[:256], rows[256:], replace(cfg, dfm_ema_decay=0.), Streams(3))[0].factor_score_network
        warm = fit_dfm(rows[:256], rows[256:], replace(cfg, dfm_epochs=3, dfm_train_analytic=False,
                                                       dfm_ema_decay=0.), Streams(3))[0].factor_score_network
        torch.testing.assert_close(frozen.log_factor_scale, warm.log_factor_scale)
        self.assertFalse(torch.equal(trained.log_factor_scale, frozen.log_factor_scale))
        # The H checkpoint restores the regime network with its numbers of scales.
        reference = FrozenDFMReference(model)
        h = HNetwork(reference, ConditioningConfig(w=[.1] * 10, C=-.5), hidden_dims=(8,))
        with tempfile.TemporaryDirectory() as directory:
            h.save_checkpoint(Path(directory) / 'h.pt')
            restored, _ = HNetwork.load_checkpoint(Path(directory) / 'h.pt')
        self.assertEqual(restored.reference._score.regimes, (2, 2))
        x = torch.randn(4, 10)
        torch.testing.assert_close(restored.reference.score(x, .3), reference.score(x, .3))


if __name__ == '__main__':
    unittest.main()
