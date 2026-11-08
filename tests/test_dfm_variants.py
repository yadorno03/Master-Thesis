import math
import unittest

import numpy as np
import torch

from dfm_variants import (
    FactorPreconditioner, GaussianFactorScore, GaussianMLPScoreNetwork,
    StandardizedInputScoreNetwork, factor_selection_score, fit_gaussian_factor_model,
    sample_gaussian_factor_model, selection_reference,
)
from synthetic_conditional_truth import FactorPopulation, make_population


def population_params(pop):
    return dict(mean=pop.mu[0], V=pop.B, S=pop.S[0], c=pop.D)


class GaussianScoreTest(unittest.TestCase):
    def setUp(self):
        self.pop = make_population(30, 'gaussian', k=3)
        self.rng = np.random.default_rng(0)

    def test_exact_score_matches_autograd_density(self):
        params = population_params(self.pop)
        score = GaussianFactorScore(params)
        cov = self.pop.covs[0]
        x = torch.tensor(self.rng.normal(size=(5, 30)), dtype=torch.float64, requires_grad=True)
        t = torch.tensor([[0.01], [0.1], [0.5], [2.0], [7.0]], dtype=torch.float64)
        for i in range(5):
            alpha, h = math.exp(-t[i].item() / 2), -math.expm1(-t[i].item())
            dist = torch.distributions.MultivariateNormal(
                torch.tensor(alpha * params['mean']), torch.tensor(alpha ** 2 * cov + h * np.eye(30)))
            grad, = torch.autograd.grad(dist.log_prob(x[i]), x)
            np.testing.assert_allclose(score(x[i:i + 1].detach(), t[i:i + 1]).numpy()[0], grad[i].numpy(),
                                       rtol=1e-8, atol=1e-8)

    def test_gaussian_mlp_network_equals_exact_score_at_true_parameters(self):
        pop = self.pop
        net = GaussianMLPScoreNetwork(30, 3, sigma_max=2.0, hidden_dims=(8, 8)).double()
        with torch.no_grad():
            net.V.copy_(torch.tensor(pop.B))
            net.c_raw.copy_(torch.logit(torch.tensor(pop.D / 2.0)))
            net.prior_mean.copy_(torch.tensor(pop.m[0]))
            net.prior_chol.copy_(torch.linalg.cholesky(torch.tensor(pop.S[0])))
        exact = GaussianFactorScore(population_params(pop))
        x = torch.tensor(pop.sample(6, self.rng), dtype=torch.float64)
        t = torch.tensor([[0.005], [0.05], [0.3], [1.0], [3.0], [8.0]], dtype=torch.float64)
        np.testing.assert_allclose(net(x, t).detach().numpy(), exact(x, t).numpy(), rtol=1e-7, atol=1e-7)

    def test_score_network_registry(self):
        from NeuralNets import FactorScoreNetwork, make_score_network, score_network_kind
        for kind, cls in (('mlp', FactorScoreNetwork), ('gaussian_mlp', GaussianMLPScoreNetwork)):
            network = make_score_network(kind, 10, 3, 2.0, (4, 4))
            self.assertIs(type(network), cls)
            self.assertEqual(score_network_kind(network), kind)
        with self.assertRaisesRegex(ValueError, 'Unknown score network'):
            make_score_network('unet', 10, 3)

        class Subclass(FactorScoreNetwork):
            pass
        with self.assertRaisesRegex(ValueError, 'Unregistered'):
            score_network_kind(Subclass(10, 3))

    def test_gaussian_mlp_reference_survives_an_h_checkpoint(self):
        import tempfile
        from pathlib import Path
        from DFM import UnconditionalDFM
        from conditioning import ConditioningConfig
        from h_model import HNetwork
        from reference import FrozenDFMReference
        torch.manual_seed(0)
        dfm = UnconditionalDFM(30, 3, sigma_max=2.0, T=2.0, t0=.01, hidden_dims=[8, 8],
                               score_network='gaussian_mlp').double()
        dfm.factor_score_network.initialize_prior(torch.tensor(self.pop.sample(200, self.rng)))
        with torch.no_grad():
            dfm.factor_score_network.g_zeta[-1].weight.normal_()  # A nonzero MLP correction.
        reference = FrozenDFMReference(dfm)
        h = HNetwork(reference, ConditioningConfig(self.pop.w, self.pop.threshold(.05)), hidden_dims=(8,))
        with tempfile.TemporaryDirectory() as tmp:
            h.save_checkpoint(Path(tmp) / 'h.pt')
            restored, _ = HNetwork.load_checkpoint(Path(tmp) / 'h.pt')
        self.assertIs(type(restored.reference._score), GaussianMLPScoreNetwork)
        x = torch.tensor(self.pop.sample(5, self.rng))
        torch.testing.assert_close(restored.reference.score(x, .5), reference.score(x, .5), rtol=0, atol=0)

    def test_fitted_gaussian_recovers_population_with_identifiable_factors(self):
        # Factors well above every idiosyncratic variance, so PCA identifies them.
        B = np.linalg.qr(self.rng.normal(size=(30, 3)))[0]
        pop = FactorPopulation(B, self.rng.uniform(0.05, 0.15, 30), np.ones(1), np.zeros((1, 3)),
                               np.diag([4.0, 2.0, 1.0])[None])
        x = pop.sample(20000, self.rng)
        params = fit_gaussian_factor_model(x, 3)
        model_cov = params['V'] @ params['S'] @ params['V'].T + np.diag(params['c'])
        cov = pop.covs[0]
        noise = np.linalg.norm(np.cov(pop.sample(20000, self.rng), rowvar=False) - cov) / np.linalg.norm(cov)
        self.assertLess(np.linalg.norm(model_cov - cov) / np.linalg.norm(cov), 2 * noise)
        draws = sample_gaussian_factor_model(params, 20000, self.rng)
        self.assertLess(np.linalg.norm(np.cov(draws, rowvar=False) - cov) / np.linalg.norm(cov), 3 * noise)


class PreconditionerTest(unittest.TestCase):
    def test_round_trip_event_and_target_ratio(self):
        pop = make_population(40, 'gaussian', k=4)
        rng = np.random.default_rng(1)
        x = pop.sample(5000, rng)
        pre = FactorPreconditioner.fit(x, 4, rho=5.0, mode='rho')
        self.assertGreater(pre.info['shrunk'], 0)
        y = pre.forward(x)
        np.testing.assert_allclose(pre.inverse(y), x, atol=1e-10)
        w = np.ones(40) / 40
        np.testing.assert_allclose(x @ w, y @ pre.transformed_weights(w), atol=1e-10)
        top = np.var(y @ pre.V[:, 0], ddof=1)
        self.assertAlmostEqual(top / (5.0 * pre.info['c_bar']), 1.0, places=6)
        self.assertEqual(FactorPreconditioner.fit(x, 4, rho=5.0, mode='pc1').info['shrunk'], 1)


class StandardizedInputTest(unittest.TestCase):
    def test_statistics_standardize_projected_inputs(self):
        pop = make_population(30, 'gaussian', k=3)
        x = torch.tensor(pop.sample(4000, np.random.default_rng(2)), dtype=torch.float32)
        net = StandardizedInputScoreNetwork(30, 3, sigma_max=2.0, hidden_dims=(8, 8), T=8.0, t0=0.005)
        net.refresh_input_statistics(x, torch.Generator().manual_seed(0), rows=4000)
        t = torch.full((4000, 1), float(net.log_grid[10].exp()))
        alpha, h = torch.exp(-t / 2), -torch.expm1(-t)
        xt = alpha * x + torch.sqrt(h) * torch.randn(x.shape, generator=torch.Generator().manual_seed(1))
        u = net._projected(xt, t)[1] @ net.V
        z = (u - net._interpolate(net.input_mean, t)) / net._interpolate(net.input_std, t)
        np.testing.assert_allclose(z.mean(0).detach().numpy(), 0, atol=0.08)
        np.testing.assert_allclose(z.std(0).detach().numpy(), 1, atol=0.08)
        self.assertTrue(torch.isfinite(net(xt[:10], t[:10])).all())


class SelectionScoreTest(unittest.TestCase):
    def test_score_detects_shrunken_factors(self):
        pop = make_population(40, 'gaussian', k=4)
        rng = np.random.default_rng(3)
        reference = selection_reference(pop.sample(5000, rng), 4)
        good = pop.sample(3000, rng)
        mean = good.mean(0)
        coordinates = (good - mean) @ pop.B
        shrunk = good - 0.5 * coordinates @ pop.B.T
        self.assertLess(factor_selection_score(good, reference)[0], 0.15)
        self.assertGreater(factor_selection_score(shrunk, reference)[0], 0.8)


if __name__ == '__main__':
    unittest.main()
