import unittest
from unittest.mock import patch, PropertyMock
import numpy as np
import torch
from synthetic_conditional_truth import (make_population, FactorPopulation, population_metadata,
    population_from_checkpoint, population_details, check_population_model, check_population_h)
from synthetic_benchmark.configuration import Config, Streams
from scipy.stats import kstest
from DFM import UnconditionalDFM
from reference import FrozenDFMReference
from h_model import HNetwork
from conditioning import ConditioningConfig

def validate_truth(pop,n,seed):
    rng=np.random.default_rng(seed); rows=[]
    for tail in (None,.1,.05,.01):
        c=None if tail is None else pop.threshold(tail)
        x,labels=pop.sample(n,rng,c,True); mean,cov=pop.moments(c)
        se=np.sqrt(np.diag(cov)/n)
        row=dict(tail=tail,mean_max_z=np.max(np.abs(x.mean(0)-mean)/se),
                 covariance_relative=np.linalg.norm(np.cov(x,rowvar=False)-cov)/np.linalg.norm(cov),
                 population_ks=kstest(x@pop.w,lambda p:pop.cdf(p,c)).statistic)
        assert row['mean_max_z']<7
        covariance_mc_scale = np.sqrt((np.trace(cov)**2 + np.sum(cov**2)) / (n - 1)) / np.linalg.norm(cov)
        assert row['covariance_relative'] < 5 * covariance_mc_scale
        assert row['population_ks']<2.5/np.sqrt(n)
        target=pop.pi if c is None else pop.component_probabilities(c)
        row['component_max_error']=np.max(np.abs(np.bincount(labels,minlength=len(target))/n-target))
        assert row['component_max_error']<5/np.sqrt(n)
        if c is not None:
            assert np.max(x@pop.w-c)<1e-12
            assert abs(pop.cdf(c)-tail)<1e-10
        rows.append(row)
    return rows

class SyntheticConditionalTests(unittest.TestCase):
    def test_truth_moments_cdf_and_components(self):
        for case in ('gaussian',):
            pop=make_population(8,case,k=3)
            np.testing.assert_allclose(pop.B.T@pop.B,np.eye(3),atol=1e-14)
            self.assertGreater(pop.variance_contributions()['residual'],0)
            validate_truth(pop,20000,913)

    def test_xu_construction_and_exact_coordinate_change(self):
        for d, k in ((20, 3), (100, 8), (20, 12), (8, 8), (2048, 16)):
            pop = make_population(d, k=k)
            raw = pop.construction
            rng = np.random.default_rng(912)
            mu = rng.uniform(0., .1, k)
            beta = rng.normal(size=(d, k))
            sigma_eps = rng.uniform(0., .4, d)
            np.testing.assert_array_equal(raw['mu_F'], mu)
            np.testing.assert_array_equal(raw['beta'], beta)
            np.testing.assert_array_equal(pop.D, sigma_eps**2)
            np.testing.assert_array_equal(raw['Sigma_F'], np.diag((1.5 * mu)**2))
            A = raw['A']
            np.testing.assert_allclose(pop.B.T @ pop.B, np.eye(k), atol=2e-14)
            np.testing.assert_allclose(pop.B @ A, beta, atol=2e-14)
            self.assertTrue(np.all(np.diag(A) >= 0))
            np.testing.assert_array_equal(A, np.triu(A))
            np.testing.assert_allclose(pop.B @ pop.m[0], beta @ mu, atol=2e-14)
            np.testing.assert_allclose(pop.B @ pop.S[0] @ pop.B.T,
                                       beta @ raw['Sigma_F'] @ beta.T, atol=2e-14)
            factors = rng.multivariate_normal(mu, raw['Sigma_F'], size=64)
            eps = rng.normal(0., sigma_eps, size=(64, d))
            np.testing.assert_allclose(factors @ beta.T + eps,
                                       (factors @ A.T) @ pop.B.T + eps, atol=2e-14)
            self.assertTrue(np.all((pop.D > 0) & (pop.D < .16)))
            self.assertGreater(np.linalg.eigvalsh(pop.S[0]).min(), 0.)
            self.assertGreater(np.linalg.norm(pop.S[0] - np.diag(np.diag(pop.S[0]))), 0.)
            for name in ('B', 'D', 'pi', 'm', 'S'):
                np.testing.assert_array_equal(getattr(pop, name), getattr(make_population(d, k=k), name))
            details = population_details(pop, 'gaussian')
            self.assertEqual(details['population_design'], 'xu-gaussian-orthonormal-v1')
            self.assertEqual(details['population_seed'], 912)
            self.assertTrue({'beta', 'mu_F', 'Sigma_F', 'D', 'A', 'B', 'm', 'S',
                             'qr_sign_convention'} <= details.keys())

    def test_population_seed_and_reconstruction(self):
        pop = make_population(20, k=8, seed=1234)
        restored = population_from_checkpoint(20, 8, 'gaussian', population_metadata(pop, 'gaussian'))
        for name in ('B', 'D', 'm', 'S'):
            np.testing.assert_array_equal(getattr(pop, name), getattr(restored, name))
        self.assertFalse(np.array_equal(pop.B, make_population(20, k=8).B))
        for seed in (-1, True, 1.5, None):
            with self.assertRaises(ValueError):
                make_population(20, k=8, seed=seed)

    def test_paper_dimensions_sample_without_dense_covariance(self):
        pop = make_population(2048, k=16)
        with patch.object(FactorPopulation, 'covs', new_callable=PropertyMock,
                          side_effect=AssertionError('dense covariance in sampler')):
            c = pop.threshold(.05)
            for threshold in (None, c):
                x = pop.sample(256, np.random.default_rng(82), c=threshold)
                self.assertEqual(x.shape, (256, 2048))
                self.assertTrue(np.isfinite(x).all())
                if threshold is not None:
                    self.assertLessEqual(np.max(x @ pop.w), threshold + 1e-12)

    def test_latent_sampling_and_exact_pit(self):
        from scipy.stats import norm
        for case in ('gaussian',):
            pop = make_population(12, case, k=8)
            returns, factors, labels = pop.sample_returns(12000, np.random.default_rng(67))
            residuals = (returns - factors @ pop.B.T) / np.sqrt(pop.D)
            conditional_pit = pop.exact_conditional_cdf_given_F(returns, factors)
            np.testing.assert_allclose(conditional_pit, norm.cdf(residuals))
            self.assertLess(kstest(conditional_pit.ravel(), 'uniform').statistic, .006)
            np.testing.assert_allclose(np.cov(residuals, rowvar=False), np.eye(pop.d), atol=.045)
            np.testing.assert_allclose(np.bincount(labels, minlength=len(pop.pi))/len(labels), pop.pi, atol=.015)
            grid = np.linspace(-2, 2, 51)[:, None] * np.ones((1, pop.d))
            expected = sum(weight * norm.cdf((grid - mean) / np.sqrt(np.diag(cov)))
                           for weight, mean, cov in zip(pop.pi, pop.mu, pop.covs))
            np.testing.assert_allclose(pop.exact_marginal_cdf(grid), expected)
            mean_ks, std_ks = pop.evaluate_generated_samples(returns)
            self.assertLess(mean_ks, .02)
            self.assertGreaterEqual(std_ks, 0)
            shifted_ks, _ = pop.evaluate_generated_samples(returns + 1.)
            self.assertGreater(shifted_ks, mean_ks + .2)
            with self.assertRaises(ValueError):
                pop.evaluate_generated_samples(np.empty((0, pop.d)))
            with self.assertRaises(ValueError):
                pop.exact_conditional_cdf_given_F(returns, factors[:-1])
            with self.assertRaises(ValueError):
                pop.exact_marginal_cdf(np.zeros((2, pop.d + 1)))

    def test_eight_factor_population(self):
        for case, regimes in (('gaussian', 1),):
            pop = make_population(100, case, k=8)
            self.assertEqual(pop.B.shape, (100, 8))
            self.assertEqual(pop.m.shape, (regimes, 8))
            self.assertEqual(pop.S.shape, (regimes, 8, 8))
            np.testing.assert_allclose(pop.B.T @ pop.B, np.eye(8), atol=1e-14)
            self.assertGreater(np.linalg.eigvalsh(pop.S).min(), 0)
            np.testing.assert_array_equal(pop.B, make_population(100, case, k=8).B)
            validate_truth(pop, 20000, 913)
            c = pop.threshold(.05)
            x = pop.sample(10000, np.random.default_rng(78), c)

    def test_dimensions_and_presets(self):
        for d, k in ((2, 3), (8, 9), (8, 2), (8, 0), (8, 3.5), (True, 3)):
            with self.assertRaisesRegex(ValueError, '3 <= k <= d'):
                make_population(d, k=k)
            with self.assertRaises(ValueError):
                Config(d=d, k=k)
        self.assertEqual(make_population(20, k=9).k, 9)
        self.assertEqual(Config(d=20, k=9).k, 9)
        # Square bases are supported, even though presets use k < d.
        np.testing.assert_allclose(make_population(8, k=8).B.T @ make_population(8, k=8).B,
                                   np.eye(8), atol=1e-14)
        self.assertEqual((Config().d, Config().k), (100, 8))
        self.assertEqual(Config().dfm_epochs, 1000)
        for mode in ('smoke', 'standard'):
            cfg = Config.preset(mode)
            self.assertEqual(make_population(cfg.d, k=cfg.k).k, cfg.k)

    def test_checkpoint_population_contract(self):
        pop = make_population(100, 'gaussian', k=8)
        metadata = population_metadata(pop, 'gaussian')
        restored = population_from_checkpoint(100, 8, 'gaussian', metadata)
        np.testing.assert_array_equal(restored.B, pop.B)
        np.testing.assert_array_equal(restored.S, pop.S)
        for invalid in ({}, dict(metadata, population_k=3), dict(metadata, population_d=50),
                        dict(metadata, population_case='unsupported'), dict(metadata, population_design='old'),
                        dict(metadata, population_design='financial-regime-factors-v3'),
                        dict(metadata, population_design='diagonal-regime-factors-v2'),
                        dict(metadata, population_seed=None), dict(metadata, population_seed=-1),
                        dict(metadata, qr_sign_convention='unspecified')):
            with self.assertRaisesRegex(ValueError, 'Retrain the DFM and H'):
                population_from_checkpoint(100, 8, 'gaussian', invalid)
        wrong_model = UnconditionalDFM(100, 3, hidden_dims=[8])
        with self.assertRaisesRegex(ValueError, 'retrain'):
            check_population_model(pop, wrong_model)
        ref = FrozenDFMReference(UnconditionalDFM(100, 8, hidden_dims=[8]).double())
        h = HNetwork(ref, ConditioningConfig(pop.w, pop.threshold(.05)), hidden_dims=[8])
        check_population_h(pop, h)
        self.assertEqual(h.features(0., torch.zeros(2, 100, dtype=torch.float64)).shape, (2, 9))
        self.assertEqual(h.network[0].in_features, 10)
    def test_log_domain_extreme_tail(self):
        pop=make_population(8,'gaussian',k=3)
        a,v=pop.portfolio; c=float(a[0]-40*np.sqrt(v[0]))
        self.assertLess(pop.log_event(c),-800)
        x=pop.sample(1000,np.random.default_rng(17),c)
        self.assertTrue(np.isfinite(x).all())
        self.assertLessEqual(np.max(x@pop.w),c+1e-12)
        m,s=pop.moments(c)
        self.assertTrue(np.isfinite(m).all())
        self.assertGreater(np.linalg.eigvalsh(s).min(),0)
    def test_streams_unique_reproducible(self):
        a=Streams(42); b=Streams(42)
        self.assertEqual(a.seeds,b.seeds)
        self.assertEqual(len(set(a.seeds.values())),len(a.seeds))


class ConfigurationTests(unittest.TestCase):
    def test_configuration_builds_the_chen_population_and_validates_settings(self):
        from synthetic_benchmark.unconditional import population_for_config
        np.testing.assert_array_equal(population_for_config(Config(d=30, k=4)).B, make_population(30, k=4).B)
        for kwargs in (dict(dfm_network='unet'), dict(torch_threads=0), dict(cases=('heavy',))):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                Config(**kwargs)
        with self.assertRaises(ValueError):
            Config.preset('research')


if __name__=='__main__': unittest.main()
