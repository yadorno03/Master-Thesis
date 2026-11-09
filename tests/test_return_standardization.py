"""Per-asset return standardization: the map, the portfolio event, and every path that crosses it."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from DFM import ReturnStandardization, UnconditionalDFM
from conditioning import ConditioningConfig
from guided_sampling import exact_final_step
from h_model import HNetwork
from reference import FrozenDFMReference
from sampling import generate_samples
from synthetic_conditional_truth import make_population, population_metadata
from synthetic_benchmark.configuration import Config, Streams
from synthetic_benchmark.unconditional import fit_dfm
from synthetic_benchmark.conditional import load_frozen_dfm, sample_dfm_rejection, sample_unguided_exact


def standardized_dfm(d=4, k=2, dtype=torch.float64, seed=5):
    """A small untrained DFM whose standardization is far from the identity."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = UnconditionalDFM(d, k, sigma_max=2., T=1.5, t0=.05, hidden_dims=[6]).to(dtype)
    returns = 3. + np.random.default_rng(seed).normal(size=(50, d)) * np.linspace(.01, 2., d)
    model.standardization.fit(returns)
    return model, returns


class ReturnStandardizationTests(unittest.TestCase):
    def test_identity_until_fit(self):
        standardization = ReturnStandardization(3)
        y = torch.randn(5, 3, generator=torch.Generator().manual_seed(1))
        torch.testing.assert_close(standardization.standardize(y), y, rtol=0, atol=0)
        np.testing.assert_array_equal(standardization.to_returns(y), y.numpy())
        w, C = standardization.portfolio_event([.2, .3, .5], -.1)
        np.testing.assert_array_equal(w, [.2, .3, .5])
        self.assertEqual(C, -.1)

    def test_fit_uses_the_mean_and_unbiased_standard_deviation_of_each_asset(self):
        returns = np.random.default_rng(1).normal(size=(40, 3)) * [1e-3, 1., 50.] + [5., -2., 0.]
        standardization = ReturnStandardization(3).double()
        standardization.fit(returns)
        np.testing.assert_allclose(standardization.mean.numpy(), returns.mean(0), rtol=1e-12)
        np.testing.assert_allclose(standardization.scale.numpy(), returns.std(0, ddof=1), rtol=1e-12)
        y = standardization.standardize(torch.tensor(returns)).numpy()
        np.testing.assert_allclose(y.mean(0), 0., atol=1e-12)
        np.testing.assert_allclose(y.std(0, ddof=1), 1., rtol=1e-12)

    def test_round_trip_keeps_the_dtype_of_the_states(self):
        for dtype, tolerance in ((torch.float32, 1e-5), (torch.float64, 1e-12)):
            with self.subTest(dtype=dtype):
                model, returns = standardized_dfm(dtype=dtype)
                standardization = model.standardization
                back = standardization.to_returns(standardization.standardize(torch.tensor(returns, dtype=dtype)))
                self.assertEqual(back.dtype, torch.empty((), dtype=dtype).numpy().dtype)
                np.testing.assert_allclose(back, returns, rtol=tolerance)

    def test_portfolio_event_is_the_return_event(self):
        model, _ = standardized_dfm()
        standardization = model.standardization
        weights, threshold = np.array([.4, -.1, .2, .5]), 1.7
        w, C = standardization.portfolio_event(weights, threshold)
        np.testing.assert_allclose(w, standardization.scale.numpy() * weights, rtol=0, atol=0)
        y = torch.randn(200, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
        returns = standardization.to_returns(y)
        # The same margin on both sides of the threshold, not only the same sign.
        np.testing.assert_allclose(y.numpy() @ w - C, returns @ weights - threshold, rtol=0, atol=1e-12)

    def test_invalid_inputs(self):
        standardization = ReturnStandardization(3)
        for returns in (np.ones((1, 3)), np.ones((4, 2)), np.full((4, 3), np.nan)):
            with self.subTest(shape=returns.shape), self.assertRaises(ValueError):
                standardization.fit(returns)
        constant = np.random.default_rng(0).normal(size=(5, 3))
        constant[:, 1] = 2.
        with self.assertRaisesRegex(ValueError, 'positive standard deviation'):
            standardization.fit(constant)
        for weights, threshold in (([1., 2.], 0.), ([1., np.nan, 0.], 0.), ([1., 0., 0.], np.inf)):
            with self.assertRaises(ValueError):
                standardization.portfolio_event(weights, threshold)


class StandardizedPipelineTests(unittest.TestCase):
    def test_reference_copies_and_validates_the_standardization(self):
        model, _ = standardized_dfm()
        reference = FrozenDFMReference(model)
        for name in ('mean', 'scale'):
            torch.testing.assert_close(getattr(reference.standardization, name),
                                       getattr(model.standardization, name), rtol=0, atol=0)
        # A copy: later changes to the DFM do not reach the frozen reference.
        with torch.no_grad():
            model.standardization.scale.mul_(2.)
        self.assertFalse(torch.equal(reference.standardization.scale, model.standardization.scale))
        with torch.no_grad():
            model.standardization.scale[0] = 0.
        with self.assertRaisesRegex(ValueError, 'strictly positive'):
            FrozenDFMReference(model)

    def test_h_checkpoint_keeps_the_standardization(self):
        model, returns = standardized_dfm()
        reference = FrozenDFMReference(model)
        h = HNetwork(reference, ConditioningConfig(*reference.standardization.portfolio_event(np.full(4, .25), 3.)),
                     hidden_dims=(5,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'h.pt'
            h.save_checkpoint(path)
            restored, _ = HNetwork.load_checkpoint(path)
        self.assertEqual(restored.config, h.config)
        for name, value in h.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
        np.testing.assert_allclose(restored.reference.standardization.to_returns(torch.zeros(1, 4, dtype=torch.float64)),
                                   returns.mean(0, keepdims=True), rtol=1e-12)

    def test_unguided_exact_samples_are_returns_inside_the_event(self):
        model, returns = standardized_dfm()
        reference = FrozenDFMReference(model)
        weights = np.full(4, .25)
        threshold = float(np.quantile(returns @ weights, .2))
        h = HNetwork(reference, ConditioningConfig(*reference.standardization.portfolio_event(weights, threshold)),
                     hidden_dims=(5,))
        samples = sample_unguided_exact(h, n_samples=64, batch_size=32, steps=6, seed=11)
        self.assertEqual(samples.shape, (64, 4))
        self.assertTrue(np.all(samples @ weights <= threshold + 1e-12))
        # The rows are the model states of the same draws, mapped to returns.
        generator = torch.Generator().manual_seed(11)
        states = torch.cat([generate_samples(reference, 32, num_steps=6, generator=generator,
                                             final_step=lambda x, g: exact_final_step(h, x, 6, g))
                            for _ in range(2)])
        np.testing.assert_array_equal(samples, reference.standardization.to_returns(states))

    def test_rejection_checks_the_event_on_returns(self):
        model, returns = standardized_dfm(dtype=torch.float32)
        reference = FrozenDFMReference(model)
        weights = np.full(4, .25)
        threshold = float(np.quantile(returns @ weights, .5))
        samples, info = sample_dfm_rejection(reference, weights, threshold, 20, steps=5, batch_size=16, seed=9,
                                             max_proposals=400)
        self.assertEqual(info['status'], 'complete')
        self.assertTrue(np.all(samples @ weights <= threshold))
        generator = torch.Generator().manual_seed(9)
        proposals = np.concatenate([reference.standardization.to_returns(
            generate_samples(reference, 16, num_steps=5, generator=generator))
            for _ in range(info['total_proposals'] // 16)])
        np.testing.assert_array_equal(samples, proposals[proposals @ weights <= threshold][:20])

    def test_saved_dfm_restores_its_standardization_and_older_checkpoints_load_as_identity(self):
        population = make_population(6, 'gaussian', k=3)
        config = Config(d=6, k=3, width=4, T=1.5, t0=.1)
        with torch.random.fork_rng():
            torch.manual_seed(2)
            model = UnconditionalDFM(6, 3, sigma_max=2., T=1.5, t0=.1, hidden_dims=[4, 4])
        model.standardization.fit(population.sample(30, np.random.default_rng(3)))
        with tempfile.TemporaryDirectory() as directory:
            for name, state in (('current', model.state_dict()),
                                ('older', {key: value for key, value in model.state_dict().items()
                                           if not key.startswith('standardization.')})):
                path = Path(directory) / f'{name}.pt'
                torch.save(dict(state=state, config=asdict(config), selected_epoch=1,
                                **population_metadata(population, 'gaussian')), path)
                reference = load_frozen_dfm(path, 'gaussian', 'cpu')[-1]
                expected = model.standardization if name == 'current' else ReturnStandardization(6)
                for buffer in ('mean', 'scale'):
                    torch.testing.assert_close(getattr(reference.standardization, buffer),
                                               getattr(expected, buffer), rtol=0, atol=0)

    def test_fit_standardizes_with_training_rows_only(self):
        config = Config(d=6, k=3, width=4, batch=16, dfm_epochs=2, dfm_initialization='pca',
                        dfm_time_sampling='log_balanced', dfm_standardization='per_asset')
        rng = np.random.default_rng(4)
        train, validation = rng.normal(size=(64, 6)), rng.normal(size=(20, 6))
        shift, scale = 3. * rng.normal(size=6), np.exp(rng.normal(size=6))
        a = fit_dfm(train, validation, config, Streams(3))[0]
        b = fit_dfm(shift + scale * train, shift + scale * validation, config, Streams(3))[0]
        np.testing.assert_allclose(b.standardization.mean.numpy(), shift + scale * train.mean(0), rtol=1e-6)
        np.testing.assert_allclose(b.standardization.scale.numpy(), scale * train.std(0, ddof=1), rtol=1e-6)
        # A per-asset affine change of the data does not change what the DFM learns.
        for name, value in a.factor_score_network.state_dict().items():
            torch.testing.assert_close(b.factor_score_network.state_dict()[name], value, rtol=1e-4, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
