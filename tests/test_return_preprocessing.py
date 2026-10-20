import unittest
from unittest.mock import patch
import numpy as np
import torch
from return_preprocessing import ReturnTransform, NumpyStandardizer
from author_data import ReturnTransform as AuthorTransform
from equity_preprocessing import standardize_training


class ReturnPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.raw = np.random.default_rng(17).normal(size=(103, 4)) * .01 + .03
        self.assets = ('D', 'B', 'C', 'A')

    def test_author_recipe_bitwise_and_numpy_facade(self):
        # Nonrepresentable float64 observations catch cast-after-fit mistakes.
        p = AuthorTransform.fit(self.raw, self.assets)
        self.assertIs(AuthorTransform, ReturnTransform)
        x = torch.tensor(self.raw, dtype=torch.float32)
        torch.testing.assert_close(p.mean, x.mean(0), rtol=0, atol=0)
        torch.testing.assert_close(p.std, x.std(0), rtol=0, atol=0)
        torch.testing.assert_close(p(x), (x-x.mean(0))/x.std(0), rtol=0, atol=0)
        adapter = NumpyStandardizer.from_transform(p)
        self.assertIs(adapter.tensor_transform, p)
        np.testing.assert_array_equal(adapter.transform(self.raw), p(x).numpy())
        np.testing.assert_allclose(adapter.inverse(adapter.transform(self.raw)), self.raw,
                                   rtol=2e-6, atol=1e-8)

    def test_numpy_recipe_and_previous_covariance_reduction(self):
        p = NumpyStandardizer.fit(self.raw, self.assets)
        mean, std = self.raw.mean(0), self.raw.std(0, ddof=1)
        np.testing.assert_array_equal(p.mean, mean)
        np.testing.assert_array_equal(p.std, std)
        np.testing.assert_array_equal(p.transform(self.raw), (self.raw-mean)/std)
        old_std = np.sqrt(np.diag(np.cov(self.raw, rowvar=False, ddof=1)))
        np.testing.assert_allclose(p.std, old_std, rtol=2e-15, atol=0)
        # Python-list checkpoint statistics must not round through float32.
        restored = NumpyStandardizer(mean.tolist(), std.tolist(), self.assets)
        np.testing.assert_array_equal(restored.mean, mean)
        np.testing.assert_array_equal(restored.std, std)
        np.testing.assert_array_equal(NumpyStandardizer.fit(self.raw.tolist()).mean, mean)

    def test_reuse_for_heldout_inverse_and_event(self):
        p = NumpyStandardizer.fit(self.raw, self.assets)
        heldout = self.raw[:9] + 2
        before = (p.mean, p.std)
        with patch.object(ReturnTransform, 'fit', side_effect=AssertionError('refit')):
            z = p.transform(heldout, assets=self.assets)
            np.testing.assert_allclose(p.inverse(z, assets=self.assets), heldout, atol=1e-15)
            weights, threshold = np.array([.2, -.3, .4, .7]), 2.
            event = p.event(weights, threshold, assets=self.assets)
            np.testing.assert_allclose(z @ np.array(event.w) - event.C,
                                       heldout @ weights - threshold, atol=1e-15)
            np.testing.assert_array_equal(z @ np.array(event.w) <= event.C,
                                          heldout @ weights <= threshold)
        np.testing.assert_array_equal(p.mean, before[0])
        np.testing.assert_array_equal(p.std, before[1])
        # Population vectors used by the synthetic OU helper remain supported.
        np.testing.assert_allclose(p.inverse(p.transform(self.raw[0])), self.raw[0])

    def test_equity_split_keeps_training_statistics(self):
        import pandas as pd
        frame = pd.DataFrame(self.raw[:5], columns=self.assets,
            index=pd.to_datetime(['2000-01-01', '2010-01-01', '2019-12-31', '2020-01-01', '2025-12-31']))
        train, test, mean, std, z = standardize_training(frame)
        np.testing.assert_array_equal(mean, frame.iloc[:3].to_numpy().mean(0))
        np.testing.assert_array_equal(std, frame.iloc[:3].to_numpy().std(0, ddof=1))
        np.testing.assert_array_equal(z, (train.to_numpy()-mean)/std)
        changed = frame.copy(); changed.iloc[3:] += 1e6
        np.testing.assert_array_equal(standardize_training(changed)[2], mean)
        shared = NumpyStandardizer.fit(train.to_numpy(), self.assets)
        with patch.object(ReturnTransform, 'fit', side_effect=AssertionError('refit')):
            np.testing.assert_array_equal(standardize_training(frame, transform=shared)[4], z)
        self.assertEqual(len(test), 2)

    def test_metadata_and_invalid_values(self):
        p = NumpyStandardizer.fit(self.raw, self.assets)
        for method in (p.transform, p.inverse):
            with self.assertRaises(ValueError):
                method(self.raw, assets=tuple(reversed(self.assets)))
            for bad in (np.zeros((0, 4)), np.ones((2, 3)), np.ones((2, 1, 4)),
                        np.full((2, 4), np.inf), np.ones((2, 4), bool), np.ones((2, 4), complex)):
                with self.subTest(shape=bad.shape, dtype=bad.dtype), self.assertRaises(ValueError):
                    method(bad)
        with self.assertRaises(ValueError):
            p.event([1]*4, 0, assets=tuple(reversed(self.assets)))
        for names in (['A']*4, ['A', 'B', '', 'C'], ['A', 'B', ' ', 'C'], ['A'], 'ABCD', [1,2,3,4]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                ReturnTransform.fit(self.raw, names)
        for bad in (np.ones((3, 4)), self.raw[:1], np.empty((3, 0)),
                    np.full((3, 4), np.nan), np.ones((3, 4), bool), np.ones((3, 4), complex)):
            for precision in ('numpy', 'author'):
                with self.subTest(precision=precision), self.assertRaises(ValueError):
                    ReturnTransform.fit(bad, self.assets, precision=precision)
        for mean, std in (([0., 0.], [1., 0.]), ([True, False], [1., 1.]),
                          ([0j, 0j], [1., 1.]), ([np.inf, 0.], [1., 1.])):
            with self.assertRaises(ValueError):
                NumpyStandardizer(mean, std)

    def test_buffers_device_gradients_and_parameter_ownership(self):
        mean, std = torch.zeros(4), torch.ones(4)
        p = ReturnTransform(mean, std, self.assets)
        mean.add_(3); std.add_(3)
        torch.testing.assert_close(p.mean, torch.zeros(4))
        self.assertEqual(set(p.state_dict()), {'mean', 'std'})
        x = torch.ones(2, 4, requires_grad=True)
        p.inverse(p(x)).sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))
        p.double()
        self.assertEqual(p(x).dtype, torch.float64)
        adapter = NumpyStandardizer.from_transform(p)
        changed = adapter.mean; changed[:] = 100
        np.testing.assert_array_equal(adapter.mean, np.zeros(4))
        if torch.cuda.is_available():
            p.cuda()
            self.assertEqual(p(x).device.type, 'cuda')
            np.testing.assert_array_equal(adapter.transform(np.ones((2, 4))), np.ones((2, 4)))


if __name__ == '__main__':
    unittest.main()
