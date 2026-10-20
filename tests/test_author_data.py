import unittest
import torch
from author_data import ReturnTransform


class AuthorDataTests(unittest.TestCase):
    def test_upstream_coordinates_inverse_and_event(self):
        x = torch.tensor([[1., 2.], [3., 8.], [2., 4.]], dtype=torch.float64)
        p = ReturnTransform.fit(x, ['B', 'A'])
        expected = (x.float() - x.float().mean(0)) / x.float().std(0)
        torch.testing.assert_close(p(x), expected, rtol=0, atol=0)
        torch.testing.assert_close(p.inverse(p(x)), x.float())
        event = p.event([.2, .8], 4.)
        torch.testing.assert_close(p(x) @ torch.tensor(event.w) - event.C,
                                   x.float() @ torch.tensor([.2, .8]) - 4.)
        self.assertEqual(p.assets, ('B', 'A'))

    def test_degenerate_data_and_wrong_order_metadata(self):
        for x, names in [(torch.ones(3, 2), ['a', 'b']),
                         (torch.randn(3, 2), ['a', 'a']),
                         (torch.randn(1, 2), ['a', 'b'])]:
            with self.assertRaises(ValueError):
                ReturnTransform.fit(x, names)
