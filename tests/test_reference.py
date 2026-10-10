import unittest
import torch
from DFM import UnconditionalDFM
from reference import FrozenDFMReference


class FrozenReferenceTests(unittest.TestCase):
    def test_snapshot_preserves_score_and_input_gradient(self):
        for dtype in (torch.float32, torch.float64):
            model = UnconditionalDFM(3, 1, t0=0.1, T=1.0, hidden_dims=[4]).to(dtype)
            x = torch.ones(2, 3, dtype=dtype, requires_grad=True)
            expected = model.factor_score_network(x, torch.full((2, 1), 0.5, dtype=dtype))
            expected.sum().backward()
            gradient = x.grad.clone()
            ref = FrozenDFMReference(model).train()
            self.assertTrue(all(not m.training for m in ref.modules()))
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in ref.parameters()))
            with torch.no_grad():
                next(model.parameters()).add_(1)
            model.T, model.t0 = 2.0, 0.2
            self.assertEqual((ref.T, ref.t0), (1.0, 0.1))
            for time in (0.5, [0.5, 0.5], [[0.5], [0.5]]):
                actual = ref.factor_score_network(x, time)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0], gradient)

    def test_invalid_inputs_and_horizon(self):
        model = UnconditionalDFM(3, 1, t0=0.1)
        ref, x = FrozenDFMReference(model), torch.ones(2, 3)
        for state, time in ((x.double(), 0.5), (x[:, :2], 0.5), (x, -0.1), (x, True), (x, [0.5]*3)):
            with self.assertRaises(ValueError): ref.score(state, time)
        model.t0 = 0
        with self.assertRaises(ValueError): FrozenDFMReference(model)
