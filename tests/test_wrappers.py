import unittest

import torch

from wrappers import HGuidance, QGuidance


class GuidanceTests(unittest.TestCase):
    def test_analytic_gradients_and_projection_inside_no_grad(self):
        state = torch.tensor([[0.2, -0.3], [0.4, 0.1]], dtype=torch.float64)
        basis = torch.tensor([[0.6], [0.8]], dtype=state.dtype)
        log_h = lambda x, t: -0.5 * x.square().sum(1, keepdim=True) - t
        h = lambda x, t: log_h(x, t).exp()
        q = lambda x, t: -x * h(x, t)
        for projection in (None, basis):
            expected = -2 * state if projection is None else -2 * (state @ basis) @ basis.T
            adapters = [HGuidance(h, B=projection, scale=2),
                        HGuidance(log_h, B=projection, output_is_log=True, scale=2),
                        QGuidance(h, q, B=projection, scale=2)]
            for adapter in adapters:
                for time in (0.2, torch.tensor([0.1, 0.3]), torch.tensor([[0.1], [0.3]])):
                    with self.subTest(adapter=type(adapter).__name__, projection=projection, time=time):
                        with torch.no_grad():
                            actual = adapter(state, time)
                        torch.testing.assert_close(actual, expected)
                        self.assertFalse(actual.requires_grad)

    def test_invalid_projection_scale_state_time_and_outputs(self):
        state = torch.zeros(2, 2, dtype=torch.float64)
        h = lambda x, t: (-x.square().sum(1, keepdim=True)).exp()
        invalid = [lambda: HGuidance(h, B=torch.ones(2, 1)),
                   lambda: HGuidance(h, scale=-1),
                   lambda: HGuidance(h, scale=float("nan")),
                   lambda: HGuidance(h)(state.long(), 0.0),
                   lambda: HGuidance(h)(state, torch.zeros(3)),
                   lambda: HGuidance(h)(state, float("inf")),
                   lambda: HGuidance(lambda x, t: x)(state, 0.0),
                   lambda: HGuidance(lambda x, t: h(x, t).float())(state, 0.0),
                   lambda: HGuidance(lambda x, t: h(x, t) * float("nan"))(state, 0.0),
                   lambda: HGuidance(lambda x, t: -h(x, t))(state, 0.0),
                   lambda: QGuidance(h, lambda x, t: t)(state, 0.0)]
        for index, call in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(ValueError):
                call()

    def test_disconnected_h_and_inference_mode_fail_explicitly(self):
        state = torch.zeros(2, 2, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "disconnected"):
            HGuidance(lambda x, t: torch.ones_like(t))(state, 0.0)
        with torch.inference_mode(), self.assertRaisesRegex(ValueError, "autograd"):
            HGuidance(lambda x, t: x.sum(1, keepdim=True), output_is_log=True)(state, 0.0)
