import unittest
from unittest.mock import patch

import torch

from DFM import UnconditionalDFM
from sampling import generate_samples


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.model = UnconditionalDFM(2, 1, T=1.0, t0=0.2, hidden_dims=[4]).double()
        self.model.factor_score_network.g_zeta.eval()
        self.modes = [module.training for module in self.model.modules()]

    def test_reverse_times_guidance_and_supplied_path(self):
        initial = torch.tensor([[1.0, -2.0]], dtype=torch.float64, requires_grad=True)
        increments = torch.tensor([[[0.1, -0.2]], [[0.3, 0.4]]], dtype=torch.float64)
        original = increments.clone()
        score_times, guidance_times = [], []

        def score(state, time):
            self.assertFalse(any(module.training for module in self.model.modules()))
            score_times.append(time.clone())
            return -state + time

        def guidance(state, time):
            guidance_times.append(time.clone())
            return torch.ones_like(state) * 0.25

        with patch.object(self.model.factor_score_network, "forward", side_effect=score):
            path = generate_samples(self.model, 1, num_steps=2, initial_state=initial,
                                    brownian_increments=increments, guidance_fn=guidance,
                                    return_trajectory=True)
        first = initial.detach() + 0.4 * (-0.5 * initial.detach() + 1.0 + 0.25) + increments[0]
        last = first + 0.4 * (-0.5 * first + 0.6 + 0.25) + increments[1]
        torch.testing.assert_close(path, torch.stack([initial.detach(), first, last], dim=1))
        torch.testing.assert_close(torch.cat(score_times), torch.tensor([[1.0], [0.6]], dtype=torch.float64))
        torch.testing.assert_close(torch.cat(guidance_times), torch.tensor([[0.0], [0.4]], dtype=torch.float64))
        torch.testing.assert_close(increments, original)
        torch.testing.assert_close(initial, torch.tensor([[1.0, -2.0]]).double())
        self.assertFalse(path.requires_grad)
        self.assertEqual([module.training for module in self.model.modules()], self.modes)

    def test_seeded_gaussian_start_and_unit_diffusion(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                self.model.to(dtype=dtype)
                expected_rng = torch.Generator().manual_seed(42)
                expected = torch.randn(3, 2, dtype=dtype, generator=expected_rng)
                for _ in range(2):
                    noise = torch.randn(3, 2, dtype=dtype, generator=expected_rng)
                    expected = 0.75 * expected + 0.5 ** 0.5 * noise
                with patch.object(self.model.factor_score_network, "forward", side_effect=lambda x, t: -x):
                    actual = generate_samples(self.model, 3, num_steps=2, t_0=0,
                                              generator=torch.Generator().manual_seed(42))
                torch.testing.assert_close(actual, expected)

    def test_invalid_inputs_and_mode_restoration_after_failure(self):
        invalid = [dict(num_steps=0), dict(num_steps=True), dict(t_0=1.0),
                   dict(t_0=float("nan")), dict(return_trajectory=1),
                   dict(initial_state=torch.zeros(1, 2, dtype=torch.float32)),
                   dict(initial_state=torch.zeros(1, 3, dtype=torch.float64)),
                   dict(initial_state=torch.full((1, 2), float("nan"), dtype=torch.float64)),
                   dict(brownian_increments=torch.zeros(1, 1, 2, dtype=torch.float64))]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate_samples(self.model, 1, **kwargs)
        with patch.object(self.model.factor_score_network, "forward", side_effect=RuntimeError("score failed")):
            with self.assertRaisesRegex(RuntimeError, "score failed"):
                generate_samples(self.model, 1, num_steps=1)
        self.assertEqual([module.training for module in self.model.modules()], self.modes)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            generate_samples(self.model, 1, num_steps=1,
                             guidance_fn=lambda x, t: torch.full_like(x, float("nan")))
        self.assertEqual([module.training for module in self.model.modules()], self.modes)


if __name__ == "__main__":
    unittest.main()
