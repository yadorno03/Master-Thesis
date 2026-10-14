import unittest
from unittest.mock import patch

import torch

from conditioning import ConditioningConfig
from h_model import HNetwork
from sampling import generate_samples
import test_h_network


def generator(seed=51):
    return torch.Generator().manual_seed(seed)


class HTrajectoryTests(unittest.TestCase):
    def test_real_paths_features_times_weights_and_terminal_labels(self):
        for dtype in (torch.float32, torch.float64):
            model = test_h_network.HNetworkTests().fixture(dtype)
            c = model
            paths = generate_samples(c.reference, 5, num_steps=4,
                                     return_trajectory=True, generator=generator())
            data = model.sample_trajectories(5, 4, generator=generator())
            expected_times = torch.tensor([0., .25, .5, .75], dtype=dtype)*model.horizon
            torch.testing.assert_close(data.times, expected_times)
            torch.testing.assert_close(data.weights, torch.full((4,), .25, dtype=dtype))
            self.assertEqual(data.features.shape, (5, 4, 3))
            self.assertEqual(data.labels.shape, (5, 1))
            for j, time in enumerate(expected_times):
                torch.testing.assert_close(data.features[:, j], paths[:, j] @ c.G(time).T)
            expected_labels = (paths[:, -1] @ c.w <= c.C).to(dtype)[:, None]
            torch.testing.assert_close(data.labels, expected_labels, rtol=0, atol=0)
            for tensor in (data.times, data.features, data.labels, data.weights):
                self.assertEqual(tensor.dtype, dtype)
                self.assertEqual(tensor.device, next(model.parameters()).device)
                self.assertFalse(tensor.requires_grad)
                self.assertIsNone(tensor.grad_fn)

    def test_full_residual_event_inclusive_boundary_and_one_step(self):
        ref = test_h_network.HNetworkTests().fixture().reference
        with torch.no_grad():
            ref._score.V.copy_(torch.eye(4, 2, dtype=torch.float64))
        model = HNetwork(ref, ConditioningConfig((0., 0., 0., 1.), .25))
        paths = torch.zeros(3, 2, 4, dtype=torch.float64)
        paths[:, 0, -1] = torch.tensor([1., 1., -1.])  # opposite initial labels
        paths[:, 1, -1] = torch.tensor([-.5, .25, .5])
        with patch('h_model.generate_samples', return_value=paths) as sampler:
            data = model.sample_trajectories(3, 1, generator=generator())
        self.assertIs(sampler.call_args.args[0], ref)
        self.assertEqual(sampler.call_args.kwargs['num_steps'], 1)
        self.assertTrue(sampler.call_args.kwargs['return_trajectory'])
        self.assertNotIn('guidance_fn', sampler.call_args.kwargs)
        self.assertNotIn('initial_state', sampler.call_args.kwargs)
        self.assertEqual(data.labels[:, 0].tolist(), [1., 1., 0.])
        self.assertEqual(data.times.tolist(), [0.])
        self.assertEqual(data.weights.tolist(), [1.])
        torch.testing.assert_close(data.features[:, 0], paths[:, 0] @ model.G(0.).T)

    def test_partial_chunks_reproducibility_and_no_h_or_reference_mutation(self):
        model = test_h_network.HNetworkTests().fixture()
        model.train()
        model.network[0].eval()  # preserve mixed modes as well as frozen reference mode
        modes = [module.training for module in model.modules()]
        before = {key: value.clone() for key, value in model.state_dict().items()}
        rng = generator(72)
        chunks = [model.sample_trajectories(n, 3, generator=rng) for n in (2, 2, 1)]
        with patch('h_model.generate_samples', wraps=generate_samples) as sampler:
            actual = model.sample_trajectories(5, 3, batch_size=2, generator=generator(72))
        self.assertEqual([call.args[1] for call in sampler.call_args_list], [2, 2, 1])
        repeat = model.sample_trajectories(5, 3, batch_size=2, generator=generator(72))
        for name in ('features', 'labels'):
            expected = torch.cat([getattr(chunk, name) for chunk in chunks])
            torch.testing.assert_close(getattr(actual, name), expected, rtol=0, atol=0)
            torch.testing.assert_close(getattr(actual, name), getattr(repeat, name), rtol=0, atol=0)
        torch.testing.assert_close(actual.times, chunks[0].times)
        torch.testing.assert_close(actual.weights, chunks[0].weights)
        self.assertEqual([module.training for module in model.modules()], modes)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        with torch.no_grad():
            for parameter in model.network.parameters():
                parameter.fill_(float('nan'))  # collector must never evaluate the learned H
        unchanged = model.sample_trajectories(5, 3, batch_size=2, generator=generator(72))
        torch.testing.assert_close(unchanged.features, actual.features, rtol=0, atol=0)
        torch.testing.assert_close(unchanged.labels, actual.labels, rtol=0, atol=0)

    def test_invalid_sizes_fail_before_sampling_or_consuming_rng(self):
        model = test_h_network.HNetworkTests().fixture()
        rng = generator()
        state = rng.get_state().clone()
        for invalid in (0, -1, True, 1.5):
            for kwargs in ({'num_trajectories': invalid}, {'num_steps': invalid}, {'batch_size': invalid}):
                arguments = dict(num_trajectories=3, num_steps=2, generator=rng)
                arguments.update(kwargs)
                with patch('h_model.generate_samples') as sampler:
                    with self.assertRaises(ValueError):
                        model.sample_trajectories(**arguments)
                    sampler.assert_not_called()
                self.assertTrue(torch.equal(rng.get_state(), state))
