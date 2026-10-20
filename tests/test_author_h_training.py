import importlib.util
import unittest
import torch
from conditioning import ConditioningConfig
from author_h import DDPMHNetwork
from author_h_training import collect_ddpm_labels, train_ddpm_h, evaluate_ddpm_h
from author_sampling import sample_author_ddpm
from test_author_reference import small_reference


@unittest.skipUnless(importlib.util.find_spec('diffusion_factor_model'), 'Install requirements-dfm.txt')
class DDPMHTrainingTests(unittest.TestCase):
    def test_grid_labels_reference_binding_and_frozen_weights(self):
        h = DDPMHNetwork(small_reference(), ConditioningConfig([.25]*4, 0), hidden_dims=(8,))
        before = {k: v.clone() for k,v in h.reference.state_dict().items()}
        data = collect_ddpm_labels(h, 6, batch_size=6, generator=torch.Generator().manual_seed(8))
        paths = sample_author_ddpm(h.reference, 6, generator=torch.Generator().manual_seed(8), return_trajectory=True)
        torch.testing.assert_close(data.states, paths[:, :-1])
        torch.testing.assert_close(data.labels[:,0], h.terminal_event(paths[:,-1]).float())
        history = train_ddpm_h(h, data, epochs=2, batch_size=4)
        self.assertEqual(len(history), 2)
        self.assertTrue(torch.isfinite(torch.tensor(history)).all())
        self.assertEqual(evaluate_ddpm_h(h, data)['trajectories'], 6)
        for key, value in h.reference.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        other = DDPMHNetwork(small_reference(), h.event, hidden_dims=(8,))
        with self.assertRaises(ValueError):
            train_ddpm_h(other, data)
