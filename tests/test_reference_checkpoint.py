from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from conditioning import portfolio_event_in_model_coordinates
from DFM import UnconditionalDFM
from h_model import HNetwork
from reference_checkpoint import load_reference_checkpoint
from sampling import generate_samples


class ReferenceCheckpointTests(unittest.TestCase):
    def fixture(self, dtype=torch.float64):
        config = dict(d=3, k=1, sigma_max=2., T=1.3, t0=.2, hidden_dims=[8])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(12)
            model = UnconditionalDFM(**config).to(dtype=dtype)
        saved = {'model_config': config, 'state_dict': model.state_dict(),
                 'assets': ['C', 'A', 'B'],
                 'preprocessing': {'mean': [.01, -.02, .03], 'std': [.2, .3, .4]}}
        return model, saved

    def test_load_preserves_reference_event_scaling_and_rng(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                model, saved = self.fixture(dtype)
                path = Path(directory) / 'reference.pt'
                torch.save(saved, path)
                original, rng = path.read_bytes(), torch.get_rng_state().clone()
                ref, assets, mean, std = load_reference_checkpoint(path)
                self.assertEqual(path.read_bytes(), original)
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))
                self.assertEqual(assets, ['C', 'A', 'B'])
                np.testing.assert_array_equal(mean, saved['preprocessing']['mean'])
                np.testing.assert_array_equal(std, saved['preprocessing']['std'])
                self.assertEqual((ref.T, ref.t0), (1.3, .2))
                self.assertTrue(all(p.dtype == dtype and p.device.type == 'cpu'
                                    and not p.requires_grad and p.grad is None for p in ref.parameters()))
                self.assertFalse(any(module.training for module in ref.modules()))
                for name, value in ref._score.state_dict().items():
                    torch.testing.assert_close(value, saved['state_dict']['factor_score_network.' + name],
                                               rtol=0, atol=0)
                x = torch.arange(12., dtype=dtype).reshape(4, 3).requires_grad_()
                times = x.new_full((4, 1), .7)
                expected = model.factor_score_network(x, times)
                actual = ref.score(x, times)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0],
                                           torch.autograd.grad(expected.sum(), x)[0], rtol=0, atol=0)
                weights, threshold = np.array([.2, -.1, .9]), .1
                event = portfolio_event_in_model_coordinates(weights, threshold, mean, std)
                raw_margin = (x.detach().numpy() * std + mean) @ weights - threshold
                np.testing.assert_allclose(x.detach().numpy() @ event.w - event.C, raw_margin)
                h = HNetwork(ref, event, hidden_dims=[4])
                torch.testing.assert_close(h.features(h.horizon, x)[:, -1], x @ h.w)
                bundle = Path(directory) / 'h.pt'
                metadata = {'assets': assets, 'preprocessing': saved['preprocessing']}
                h.save_checkpoint(bundle, metadata=metadata)
                restored, restored_metadata = HNetwork.load_checkpoint(bundle)
                self.assertEqual(restored_metadata, metadata)
                self.assertEqual(restored.config, event)
                self.assertEqual(restored.horizon, ref.T - ref.t0)
                torch.testing.assert_close(restored.reference.score(x, times), actual, rtol=0, atol=0)
                draws = [generate_samples(candidate, 4, num_steps=3,
                         generator=torch.Generator().manual_seed(37)) for candidate in (model, ref)]
                torch.testing.assert_close(*draws, rtol=0, atol=0)

    def test_rejects_invalid_metadata_and_state_without_advancing_rng(self):
        _, original = self.fixture()
        defects = [
            ('assets', value) for value in ('CAB', ['C', 'A'], ['C', 'C', 'B'],
                                           ['C', '', 'B'], ['C', 2, 'B'], ['C', [], 'B'])]
        defects += [('preprocessing', {'mean': mean, 'std': std}) for mean, std in (
            ([0., 0.], [1.] * 3), ([0.] * 3, [1., 0., 1.]),
            ([0.] * 3, [1., -1., 1.]), ([float('nan')] * 3, [1.] * 3),
            ([0.] * 3, [float('inf')] * 3), ([False] * 3, [1.] * 3),
            (torch.zeros(3, dtype=torch.complex128), [1.] * 3), ([[0.] * 3], [1.] * 3))]
        defects += [('model_config', {**original['model_config'], **change}) for change in (
            {'t0': 0.}, {'t0': 1.3}, {'T': float('inf')}, {'sigma_max': 0.}, {'d': 4})]
        defects.append(('model_config', {k: v for k, v in original['model_config'].items() if k != 'T'}))
        for defect in ('mixed_precision', 'half_precision', 'nonfinite', 'missing', 'unexpected', 'projection'):
            state = deepcopy(original['state_dict'])
            key = 'factor_score_network.c_raw'
            if defect == 'mixed_precision':
                state[key] = state[key].float()
            elif defect == 'half_precision':
                state = {name: value.half() for name, value in state.items()}
            elif defect == 'nonfinite':
                state[key].fill_(float('nan'))
            elif defect == 'missing':
                del state[key]
            elif defect == 'unexpected':
                state['extra'] = torch.ones(1, dtype=torch.float64)
            else:
                state['factor_score_network.V'].zero_()
            defects.append(('state_dict', state))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.pt'
            for index, (key, value) in enumerate(defects):
                with self.subTest(index=index, field=key):
                    torch.save({**original, key: value}, path)
                    rng = torch.get_rng_state().clone()
                    with self.assertRaises((ValueError, RuntimeError)):
                        load_reference_checkpoint(path)
                    self.assertTrue(torch.equal(torch.get_rng_state(), rng))
