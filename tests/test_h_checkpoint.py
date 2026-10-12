from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

from h_model import HNetwork
import test_h_network


class HCheckpointTests(unittest.TestCase):
    def test_roundtrip_preserves_reference_event_predictions_and_gradients(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                model = test_h_network.HNetworkTests().fixture(dtype)
                ref = model.conditioning.reference
                metadata = {"assets": ["A", "B", "C", "D"], "mean": [0.] * 4,
                            "std": [1.] * 4, "seed": 27, "steps": 8}
                state = deepcopy(model.state_dict())
                modes = [module.training for module in model.modules()]
                rng = torch.get_rng_state().clone()
                path = Path(directory) / "h.pt"
                model.save_checkpoint(path, metadata=metadata)
                restored, saved_metadata = HNetwork.load_checkpoint(path)
                self.assertEqual(saved_metadata, metadata)
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))
                self.assertEqual(modes, [module.training for module in model.modules()])
                self.assertFalse(any(module.training for module in restored.modules()))
                self.assertEqual(restored.conditioning.config, model.conditioning.config)
                loaded_ref = restored.conditioning.reference
                self.assertEqual((loaded_ref.T, loaded_ref.t0), (ref.T, ref.t0))
                self.assertEqual(restored.horizon, model.horizon)
                self.assertTrue(all(not p.requires_grad and p.grad is None
                                    for p in loaded_ref.parameters()))
                self.assertTrue(all(p.requires_grad for p in restored.network.parameters()))
                for name, value in state.items():
                    torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
                    torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
                x = torch.arange(12., dtype=dtype).reshape(3, 4).requires_grad_()
                for t in (0., .3, model.horizon):
                    s = ref.t0 if t == model.horizon else ref.T-t
                    torch.testing.assert_close(loaded_ref.score(x, s), ref.score(x, s))
                    outputs, gradients = [], []
                    for candidate in (model, restored):
                        y = candidate.conditioning.features(t, x)
                        outputs.append(candidate(t, y))
                        gradients.append(torch.autograd.grad(candidate.log_prob(t, y).sum(), x)[0])
                    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
                    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
                for candidate in (model, restored):
                    paths = candidate.sample_trajectories(5, 8,
                        generator=torch.Generator().manual_seed(94))
                    if candidate is model:
                        expected = paths
                    else:
                        torch.testing.assert_close(paths.features, expected.features, rtol=0, atol=0)
                        torch.testing.assert_close(paths.labels, expected.labels, rtol=0, atol=0)

    def test_invalid_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "h.pt"
            test_h_network.HNetworkTests().fixture().save_checkpoint(path)
            self.assertEqual(HNetwork.load_checkpoint(path)[1], {})
            original = torch.load(path, weights_only=True)
            for defect in ("version", "precision", "nonfinite", "dtype", "missing", "horizon", "event"):
                with self.subTest(defect=defect):
                    saved = deepcopy(original)
                    if defect == "version":
                        saved["format_version"] = 99
                    elif defect == "precision":
                        saved["dtype"] = "float16"
                    elif defect == "nonfinite":
                        saved["h"]["state_dict"]["0.weight"].fill_(float("nan"))
                    elif defect == "dtype":
                        saved["reference"]["state_dict"]["V"] = saved["reference"]["state_dict"]["V"].float()
                    elif defect == "missing":
                        del saved["h"]["state_dict"]["0.bias"]
                    elif defect == "horizon":
                        saved["reference"]["t0"] = saved["reference"]["T"]
                    else:
                        saved["event"]["w"] = (0., 0., 0., 0.)
                    torch.save(saved, path)
                    with self.assertRaises((ValueError, RuntimeError)):
                        HNetwork.load_checkpoint(path)

    def test_save_rejects_nonfinite_weights_and_non_json_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "h.pt"
            model = test_h_network.HNetworkTests().fixture()
            for metadata in ({"bad": float("nan")}, {"bad": torch.ones(1)}):
                with self.assertRaises((ValueError, TypeError)):
                    model.save_checkpoint(path, metadata=metadata)
            with torch.no_grad():
                model.network[-1].bias.fill_(float("inf"))
            with self.assertRaises(ValueError):
                model.save_checkpoint(path)
            self.assertFalse(path.exists())

    def test_save_rejects_unsupported_reference_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "h.pt"
            model = test_h_network.HNetworkTests().fixture()
            model.conditioning.reference._score.g_zeta[1] = torch.nn.Tanh()
            with self.assertRaisesRegex(ValueError, "Linear/ReLU"):
                model.save_checkpoint(path)
            self.assertFalse(path.exists())
