import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from run_author import main


@unittest.skipUnless(importlib.util.find_spec('diffusion_factor_model'), 'Install requirements-dfm.txt')
class AuthorRunnerTests(unittest.TestCase):
    def test_checkpoint_to_h_to_guided_outputs_without_notebooks(self):
        previous = torch.get_num_threads(); torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                p = Path(directory)
                np.save(p/'returns.npy', np.random.default_rng(2).normal(size=(6,4)))
                (p/'assets.json').write_text(json.dumps(['a','b','c','d']))
                (p/'weights.json').write_text(json.dumps([.25]*4))
                main(['train','--returns',str(p/'returns.npy'),'--assets',str(p/'assets.json'),
                      '--output',str(p/'fit'),'--height','2','--width','2','--model-dim','8',
                      '--timesteps','4','--save-index','1','--epochs','1','--batch-size','2','--no-amp'])
                main(['sample','--dfm',str(p/'fit/dfm.pt'),'--output',str(p/'plain.npy'),'--count','3'])
                main(['train-h','--dfm',str(p/'fit/dfm.pt'),'--weights',str(p/'weights.json'),
                      '--threshold','0','--output',str(p/'h.pt'),'--trajectories','6','--epochs','1'])
                main(['guide','--h',str(p/'h.pt'),'--output',str(p/'guided.npy'),'--count','3',
                      '--guidance-scale','0','--initialization','gaussian'])
                np.testing.assert_array_equal(np.load(p/'plain.npy'), np.load(p/'guided.npy'))
                self.assertFalse(json.loads((p/'guided.json').read_text())['released_config_defaults'])
        finally:
            torch.set_num_threads(previous)
