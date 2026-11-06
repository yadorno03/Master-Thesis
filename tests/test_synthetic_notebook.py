"""Execute the actual notebook paths with tiny training budgets, never 1,000 epochs."""
import ast
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


class SyntheticNotebookTests(unittest.TestCase):
    def test_notebook_uses_importable_support_without_import_side_effects(self):
        root = Path(__file__).resolve().parents[1]
        cells = json.loads((root / 'synthetic_dfm_conditional_benchmark.ipynb').read_text())['cells']
        # Markdown can be expanded without changing the executable contract.
        for i, cell in enumerate(cells, 1):
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            compile(source, f'cell {i}', 'exec')
            tree = ast.parse(source)
            self.assertFalse(any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                                 for node in ast.walk(tree)))
        # Preload dependencies, then import support in a fresh process and empty
        # working directory. Imports must neither draw randomness nor start a run.
        code = '''
import importlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import pandas
import matplotlib.pyplot
import scipy.spatial.distance
import scipy.stats
import IPython.display
import DFM, reference, conditioning, h_model, guided_sampling, sampling, synthetic_conditional_truth
np.random.seed(215)
torch.manual_seed(216)
numpy_before = np.random.get_state()
torch_before = torch.get_rng_state().clone()
files_before = set(Path.cwd().rglob('*'))
with patch.object(torch, 'manual_seed', side_effect=AssertionError('seed on import')), \\
     patch.object(np.random, 'default_rng', side_effect=AssertionError('stream on import')), \\
     patch.object(Path, 'mkdir', side_effect=AssertionError('directory on import')):
    for name in ('configuration', 'reporting', 'unconditional', 'conditional'):
        importlib.import_module('synthetic_benchmark.' + name)
assert torch.equal(torch_before, torch.get_rng_state())
numpy_after = np.random.get_state()
assert numpy_before[0] == numpy_after[0]
np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
assert numpy_before[2:] == numpy_after[2:]
assert files_before == set(Path.cwd().rglob('*'))
'''
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE='1',
                       MPLBACKEND='Agg', MPLCONFIGDIR=str(Path(tmp) / 'mpl'))
            subprocess.run([sys.executable, '-c', code], cwd=tmp, env=env,
                           capture_output=True, text=True, check=True, timeout=60)

    def test_every_cell_in_order_from_empty_directory(self):
        import nbformat
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
        from jupyter_client.kernelspec import KernelSpecManager

        root = Path(__file__).resolve().parents[1]
        notebook_path = root / 'synthetic_dfm_conditional_benchmark.ipynb'
        original = notebook_path.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / 'repo'
            work.mkdir()
            for name in ('DFM.py', 'NeuralNets.py', 'checkers.py', 'conditioning.py',
                         'guided_sampling.py', 'h_model.py', 'reference.py',
                         'sampling.py', 'synthetic_conditional_truth.py'):
                shutil.copy2(root / name, work / name)
            shutil.copytree(root / 'synthetic_benchmark', work / 'synthetic_benchmark',
                            ignore=shutil.ignore_patterns('__pycache__'))
            notebook = nbformat.read(notebook_path, as_version=4)
            # Use d=100, k=8 and reduced budgets in a temporary notebook. Keep
            # architecture, horizon, thresholds, seeds, and all execution paths.
            replacements = {
                'CONFIG': "replace(replace(Config.preset(MODE), **CONFIG_OVERRIDES), d=100, k=8, n_train=256, n_val=128, "
                          "n_eval=128, dfm_epochs=2, steps=64)",
            }
            replaced = set()
            gaussian_replaced = set()
            for i, cell in enumerate(notebook.cells):
                if cell.cell_type != 'code':
                    continue
                tree = ast.parse(cell.source)
                for node in tree.body:
                    if isinstance(node, ast.Assign) and len(node.targets) == 1:
                        target = node.targets[0]
                        if isinstance(target, ast.Name) and target.id in replacements:
                            node.value = ast.parse(replacements[target.id], mode='eval').body
                            replaced.add(target.id)
                for call in ast.walk(tree):
                    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                            and call.func.id == 'run_gaussian_conditional_simulation'):
                        continue
                    for keyword in call.keywords:
                        if keyword.arg in ('epochs', 'n_train', 'n_validation', 'n_test'):
                            keyword.value = ast.Constant(1 if keyword.arg == 'epochs' else 2048)
                            gaussian_replaced.add(keyword.arg)
                        if keyword.arg == 'assessment_settings':
                            for setting in keyword.value.keywords:
                                small = {'rejection_n': 64, 'rejection_batch_size': 128,
                                         'rejection_max_proposals': 1024,
                                         'rejection_reference_max_proposals': 1024, 'projections': 8}
                                if setting.arg in small:
                                    setting.value = ast.Constant(small[setting.arg])
                            gaussian_replaced.add(keyword.arg)
                cell.source = ast.unparse(ast.fix_missing_locations(tree))
                cell.outputs = []
                cell.execution_count = None
            self.assertEqual(replaced, set(replacements))
            self.assertEqual(gaussian_replaced, {'epochs', 'n_train', 'n_validation', 'n_test', 'assessment_settings'})
            copied_notebook = work / notebook_path.name
            nbformat.write(notebook, copied_notebook)
            self.assertFalse((work / 'artifacts').exists())
            self.assertFalse((work / 'data').exists())

            # Explicitly launch the interpreter running the tests, including in
            # a fresh venv; never fall back to a globally registered kernel.
            kernels = Path(tmp) / 'kernels'
            spec = kernels / 'synthetic-test'
            spec.mkdir(parents=True)
            (spec / 'kernel.json').write_text(json.dumps({
                'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                'display_name': 'Synthetic test', 'language': 'python',
            }))
            manager = KernelManager(kernel_name='synthetic-test',
                                    connection_file=str(Path(tmp) / 'kernel.json'),
                                    kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
            env = dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR=str(Path(tmp) / 'mpl'),
                       IPYTHONDIR=str(Path(tmp) / 'ipython'),
                       JUPYTER_RUNTIME_DIR=str(Path(tmp) / 'jupyter'))
            client = NotebookClient(notebook, km=manager, timeout=300,
                                    resources={'metadata': {'path': str(work)}})
            executed = client.execute(cwd=str(work), env=env, cleanup_kc=True)
            nbformat.write(executed, copied_notebook)
            code_cells = [c for c in executed.cells if c.cell_type == 'code']
            self.assertEqual([c.execution_count for c in code_cells],
                             list(range(1, len(code_cells) + 1)))
            self.assertFalse(any(o.output_type == 'error' for c in code_cells for o in c.outputs))
            runs = list((work / 'artifacts' / 'synthetic_unconditional_benchmark').iterdir())
            self.assertEqual(len(runs), 1)
            output = runs[0]
            for case in ('gaussian',):
                dfm_dir = output / f'{case}_seed42_primary'
                import pandas as pd
                from synthetic_conditional_truth import make_population
                metrics = pd.read_csv(dfm_dir / 'metrics.csv')
                with np.load(dfm_dir / 'unconditional_samples.npz') as samples:
                    pop = make_population(100, case, k=8)
                    for method, sample_key in (('DFM', 'generated'), ('truth-vs-truth', 'truth2')):
                        actual = metrics.loc[metrics.method == method,
                                             ['marginal_pit_ks_mean', 'marginal_pit_ks_std']].values[0]
                        np.testing.assert_allclose(actual, pop.evaluate_generated_samples(samples[sample_key]))
                self.assertTrue((dfm_dir / 'dfm.pt').is_file())
                self.assertTrue((dfm_dir / 'unconditional.png').is_file())
                checkpoint = torch.load(dfm_dir / 'dfm.pt', weights_only=True)
                self.assertEqual(checkpoint['population_k'], 8)
                self.assertEqual(checkpoint['config']['dfm_epochs'], 2)
                from synthetic_conditional_truth import population_details, population_metadata
                identity = population_metadata(pop, case)
                for key, value in identity.items():
                    self.assertEqual(checkpoint[key], value)
                saved_population = json.loads((dfm_dir / 'population.json').read_text())
                self.assertEqual(saved_population, population_details(pop, case))
                h_paths = list((output / f'h_{case}_seed42_tail0.05').glob('*/h.pt'))
                self.assertEqual(len(h_paths), 1)
                from synthetic_benchmark.conditional import load_trained_h, load_frozen_dfm
                _, h_metadata, _, restored_population = load_trained_h(h_paths[0].parent, 'cpu')
                for key, value in identity.items():
                    self.assertEqual(h_metadata[key], value)
                np.testing.assert_array_equal(restored_population.S, pop.S)
                # Real checkpoint loaders reject previous population identities.
                old_checkpoint = dict(checkpoint, population_design='diagonal-regime-factors-v2')
                old_dfm_path = work / 'old_dfm.pt'
                torch.save(old_checkpoint, old_dfm_path)
                with self.assertRaisesRegex(ValueError, 'Retrain the DFM and H'):
                    load_frozen_dfm(old_dfm_path, case, 'cpu')
                old_h_dir = work / 'old_h'
                old_h_dir.mkdir()
                from h_model import HNetwork
                old_h, old_metadata = HNetwork.load_checkpoint(h_paths[0])
                old_metadata['population_design'] = 'diagonal-regime-factors-v2'
                old_h.save_checkpoint(old_h_dir / 'h.pt', metadata=old_metadata)
                with self.assertRaisesRegex(ValueError, 'Retrain the DFM and H'):
                    load_trained_h(old_h_dir, 'cpu')
                self.assertTrue((h_paths[0].parent / 'validation_loss.png').is_file())
                sample_dirs = list(h_paths[0].parent.glob('samples_*'))
                self.assertEqual(len(sample_dirs), 1)
                self.assertTrue((sample_dirs[0] / 'conditional_comparison.png').is_file())
                self.assertTrue((sample_dirs[0] / 'conditional_portfolio_ecdf.png').is_file())
                self.assertTrue((sample_dirs[0] / 'conditional_metrics.csv').is_file())
                moment_metrics = pd.read_csv(sample_dirs[0] / 'conditional_moments.csv')
                self.assertEqual(set(moment_metrics.source), {
                    'guided', 'filtered', 'truth', 'truth_reference',
                    'rejection', 'rejection_reference', 'cdg_ml_guided', 'cdg_ml_filtered',
                })
                assessment = json.loads((sample_dirs[0] / 'sampling.json').read_text())
                self.assertEqual(assessment['assessment_config']['rejection_max_proposals'], 1024)
                self.assertIn('matching_rule', assessment)
                guided = assessment['guided_sampling']
                self.assertEqual(guided['seed'], assessment['guided_seed'])
                self.assertEqual(guided['steps'], assessment['steps'])
                self.assertFalse(guided['terminal_filtering'])
                initialization = guided['initialization']
                self.assertEqual(initialization['method'], 'h_rejection')
                self.assertEqual(initialization['returned'], 128)
                self.assertEqual(initialization['max_proposals'], 12800)
                self.assertLessEqual(initialization['proposals'], initialization['max_proposals'])
                self.assertGreaterEqual(initialization['proposals'], initialization['accepted'])
                self.assertGreaterEqual(initialization['accepted'], initialization['returned'])
                diagnostics = json.loads((sample_dirs[0] / 'rejection_diagnostics.json').read_text())
                self.assertEqual(set(diagnostics), {'rejection', 'rejection_reference'})
                conditional_metrics = pd.read_csv(sample_dirs[0] / 'conditional_metrics.csv')
                self.assertGreaterEqual(len(conditional_metrics), 7)
                self.assertTrue((conditional_metrics.n_used <= conditional_metrics.n_left).all())
                self.assertTrue((conditional_metrics.n_used <= conditional_metrics.n_right).all())
                with np.load(sample_dirs[0] / 'conditional_samples.npz') as samples:
                    self.assertEqual(samples['guided_samples'].shape, (128, 100))
                    self.assertTrue(np.isfinite(samples['guided_samples']).all())
                    np.testing.assert_array_equal(samples['filtered_guided_samples'],
                                                  samples['guided_samples'][samples['guided_event_mask']])
                    self.assertTrue(np.all(samples['true_conditional_samples'] @ samples['weights']
                                           <= samples['threshold'] + 1e-10))
                    for key, stream in (('rejection_samples', 'rejection'),
                                        ('independent_rejection_samples', 'rejection_reference')):
                        self.assertEqual(samples[key].shape[1], 100)
                        self.assertTrue(np.all(samples[key] @ samples['weights'] <= samples['threshold']))
                        self.assertEqual(len(samples[key]), diagnostics[stream]['retained_count'])
                        self.assertLessEqual(diagnostics[stream]['total_proposals'], 1024)
                    self.assertEqual(samples['independent_true_conditional_samples'].shape, (128, 100))
                    self.assertEqual(samples['projection_directions'].shape, (8, 100))
                    # Use the conditional Gaussian target, not unconditional moments.
                    conditional_mean, conditional_covariance = pop.moments(c=float(samples['threshold']))
                    scale = np.sqrt(np.trace(pop.moments()[1]))
                    for source, key in (('guided', 'guided_samples'), ('truth', 'true_conditional_samples')):
                        observed = moment_metrics.loc[moment_metrics.source == source].iloc[0]
                        self.assertEqual(observed.n, len(samples[key]))
                        expected = np.linalg.norm(samples[key].astype(float).mean(0) - conditional_mean) / scale
                        self.assertAlmostEqual(observed.mean_error, expected)
                        expected_covariance = (np.linalg.norm(np.cov(samples[key], rowvar=False) - conditional_covariance)
                                               / np.linalg.norm(conditional_covariance))
                        self.assertAlmostEqual(observed.covariance_error, expected_covariance)
                # CDG-ML is a separately trained baseline with all five refinements off.
                comparison = json.loads((output / 'h_comparison.json').read_text())
                self.assertTrue((output / 'h_comparison.png').is_file())
                self.assertTrue(comparison['shared_trajectory_splits'])
                self.assertTrue(comparison['shared_reference_samples'])
                raw_path = Path(comparison['variants']['raw']['h_checkpoint'])
                refined_path = Path(comparison['variants']['refined']['h_checkpoint'])
                self.assertNotEqual(raw_path, refined_path)
                # CDG-ML and Refined CDG-ML validation curves, then the single performance figure.
                figure_paths = [raw_path.parent / 'validation_loss.png',
                                refined_path.parent / 'validation_loss.png',
                                (output / 'h_comparison.png').resolve()]
                conditional_cell = next(c for c in code_cells
                                        if 'gaussian_results = run_gaussian_conditional_simulation(' in c.source)
                figure_outputs = [o.data['image/png'] for o in conditional_cell.outputs
                                  if o.output_type in ('display_data', 'execute_result')
                                  and 'image/png' in o.data]
                # Both H losses precede the single performance figure,
                # including when the notebook uses the Agg backend.
                self.assertEqual(len(figure_outputs), len(figure_paths))
                for encoded, path in zip(figure_outputs, figure_paths):
                    self.assertEqual(base64.b64decode(encoded), path.read_bytes())
                raw_model, raw_metadata = HNetwork.load_checkpoint(raw_path)
                refined_model, refined_metadata = HNetwork.load_checkpoint(refined_path)
                self.assertFalse(raw_model.log_remaining_time)
                self.assertTrue(refined_model.log_remaining_time)
                self.assertEqual(raw_model.normalization_times.numel(), 0)
                self.assertGreater(refined_model.normalization_times.numel(), 0)
                self.assertEqual(raw_metadata['training_options'], dict(
                    normalize_inputs=False, loss='brier', late_time_fraction=0.,
                    last_step_fraction=0., use_last_step_targets=False, times_per_trajectory=None,
                ))
                self.assertEqual(raw_metadata['reference_checkpoint'], refined_metadata['reference_checkpoint'])
                self.assertEqual(raw_metadata['stream_seeds'], refined_metadata['stream_seeds'])
                for name, tensor in raw_model.reference.state_dict().items():
                    torch.testing.assert_close(tensor, refined_model.reference.state_dict()[name], rtol=0, atol=0)
                raw_samples_path = Path(comparison['variants']['raw']['samples']) / 'conditional_samples.npz'
                with np.load(raw_samples_path) as raw, np.load(sample_dirs[0] / 'conditional_samples.npz') as refined:
                    self.assertEqual(raw['guided_samples'].shape, (128, 100))
                    np.testing.assert_array_equal(raw['filtered_guided_samples'],
                                                  raw['guided_samples'][raw['guided_event_mask']])
                    for key in ('true_conditional_samples', 'independent_true_conditional_samples',
                                'rejection_samples', 'independent_rejection_samples', 'projection_directions'):
                        np.testing.assert_array_equal(raw[key], refined[key])
                # Reopening a completed comparison preserves that display order
                # without training or drawing either H model again.
                from synthetic_benchmark.conditional import compare_saved_h_run
                with patch('synthetic_benchmark.conditional.display') as show, \
                     patch('synthetic_benchmark.conditional.fit_and_evaluate_h',
                           side_effect=AssertionError('cached comparison retrained H')), \
                     patch('synthetic_benchmark.conditional.sample_h_guided',
                           side_effect=AssertionError('cached comparison redrew H samples')):
                    cached = compare_saved_h_run(refined_path.parent, device='cpu')
                self.assertEqual([Path(call.args[0].filename) for call in show.call_args_list],
                                 figure_paths)
                self.assertEqual(cached['comparison_path'], (output / 'h_comparison.png').resolve())
            # Optionally copy the executed working directory for manual inspection.
            if os.environ.get('SYNTHETIC_VALIDATION_OUTPUT'):
                shutil.copytree(work, os.environ['SYNTHETIC_VALIDATION_OUTPUT'])
        self.assertEqual(notebook_path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
