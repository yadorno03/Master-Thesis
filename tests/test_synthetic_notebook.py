"""Execute the actual notebook paths with tiny training budgets, never 1,000 epochs."""
import ast
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

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
        import pandas as pd
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
            # Use the Chen population at d=100, k=8 and reduced budgets in a temporary notebook.
            # Keep architecture, horizon, thresholds, seeds, and all execution paths.
            replacements = {
                'CONFIG': "replace(replace(Config.preset(MODE), **CONFIG_OVERRIDES), d=100, k=8, n_train=256, "
                          "n_val=128, n_eval=128, dfm_epochs=2, steps=64)",
                'REJECTION_ROWS': '64',
            }
            small = {'H_RECIPE': {'n_train': 2048, 'n_validation': 2048, 'n_test': 2048, 'epochs': 1,
                                  'validation_interval_steps': 8},
                     'ASSESSMENT': {'rejection_batch_size': 128, 'projections': 8}}
            replaced = set()
            for cell in notebook.cells:
                if cell.cell_type != 'code':
                    continue
                tree = ast.parse(cell.source)
                for node in tree.body:
                    if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                            and isinstance(node.targets[0], ast.Name)):
                        continue
                    name = node.targets[0].id
                    if name in replacements:
                        node.value = ast.parse(replacements[name], mode='eval').body
                        replaced.add(name)
                    elif name in small:
                        for keyword in node.value.keywords:
                            if keyword.arg in small[name]:
                                keyword.value = ast.Constant(small[name][keyword.arg])
                                replaced.add(f'{name}.{keyword.arg}')
                cell.source = ast.unparse(ast.fix_missing_locations(tree))
                cell.outputs = []
                cell.execution_count = None
            self.assertEqual(replaced, set(replacements) | {f'{name}.{key}' for name, keys in small.items()
                                                            for key in keys})
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
            client = NotebookClient(notebook, km=manager, timeout=600,
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

            from synthetic_conditional_truth import (
                make_population, population_details, population_metadata,
            )
            from synthetic_benchmark.reporting import (
                DFM_LABELS, EVENT_TOLERANCE, SOURCE_LABELS, SOURCES, factor_variance_summary,
            )
            from NeuralNets import FactorScoreNetwork, GaussianMLPScoreNetwork
            case = 'gaussian'
            pop = make_population(100, case, k=8)
            identity = population_metadata(pop, case)
            summary_metrics = pd.read_csv(output / 'metrics.csv')
            self.assertEqual(sorted(summary_metrics.method), sorted([*DFM_LABELS.values(), 'truth-vs-truth']))
            for name in ('dfm_distributions.png', 'dfm_values.csv', 'dfm_distances.csv', 'conclusions.md'):
                self.assertTrue((output / name).is_file(), name)
            draws = {}
            for network, score_type in (('mlp', FactorScoreNetwork),
                                        ('gaussian_mlp', GaussianMLPScoreNetwork)):
                dfm_dir = output / f'{case}_seed42_{network}'
                metrics = pd.read_csv(dfm_dir / 'metrics.csv')
                self.assertEqual(list(metrics.method), [DFM_LABELS[network], 'truth-vs-truth'])
                with np.load(dfm_dir / 'unconditional_samples.npz') as samples:
                    draws[network] = {key: samples[key] for key in samples.files}
                for method, sample_key in ((DFM_LABELS[network], 'generated'), ('truth-vs-truth', 'truth2')):
                    row = metrics.loc[metrics.method == method].iloc[0]
                    np.testing.assert_allclose(
                        np.asarray(row[['marginal_pit_ks_mean', 'marginal_pit_ks_std']], dtype=float),
                                               pop.evaluate_generated_samples(draws[network][sample_key]))
                    for key, value in factor_variance_summary(draws[network][sample_key], pop).items():
                        self.assertAlmostEqual(row[key], value)
                self.assertTrue((dfm_dir / 'unconditional.png').is_file())
                checkpoint = torch.load(dfm_dir / 'dfm.pt', weights_only=True)
                self.assertEqual(checkpoint['config']['dfm_network'], network)
                self.assertEqual(checkpoint['config']['dfm_epochs'], 2)
                # The DFM standardized the returns with the statistics of its training rows only.
                self.assertEqual(checkpoint['config']['dfm_standardization'], 'per_asset')
                from synthetic_benchmark.configuration import Streams
                train = pop.sample(256, Streams(42).np('train'))
                np.testing.assert_allclose(checkpoint['state']['standardization.mean'].numpy(), train.mean(0),
                                           rtol=0, atol=1e-6)
                np.testing.assert_allclose(checkpoint['state']['standardization.scale'].numpy(),
                                           train.std(0, ddof=1), rtol=1e-6)
                self.assertEqual('factor_score_network.prior_chol' in checkpoint['state'],
                                 network == 'gaussian_mlp')
                for key, value in identity.items():
                    self.assertEqual(checkpoint[key], value)
                self.assertEqual(json.loads((dfm_dir / 'population.json').read_text()),
                                 population_details(pop, case))
            # Both networks were trained and evaluated on the same draws.
            for key in ('truth', 'truth2'):
                np.testing.assert_array_equal(draws['mlp'][key], draws['gaussian_mlp'][key])

            from synthetic_benchmark.conditional import load_trained_h, load_frozen_dfm
            from h_model import HNetwork
            gaussian_mlp_checkpoint_path = output / f'{case}_seed42_gaussian_mlp' / 'dfm.pt'
            _, _, _, _, frozen = load_frozen_dfm(gaussian_mlp_checkpoint_path, case, 'cpu')
            self.assertIs(type(frozen._score), GaussianMLPScoreNetwork)
            # Real checkpoint loaders reject previous population identities.
            checkpoint = torch.load(gaussian_mlp_checkpoint_path, weights_only=True)
            old_dfm_path = work / 'old_dfm.pt'
            torch.save(dict(checkpoint, population_design='diagonal-regime-factors-v2'), old_dfm_path)
            with self.assertRaisesRegex(ValueError, 'Retrain the DFM and H'):
                load_frozen_dfm(old_dfm_path, case, 'cpu')

            models = ('refined_exact', 'cdg_ml_exact')
            # The final table reads both H models against DFM rejection back from the saved comparisons.
            conditioning_error = pd.read_csv(output / 'conditioning_error.csv', index_col=[0, 1])
            self.assertEqual(list(conditioning_error.columns),
                             ['CDG-ML', 'Refined CDG-ML', 'Rejection-vs-rejection reference', 'Rows compared'])
            self.assertEqual(list(conditioning_error.index.unique(0)), ['5% event', '1% event'])
            for tail, variable in ((0.05, 'conditional_5'), (0.01, 'conditional_1')):
                run_dir = output / f'conditional_tail{tail}'
                h_paths = {key: list((run_dir / key / f'h_{case}_seed42_tail{tail}').glob('*/h.pt'))
                           for key in ('refined', 'cdg_ml')}
                self.assertEqual({key: len(paths) for key, paths in h_paths.items()}, {'refined': 1, 'cdg_ml': 1})
                refined_path, raw_path = h_paths['refined'][0], h_paths['cdg_ml'][0]
                h_model, h_metadata, _, restored_population = load_trained_h(refined_path.parent, 'cpu')
                self.assertIs(type(h_model.reference._score), GaussianMLPScoreNetwork)
                for key, value in identity.items():
                    self.assertEqual(h_metadata[key], value)
                np.testing.assert_array_equal(restored_population.S, pop.S)
                if tail == 0.05:
                    old_h_dir = work / 'old_h'
                    old_h_dir.mkdir()
                    old_h, old_metadata = HNetwork.load_checkpoint(refined_path)
                    old_metadata['population_design'] = 'diagonal-regime-factors-v2'
                    old_h.save_checkpoint(old_h_dir / 'h.pt', metadata=old_metadata)
                    with self.assertRaisesRegex(ValueError, 'Retrain the DFM and H'):
                        load_trained_h(old_h_dir, 'cpu')
                for name in ('conditional_samples.npz', 'comparisons.csv', 'moments.csv', 'summary.csv',
                             'h_diagnostics.csv', 'run.json', 'conditional_distributions.png',
                             'table_values.csv', 'table_distances.csv'):
                    self.assertTrue((run_dir / name).is_file(), name)
                run = json.loads((run_dir / 'run.json').read_text())
                cap = int(np.ceil(4 * 64 / tail))
                self.assertEqual(run['assessment_config']['rejection_max_proposals'], cap)
                self.assertEqual(run['dfm_network'], 'gaussian_mlp')
                self.assertEqual(run['dfm_standardization'], 'per_asset')
                self.assertEqual(run['plot_sources'], ['truth', 'rejection', 'refined_exact'])
                self.assertIn('matching_rule', run)
                self.assertEqual(run['h_recipe']['validation_interval_steps'], 8)
                # The last step's noise along the return portfolio: sqrt(dt) |scale * w|.
                scale = torch.load(gaussian_mlp_checkpoint_path, weights_only=True)['state']['standardization.scale']
                self.assertAlmostEqual(run['near_band'], 3 * np.sqrt((8. - .005) / 64)
                                       * np.linalg.norm(scale.double().numpy() * pop.w))
                # H holds the return event in the DFM's coordinates; the run records it in returns.
                self.assertAlmostEqual(run['threshold'], pop.threshold(tail))
                np.testing.assert_allclose(np.asarray(h_model.config.w), scale.double().numpy() * pop.w,
                                           rtol=0, atol=1e-15)
                for source in ('refined_exact', 'cdg_ml_exact'):
                    guided = run['sampling_diagnostics'][source]
                    self.assertEqual(guided['seed'], run['guided_seed'])
                    self.assertEqual(guided['final_step'], 'exact')
                    self.assertFalse(guided['terminal_filtering'])
                    initialization = guided['initialization']
                    self.assertEqual(initialization['method'], 'h_rejection')
                    self.assertEqual(initialization['returned'], 128)
                    self.assertEqual(initialization['max_proposals'], int(np.ceil(20 * 128 / tail)))
                    self.assertLessEqual(initialization['proposals'], initialization['max_proposals'])
                    self.assertGreaterEqual(initialization['accepted'], initialization['returned'])
                moment_metrics = pd.read_csv(run_dir / 'moments.csv')
                self.assertEqual(set(moment_metrics.source), {*SOURCES, 'truth_reference', 'rejection_reference'})
                comparisons = pd.read_csv(run_dir / 'comparisons.csv')
                self.assertTrue((comparisons.n_used <= comparisons.n_left).all())
                self.assertTrue((comparisons.n_used <= comparisons.n_right).all())
                self.assertTrue({(m, r) for m in models for r in ('rejection', 'truth')} | {('rejection', 'truth')}
                                <= set(zip(comparisons.left, comparisons.right)))
                event = conditioning_error.loc[f'{tail:.0%} event']
                n = int(comparisons[(comparisons.left == 'refined_exact')
                                    & (comparisons.right == 'rejection')].n_used.iloc[0])
                self.assertTrue((event['Rows compared'] == n).all())
                matched = comparisons.set_index(['left', 'right', 'target_n'])
                for left, right, column in (('cdg_ml_exact', 'rejection', 'CDG-ML'),
                                            ('refined_exact', 'rejection', 'Refined CDG-ML'),
                                            ('rejection', 'rejection_reference', 'Rejection-vs-rejection reference')):
                    np.testing.assert_allclose(
                        event[column].to_numpy(dtype=float),
                        matched.loc[(left, right, n), ['portfolio_ks', 'portfolio_w1', 'sliced_w1']].to_numpy(dtype=float))
                summary = pd.read_csv(run_dir / 'summary.csv', index_col=0)
                self.assertEqual(list(summary.columns), [SOURCE_LABELS[s] for s in SOURCES])
                # The readable tables show the plotted sources only; the CSVs above keep all four.
                self.assertEqual(list(pd.read_csv(run_dir / 'table_values.csv', index_col=0).columns),
                                 ['Exact', 'DFM rejection', 'Refined CDG-ML'])
                self.assertEqual(list(pd.read_csv(run_dir / 'table_distances.csv', index_col=0).columns),
                                 ['DFM rejection', 'Refined CDG-ML', 'Noise level (95%, n=64)',
                                  'Refined CDG-ML within noise?'])
                with np.load(run_dir / 'conditional_samples.npz') as samples:
                    w, threshold = samples['weights'], float(samples['threshold'])
                    self.assertAlmostEqual(threshold, pop.threshold(tail))
                    for source in (*models, 'truth', 'truth_reference'):
                        self.assertEqual(samples[source].shape, (128, 100))
                        self.assertTrue(np.isfinite(samples[source]).all())
                    # Exact final step: every sampled row satisfies the event, nothing is discarded.
                    for source in models:
                        self.assertTrue(np.all(samples[source] @ w <= threshold + EVENT_TOLERANCE), source)
                    self.assertTrue(np.all(samples['truth'] @ w <= threshold + 1e-10))
                    for source in ('rejection', 'rejection_reference'):
                        self.assertEqual(samples[source].shape[1], 100)
                        self.assertTrue(np.all(samples[source] @ w <= threshold))
                        self.assertEqual(len(samples[source]), run['sampling_diagnostics'][source]['retained_count'])
                        self.assertLessEqual(run['sampling_diagnostics'][source]['total_proposals'], cap)
                    self.assertEqual(samples['projection_directions'].shape, (8, 100))
                    # Use the conditional Gaussian target, not unconditional moments.
                    conditional_mean, conditional_covariance = pop.moments(c=threshold)
                    scale = np.sqrt(np.trace(pop.moments()[1]))
                    for source in ('refined_exact', 'truth'):
                        observed = moment_metrics.loc[moment_metrics.source == source].iloc[0]
                        values = samples[source].astype(float)
                        self.assertEqual(observed.n, len(values))
                        self.assertAlmostEqual(observed.mean_error,
                                               np.linalg.norm(values.mean(0) - conditional_mean) / scale)
                        self.assertAlmostEqual(observed.covariance_error,
                                               np.linalg.norm(np.cov(values, rowvar=False) - conditional_covariance)
                                               / np.linalg.norm(conditional_covariance))
                # Excluded methods retain their samples and diagnostics, but produce no graphs.
                self.assertFalse((raw_path.parent / 'validation_loss.png').exists())
                figure_paths = [refined_path.parent / 'validation_loss.png',
                                run_dir / 'conditional_distributions.png']
                conditional_cell = next(c for c in code_cells
                                        if f'{variable} = run_conditional_benchmark(' in c.source)
                figure_outputs = [o.data['image/png'] for o in conditional_cell.outputs
                                  if o.output_type in ('display_data', 'execute_result')
                                  and 'image/png' in o.data]
                self.assertEqual(len(figure_outputs), len(figure_paths))
                for encoded, path in zip(figure_outputs, figure_paths):
                    self.assertEqual(base64.b64decode(encoded), path.read_bytes())
                # CDG-ML is a separately trained H with all five refinements off.
                raw_model, raw_metadata = HNetwork.load_checkpoint(raw_path)
                refined_model, refined_metadata = HNetwork.load_checkpoint(refined_path)
                self.assertFalse(raw_model.log_remaining_time)
                self.assertTrue(refined_model.log_remaining_time)
                self.assertEqual(raw_model.normalization_times.numel(), 0)
                self.assertGreater(refined_model.normalization_times.numel(), 0)
                self.assertEqual(raw_metadata['training_options'], dict(
                    normalize_inputs=False, loss='brier', late_time_fraction=0.,
                    last_step_fraction=0., use_last_step_targets=False, times_per_trajectory=None,
                    validation_interval_steps=8,
                ))
                self.assertEqual(raw_metadata['reference_checkpoint'], refined_metadata['reference_checkpoint'])
                self.assertEqual(raw_metadata['stream_seeds'], refined_metadata['stream_seeds'])
                for name, tensor in raw_model.reference.state_dict().items():
                    torch.testing.assert_close(tensor, refined_model.reference.state_dict()[name], rtol=0, atol=0)
            # Optionally copy the executed working directory for manual inspection.
            if os.environ.get('SYNTHETIC_VALIDATION_OUTPUT'):
                shutil.copytree(work, os.environ['SYNTHETIC_VALIDATION_OUTPUT'])
        self.assertEqual(notebook_path.read_bytes(), original)

if __name__ == '__main__':
    unittest.main()
