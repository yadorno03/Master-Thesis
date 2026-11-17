"""Market benchmark: panel construction, tables against sample references, and a reduced end-to-end run."""
from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest

import matplotlib

matplotlib.use('Agg')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from market_benchmark.conditional import (  # noqa: E402
    conditional_folder, event_threshold, run_market_conditional, run_market_rejection_comparison,
    show_market_conditional, show_market_rejection_comparison,
)
from market_benchmark.data import Panel, build_panel, load_or_build_panel, split_panel  # noqa: E402
from market_benchmark.reporting import (  # noqa: E402
    DFM_DISTANCE_LABELS, EVENT_DISTANCE_LABELS, MARKET_EVENT_LABEL, bootstrap_noise_levels, leading_direction,
    market_dfm_tables, noise_rows,
)
from market_benchmark.unconditional import (  # noqa: E402
    load_market_dfm, run_market_unconditional, show_market_unconditional,
)
from synthetic_benchmark.configuration import notebook_config, rejection_assessment  # noqa: E402
from synthetic_benchmark.reporting import (  # noqa: E402
    CONDITIONAL_VALUE_ROWS, DFM_TAIL_ROWS, DFM_VALUE_ROWS, MODELS, REJECTION_REFERENCE, REJECTION_VALUE_ROWS,
    SMC_ROWS, conditioning_error_table, format_return, random_directions, readable_conditioning_error,
)


def factor_panel(days=700, d=12, seed=5):
    """Heavy-tailed one-factor returns standing in for market data."""
    rng = np.random.default_rng(seed)
    market = .01 * rng.standard_t(4, size=(days, 1))
    returns = market * rng.uniform(.5, 1.5, d) + .01 * rng.standard_t(5, size=(days, d))
    dates = pd.bdate_range('2020-01-01', periods=days).strftime('%Y-%m-%d')
    return Panel(returns, dates, tuple(f'T{i:02d}' for i in range(d)))


class MarketDataTests(unittest.TestCase):
    def test_panel_keeps_complete_liquid_histories_by_index_then_ticker(self):
        rng = np.random.default_rng(1)
        sessions = pd.bdate_range('2021-01-01', periods=41)
        prices = pd.DataFrame(100 * np.exp(np.cumsum(.01 * rng.normal(size=(41, 6)), axis=0)), index=sessions,
                              columns=['ZZ', 'BB', 'AA', 'STALE', 'SHORT', 'OTHER'])
        prices.loc[sessions[10:30], 'STALE'] = prices.loc[sessions[10], 'STALE']
        prices.loc[sessions[15:18], 'SHORT'] = np.nan
        universe = pd.DataFrame(dict(ticker=['ZZ', 'BB', 'AA', 'STALE', 'SHORT'], company=list('abcde'),
                                     source_index=['S&P 500', 'S&P 400', 'S&P 400', 'S&P 500', 'S&P 500']))
        with tempfile.TemporaryDirectory() as directory:
            prices_path, universe_path = Path(directory) / 'prices.csv.gz', Path(directory) / 'universe.csv'
            prices.to_csv(prices_path)
            universe.to_csv(universe_path, index=False)
            panel = build_panel(prices_path, universe_path, d=3, days=30)
            # S&P 500 members first, then alphabetical; stale, short and unlisted tickers are left out.
            self.assertEqual(panel.tickers, ('ZZ', 'AA', 'BB'))
            self.assertEqual((panel.returns.shape, panel.dates[-1]), ((30, 3), sessions[-1].strftime('%Y-%m-%d')))
            np.testing.assert_allclose(panel.returns[:, 0], prices['ZZ'].pct_change().to_numpy()[-30:])
            np.testing.assert_allclose(panel.weights, np.full(3, 1 / 3))
            with self.assertRaisesRegex(ValueError, 'complete history'):
                build_panel(prices_path, universe_path, d=4, days=30)
            with self.assertRaisesRegex(ValueError, 'sessions'):
                build_panel(prices_path, universe_path, d=2, days=41)
            saved = Path(directory) / 'processed' / 'panel.npz'
            built = load_or_build_panel(saved, prices=prices_path, universe=universe_path, d=3, days=30)
            self.assertEqual(load_or_build_panel(saved).sha256(), built.sha256())
            self.assertEqual(Panel.load(saved).tickers, panel.tickers)
        self.assertEqual(panel.first_assets(2).tickers, ('ZZ', 'AA'))
        with self.assertRaisesRegex(ValueError, 'finite'):
            Panel(np.array([[np.nan]]), ['2021-01-01'], ('A',))

    def test_split_uses_the_last_days_without_overlap(self):
        panel = factor_panel(days=100, d=4)
        chronological = split_panel(panel, 50, 20, 20, split='chronological')
        np.testing.assert_array_equal(chronological.train, panel.returns[10:60])
        np.testing.assert_array_equal(chronological.evaluation, panel.returns[80:])
        np.testing.assert_array_equal(chronological.insample, panel.returns[10:80])
        self.assertEqual((chronological.info['first_day'], chronological.info['evaluation_first_day']),
                         (panel.dates[10], panel.dates[80]))
        random = split_panel(panel, 50, 20, 20, seed=3)
        rows = np.concatenate((random.train, random.validation, random.evaluation))
        self.assertEqual(len({row.tobytes() for row in rows}), 90)
        self.assertEqual({row.tobytes() for row in rows}, {row.tobytes() for row in panel.returns[10:]})
        np.testing.assert_array_equal(split_panel(panel, 50, 20, 20, seed=3).train, random.train)
        self.assertFalse(np.array_equal(split_panel(panel, 50, 20, 20, seed=4).train, random.train))
        self.assertEqual(random.info['panel_sha256'], panel.sha256())
        for kwargs in (dict(split='blocked'), dict(seed=-1)):
            with self.assertRaises(ValueError):
                split_panel(panel, 50, 20, 20, **kwargs)
        with self.assertRaisesRegex(ValueError, 'fewer'):
            split_panel(panel, 80, 20, 20)


class MarketReportingTests(unittest.TestCase):
    def test_bootstrap_noise_levels(self):
        pool = factor_panel(days=600, d=6).returns
        weights, directions = np.full(6, 1 / 6), random_directions(1, 8, 6)
        small = bootstrap_noise_levels(pool, 50, weights, directions, replicates=40, seed=3)
        large = bootstrap_noise_levels(pool, 400, weights, directions, replicates=40, seed=3)
        self.assertEqual(set(small), set(DFM_DISTANCE_LABELS))
        # The sampling variation between two market samples shrinks as they grow.
        self.assertTrue(all(0 < large[key] < small[key] for key in small))
        self.assertEqual(bootstrap_noise_levels(pool, 50, weights, directions, replicates=40, seed=3), small)
        with self.assertRaisesRegex(ValueError, 'replicates'):
            bootstrap_noise_levels(pool, 50, weights, directions, replicates=0, seed=3)

    def test_unconditional_tables_and_leading_direction(self):
        panel = factor_panel(days=900, d=8)
        split = split_panel(panel, 400, 100, 400)
        direction = leading_direction(split.train, split.weights)
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.)
        self.assertGreater(float(direction @ split.weights), 0.)
        shifted = split.evaluation + .02
        # A DFM that returns the training days has the 5% and 1% event frequencies of the training days.
        values, distances = market_dfm_tables(
            dict(mlp=shifted, gaussian_mlp=split.evaluation, regime_mlp=split.train), split.evaluation,
            split.insample, split.weights, train=split.train, verdict='gaussian_mlp', projections=8, replicates=20)
        self.assertEqual((list(values.index), list(values.columns)), ([*DFM_VALUE_ROWS, *DFM_TAIL_ROWS], [
            'Hold-out market days', 'DFM (MLP)', 'DFM (Gaussian + MLP)', 'DFM (Regimes + MLP)']))
        self.assertEqual((list(distances.index), list(distances.columns)), (list(DFM_DISTANCE_LABELS.values()), [
            'DFM (MLP)', 'DFM (Gaussian + MLP)', 'DFM (Regimes + MLP)', 'Noise level (95%, n=400)',
            'DFM (Gaussian + MLP) within noise?']))
        self.assertEqual(noise_rows(distances), 400)
        # The hold-out days themselves are at distance zero, within the noise; a shifted copy is not.
        self.assertEqual([float(value) for value in distances['DFM (Gaussian + MLP)']], [0.] * 3)
        self.assertEqual(distances['DFM (Gaussian + MLP) within noise?'].tolist(), ['yes'] * 3)
        self.assertGreater(float(distances.loc[DFM_DISTANCE_LABELS['portfolio_ks'], 'DFM (MLP)']), .5)
        portfolio = split.evaluation @ split.weights
        self.assertEqual(values.loc[DFM_VALUE_ROWS[0], 'Hold-out market days'], format_return(portfolio.mean()))
        self.assertEqual(values.loc[DFM_VALUE_ROWS[0], 'DFM (MLP)'], format_return(portfolio.mean() + .02))
        self.assertEqual(values.loc[list(DFM_TAIL_ROWS[3:]), 'DFM (Regimes + MLP)'].tolist(), ['5.0%', '1.0%'])
        standardized = (split.train - split.train.mean(0)) / split.train.std(0, ddof=1)
        level = np.sqrt((standardized ** 2).mean(1))
        self.assertEqual(values.loc[DFM_TAIL_ROWS[2], 'DFM (Regimes + MLP)'], f'{level.std() / level.mean():.2f}')
        self.assertGreater(float(values.loc[DFM_TAIL_ROWS[0], 'Hold-out market days']), .5)
        with self.assertRaisesRegex(ValueError, 'verdict'):
            market_dfm_tables(dict(mlp=shifted), split.evaluation, split.insample, split.weights, train=split.train)


class MarketRunTests(unittest.TestCase):
    def test_reduced_run_trains_samples_and_reports_every_model(self):
        panel = factor_panel()
        config = replace(notebook_config('smoke'), n_train=400, n_val=100, n_eval=200, dfm_epochs=3, steps=24,
                         torch_threads=1)
        split = split_panel(panel, config.n_train, config.n_val, config.n_eval)
        tail = .2
        threshold = event_threshold(split, tail)
        self.assertAlmostEqual(float(np.mean(split.train @ split.weights <= threshold)), tail, delta=.01)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            unconditional = run_market_unconditional(config, split, output)
            for network in ('mlp', 'gaussian_mlp', 'regime_mlp'):
                with np.load(output / f'market_seed42_{network}' / 'unconditional_samples.npz') as saved:
                    self.assertEqual(saved['generated'].shape, (200, 12))
            self.assertEqual(list(unconditional['values'].index), [*DFM_VALUE_ROWS, *DFM_TAIL_ROWS])
            self.assertEqual(unconditional['distances'].columns[-1], 'DFM (Regimes + MLP) within noise?')
            regime_folder = output / 'market_seed42_regime_mlp'
            regimes = json.loads((regime_folder / 'regimes.json').read_text())
            self.assertEqual(np.shape(regimes['probabilities']), (4, 3))
            self.assertEqual(load_market_dfm(regime_folder)[1]._score.regimes, (4, 3))
            # The networks with an analytic factor part train their MLP only and are refit; DFM (MLP) is not.
            self.assertEqual(json.loads((output / 'network_settings.json').read_text()),
                             {'mlp': {}, **{network: dict(dfm_train_analytic=False, dfm_refit_knots=12)
                                            for network in ('gaussian_mlp', 'regime_mlp')}})
            self.assertEqual(load_market_dfm(regime_folder)[1]._score.refit_knots, 12)
            self.assertEqual(load_market_dfm(output / 'market_seed42_mlp')[1]._score.refit_knots, 0)
            self.assertIn('validation_factor', pd.read_csv(regime_folder / 'dfm_curve.csv').columns)
            checkpoint, reference = load_market_dfm(output / 'market_seed42_gaussian_mlp')
            self.assertEqual((checkpoint['case'], checkpoint['panel_sha256'], reference.d),
                             ('market', panel.sha256(), 12))
            self.assertEqual(json.loads((output / 'split.json').read_text())['split'], 'random')
            # The saved tables are displayed unchanged.
            for shown, computed in zip(show_market_unconditional(output),
                                       (unconditional['values'], unconditional['distances'])):
                pd.testing.assert_frame_equal(shown, computed)
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                run_market_unconditional(replace(config, n_train=300), split, Path(directory) / 'other')

            guided = replace(config, dfm_network='regime_mlp')
            recipe = dict(n_train=600, n_validation=300, n_test=300, epochs=1, validation_interval_steps=8,
                          n_samples=120, sample_batch_size=64,
                          assessment_settings=rejection_assessment(
                              tail, 60, rejection_batch_size=128, rejection_seed=42003,
                              rejection_reference_seed=42004, truth_reference_seed=42005, projection_seed=42006,
                              matching_seed=42007, projections=8, noise_seed=42009))
            result = run_market_conditional(guided, split, output, tail=tail, label='regime_mlp',
                                            plot_sources=('refined_exact', 'refined_smc'), **recipe)
            self.assertEqual(result['run_dir'], output / conditional_folder(tail, 'regime_mlp'))
            self.assertEqual((conditional_folder(.05), result['run_dir'].name),
                             ('conditional_tail0.05', 'conditional_regime_mlp_tail0.2'))
            samples = result['samples']
            self.assertEqual(set(samples), {*MODELS, 'rejection', 'rejection_reference', 'market',
                                            'market_reference'})
            for source in MODELS:
                self.assertEqual(samples[source].shape, (120, 12))
                self.assertTrue((samples[source] @ split.weights <= threshold + 1e-5).all(), source)
            # Market rows in the event: hold-out days, and training and validation days as their reference.
            event = split.evaluation @ split.weights <= threshold
            np.testing.assert_array_equal(samples['market'], split.evaluation[event])
            self.assertEqual(len(samples['market_reference']),
                             int((split.insample @ split.weights <= threshold).sum()))
            # The tables of the synthetic notebook, with the hold-out days in the event as reference.
            values, distances = result['values'], result['distances']
            self.assertEqual((list(values.index), list(values.columns)), (list(CONDITIONAL_VALUE_ROWS), [
                MARKET_EVENT_LABEL, 'DFM rejection', 'CDG-ML', 'Refined CDG-ML', 'Refined CDG-ML + SMC']))
            self.assertEqual(values.loc['Rows', MARKET_EVENT_LABEL], f'{event.sum():,}')
            self.assertEqual(values.loc[CONDITIONAL_VALUE_ROWS[-1], MARKET_EVENT_LABEL], f'{event.mean():.2%}')
            self.assertEqual(values.loc[CONDITIONAL_VALUE_ROWS[0], 'Refined CDG-ML + SMC'],
                             format_return((samples['refined_smc'] @ split.weights).mean()))
            n = min(int(event.sum()), 60)
            self.assertEqual((list(distances.index), list(distances.columns)), (list(EVENT_DISTANCE_LABELS.values()), [
                'DFM rejection', 'CDG-ML', 'Refined CDG-ML', 'Refined CDG-ML + SMC', f'Noise level (95%, n={n})',
                'Refined CDG-ML + SMC within noise?']))
            saved = pd.read_csv(result['run_dir'] / 'comparisons.csv')
            self.assertFalse(saved[['left', 'right']].isin(['truth', 'truth_reference']).any().any())
            self.assertEqual(list(result['smc'].index), list(SMC_ROWS))
            run = show_market_conditional(output, tail, 'regime_mlp')
            self.assertEqual((run['dfm_network'], run['models'], run['market_event_days']),
                             ('regime_mlp', list(MODELS), int(event.sum())))
            for name in ('conditional_distributions.png', 'values.csv', 'distances.csv', 'smc.csv',
                         'h_diagnostics.csv', 'conditional_samples.npz', 'refined/h.pt', 'cdg_ml/validation_loss.png'):
                self.assertTrue((result['run_dir'] / name).exists(), name)
            # The conditioning error of the synthetic notebook: every model against DFM rejection.
            error = conditioning_error_table({tail: saved})
            self.assertEqual(list(error.columns), [*distances.columns[1:4], REJECTION_REFERENCE, 'Rows compared'])
            note, readable = readable_conditioning_error(error)
            self.assertIn('60 matched rows', note)
            self.assertEqual(readable.shape, (3, 4))
            # DFM rejection of both frozen DFMs: the guided one reuses the rejection sample of its run, so its
            # distances and the noise level are those of the distance table.
            comparison = run_market_rejection_comparison(
                guided, split, output, tail=tail, networks=('gaussian_mlp', 'regime_mlp'), label='regime_mlp',
                assessment_settings=recipe['assessment_settings'])
            regime = 'DFM rejection, DFM (Regimes + MLP)'
            self.assertEqual((list(comparison.index), list(comparison.columns)), (
                [*REJECTION_VALUE_ROWS, *EVENT_DISTANCE_LABELS.values()],
                [MARKET_EVENT_LABEL, 'DFM rejection, DFM (Gaussian + MLP)', regime, f'Noise level (95%, n={n})']))
            self.assertEqual(comparison.loc[REJECTION_VALUE_ROWS[1], regime],
                             format_return((samples['rejection'] @ split.weights).mean()))
            self.assertEqual(comparison.loc[REJECTION_VALUE_ROWS[0], regime],
                             f"{run['sampling_diagnostics']['rejection']['observed_acceptance_fraction']:.2%}")
            rows = list(EVENT_DISTANCE_LABELS.values())
            self.assertEqual(comparison.loc[rows, regime].tolist(), distances.loc[rows, 'DFM rejection'].tolist())
            self.assertEqual(comparison.iloc[3:, -1].tolist(), distances.iloc[:, -2].tolist())
            pd.testing.assert_frame_equal(show_market_rejection_comparison(output, tail), comparison)
            # A DFM trained on another panel is refused, as are unknown plot sources.
            other = split_panel(factor_panel(seed=6), config.n_train, config.n_val, config.n_eval)
            with self.assertRaisesRegex(ValueError, 'different panel'):
                run_market_conditional(guided, other, output, tail=.1, **recipe)
            with self.assertRaisesRegex(ValueError, 'plot_sources'):
                run_market_conditional(guided, split, output, tail=.1, plot_sources=('rejection',), **recipe)


if __name__ == '__main__':
    unittest.main()
