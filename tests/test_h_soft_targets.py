"""Window-based soft targets: reuse-final and unguided-continuation estimators."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from DFM import UnconditionalDFM
from conditioning import ConditioningConfig
from h_model import (
    HNetwork, _continuation_final_probability, _training_targets, _resolve_soft_rows,
    evaluate_h_objective, evaluate_h_scores, soft_target_indices, train_h, diagnose_h_terminal,
)
from reference import FrozenDFMReference
from sampling import generate_samples
import test_h_network


def generator(seed):
    return torch.Generator().manual_seed(seed)


class AffineGaussianH(HNetwork):
    """Zero network g, so the score is -D_t x: time-dependent affine Euler steps."""

    def __init__(self, steps):
        with torch.random.fork_rng():
            torch.manual_seed(5)
            dfm = UnconditionalDFM(3, 1, sigma_max=2., T=2.1, t0=.1, hidden_dims=[4]).double()
        with torch.no_grad():
            dfm.factor_score_network.c_raw.copy_(torch.tensor([-1.5, 0., 1.5]).double())
            for parameter in dfm.factor_score_network.g_zeta.parameters():
                parameter.zero_()
        super().__init__(FrozenDFMReference(dfm), ConditioningConfig((.6, -.3, .7), -.1))
        self.steps = steps

    def multiplier(self, j):
        """Per-asset Euler multiplier at absolute step j, read from the reference itself."""
        dt = self.horizon / self.steps
        ones = torch.ones(1, 3, dtype=torch.float64)
        s = self.reference.T - torch.tensor(j * dt, dtype=torch.float64)
        return 1 + dt * (.5 + self.reference.score(ones, s)[0])

    def analytic_probability(self, x, start, *, restart_at_zero=False):
        """P(w'X_J <= C | X_start = x) for the Gaussian Euler recursion (wrong if restarted)."""
        dt = self.horizon / self.steps
        phi = torch.ones(3, dtype=torch.float64)
        variance = torch.zeros(3, dtype=torch.float64)
        for position, j in enumerate(range(start, self.steps)):
            a = self.multiplier(position if restart_at_zero else j)
            phi, variance = phi * a, variance * a.square() + dt
        mean = (x * phi) @ self.w
        sd = (variance * self.w.square()).sum().sqrt()
        return float(torch.special.ndtr((self.C - mean) / sd))


def fixture(n=6, steps=8, seed=113, **options):
    model = test_h_network.HNetworkTests().fixture()
    return model, model.sample_trajectories(n, steps, generator=generator(seed), **options)


def continuation_data(window_rows=3, branches=3, n=6, steps=8, seed=113, branch_seed=7, **options):
    model = test_h_network.HNetworkTests().fixture()
    window = (window_rows + .2) * model.horizon / steps
    data = model.sample_trajectories(
        n, steps, generator=generator(seed), soft_target_mode='continuation',
        soft_target_window=window, branch_count=branches, branch_generator=generator(branch_seed),
        **options)
    return model, data, window


class WindowTests(unittest.TestCase):
    def test_window_is_inclusive_time_based_and_always_keeps_last_row(self):
        times = torch.arange(10, dtype=torch.float64) * .5       # horizon 5; remaining 5 ... .5
        self.assertEqual(soft_target_indices(times, 5., 1.5).tolist(), [7, 8, 9])
        self.assertEqual(soft_target_indices(times, 5., .1).tolist(), [9])
        self.assertEqual(soft_target_indices(times, 5., 100.).tolist(), list(range(10)))
        self.assertEqual(soft_target_indices(times.float(), 5., 1.5).tolist(), [7, 8, 9])
        tau, steps = 7.995, 400
        grid = torch.arange(steps, dtype=torch.float32) * (tau / steps)
        self.assertEqual(len(soft_target_indices(grid, tau, 5 * tau / steps)), 5)
        self.assertEqual(len(soft_target_indices(grid, tau, .1)), 5)
        self.assertEqual(len(soft_target_indices(grid, tau, .5)), 25)

    def test_option_contradictions_rejected_before_side_effects(self):
        model, data = fixture()
        rng = generator(1)
        state = rng.get_state().clone()
        bad = [dict(soft_target_mode='reuse_final', use_last_step_targets=True),
               dict(soft_target_mode='reuse_final', soft_target_window=0., use_last_step_targets=True),
               dict(soft_target_mode='reuse_final', soft_target_window=-1., use_last_step_targets=True),
               dict(soft_target_mode='reuse_final', soft_target_window=float('nan'),
                    use_last_step_targets=True),
               dict(soft_target_mode='reuse_final', soft_target_window=True, use_last_step_targets=True),
               dict(soft_target_mode='continuation', soft_target_window=.5),
               dict(soft_target_mode='reuse_final', soft_target_window=.5, use_last_step_targets=False),
               dict(soft_target_mode='off', soft_target_window=.5, use_last_step_targets=True),
               dict(soft_target_mode='bogus', soft_target_window=.5, use_last_step_targets=True)]
        with patch('h_model.torch.optim.Adam') as optimizer:
            for options in bad:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    train_h(model, data, epochs=1, generator=rng, **options)
            optimizer.assert_not_called()
        self.assertTrue(torch.equal(rng.get_state(), state))
        base = dict(soft_target_mode='continuation', soft_target_window=.5)
        with patch('h_model.generate_samples') as sampler:
            for options in (base, dict(base, branch_count=2),
                            dict(base, branch_count=0, branch_generator=generator(2)),
                            dict(base, branch_count=2, branch_generator=generator(2), branch_batch_size=0),
                            dict(base, soft_target_window=0., branch_count=2, branch_generator=generator(2)),
                            {'branch_count': 2},
                            dict(soft_target_mode='reuse_final', soft_target_window=.5,
                                 branch_generator=generator(2))):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    model.sample_trajectories(4, 3, generator=rng, **options)
            sampler.assert_not_called()
        self.assertTrue(torch.equal(rng.get_state(), state))


class TargetSelectionTests(unittest.TestCase):
    def test_legacy_off_targets_and_defaults_unchanged(self):
        model, data = fixture()
        indices = torch.tensor([4, 0, 2])
        binary = data.labels[indices, None].expand(-1, 8, -1)
        torch.testing.assert_close(_training_targets(data, indices, False), binary)
        expected = binary.clone()
        expected[:, -1] = data.last_probabilities[indices]
        torch.testing.assert_close(_training_targets(data, indices, True), expected)
        torch.testing.assert_close(_training_targets(data, indices, True, 'off', None), expected)
        for name in ('soft_target_indices', 'soft_targets', 'soft_target_standard_errors'):
            self.assertIsNone(getattr(data, name))
        outcomes = []
        for options in ({}, {'soft_target_mode': 'off'}):
            fresh, same = fixture()
            rng = generator(11)
            history = train_h(fresh, same, epochs=3, batch_size=2, generator=rng, **options)
            outcomes.append((history, rng.get_state(),
                             [p.detach().clone() for p in fresh.network.parameters()]))
        self.assertEqual(outcomes[0][0], outcomes[1][0])
        self.assertTrue(torch.equal(outcomes[0][1], outcomes[1][1]))
        for a, b in zip(outcomes[0][2], outcomes[1][2]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_reuse_final_uses_each_paths_own_probability_inside_window_only(self):
        model, data = fixture(steps=8)
        dt = model.horizon / 8
        rows = _resolve_soft_rows(model, data, 'reuse_final', 3.2 * dt)
        self.assertEqual(rows.tolist(), [5, 6, 7])
        indices = torch.tensor([3, 1, 5])
        targets = _training_targets(data, indices, True, 'reuse_final', rows)
        for position, path in enumerate(indices.tolist()):
            for j in range(8):
                expected = data.last_probabilities[path, 0] if j >= 5 else data.labels[path, 0]
                self.assertEqual(float(targets[position, j, 0]), float(expected))
        self.assertEqual(_resolve_soft_rows(model, data, 'reuse_final', 3 * dt).tolist(), [5, 6, 7])
        self.assertEqual(_resolve_soft_rows(model, data, 'reuse_final', 1e-9).tolist(), [7])
        # distinct paths receive distinct values: nothing is copied between paths
        self.assertGreater(float(data.last_probabilities.std()), 0)

    def test_full_time_and_sampled_time_use_the_same_target_for_every_path_and_time(self):
        for mode in ('reuse_final', 'continuation'):
            with self.subTest(mode=mode):
                model, data, window = continuation_data(window_rows=3)
                rows = _resolve_soft_rows(model, data, mode, window)
                indices = torch.tensor([4, 0, 5, 2])
                full = _training_targets(data, indices, True, mode, rows)
                js = torch.tensor([[7, 0, 5], [6, 6, 1], [4, 5, 7], [2, 3, 7]])
                gathered = full.gather(1, js[:, :, None])
                for position, path in enumerate(indices.tolist()):
                    for slot, j in enumerate(js[position].tolist()):
                        if j < 5:
                            expected = data.labels[path, 0]
                        elif mode == 'reuse_final':
                            expected = data.last_probabilities[path, 0]
                        else:
                            expected = data.soft_targets[path, j - 5, 0]
                        self.assertEqual(float(gathered[position, slot, 0]), float(expected))
                # the configured validation objective applies exactly this target rule
                model.eval()
                logits = model._logits(data.times[None].expand(6, -1).reshape(-1, 1),
                                       data.features.reshape(-1, 3)).reshape(6, 8, 1)
                weights = (data.weights * 0 + 1 / 8)
                target = _training_targets(data, torch.arange(6), True, mode, rows)
                by_hand = (torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, target, reduction='none') * weights[None, :, None]).sum() / 6
                actual = evaluate_h_objective(model, data, batch_size=4, loss='bce',
                                              use_last_step_targets=True, soft_target_mode=mode,
                                              soft_target_window=window)
                self.assertAlmostEqual(actual, float(by_hand), places=10)


class ContinuationTests(unittest.TestCase):
    def test_last_row_reuses_analytic_probability_and_consumes_no_branch_rng(self):
        model, data = fixture(n=5, steps=6)
        x = data.last_states
        branch = generator(3)
        state = branch.get_state().clone()
        value = _continuation_final_probability(model, x, 5, 6, generator=branch)
        torch.testing.assert_close(value, data.last_probabilities, rtol=0, atol=0)
        self.assertTrue(torch.equal(branch.get_state(), state))
        _, collected = fixture(n=5, steps=6, soft_target_mode='continuation',
                               soft_target_window=1e-9, branch_count=4, branch_generator=branch)
        self.assertTrue(torch.equal(branch.get_state(), state))
        self.assertEqual(collected.soft_target_indices.tolist(), [5])
        torch.testing.assert_close(collected.soft_targets[:, 0], collected.last_probabilities, rtol=0, atol=0)
        self.assertTrue((collected.soft_target_standard_errors == 0).all())

    def test_continuation_uses_absolute_times_and_original_step_size(self):
        model, _ = fixture(n=1, steps=4)
        steps, n = 7, 5
        dt = model.horizon / steps
        increments = dt ** .5 * torch.randn(steps, n, model.reference.d, generator=generator(21),
                                            dtype=torch.float64)
        initial = torch.randn(n, model.reference.d, generator=generator(22), dtype=torch.float64)
        paths = generate_samples(model.reference, n, num_steps=steps, initial_state=initial,
                                 brownian_increments=increments, return_trajectory=True)
        expected = model.exact_last_step_probability(paths[:, steps - 1], dt)
        for start in range(steps):
            actual = _continuation_final_probability(
                model, paths[:, start], start, steps, increments=increments[start:steps - 1])
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        # restarting at generative time zero (a generate_samples call from x) gives a different value
        x = paths[:, 2]
        for j in range(2, steps - 1):
            s = torch.full((n, 1), (j - 2) * dt, dtype=x.dtype)
            x = x + dt * (.5 * x + model.reference.score(x, model.reference.T - s)) + increments[j]
        restart = model.exact_last_step_probability(x, dt)
        self.assertGreater(float((restart - expected).abs().max()), 1e-6)

    def test_multistep_mean_matches_analytic_affine_gaussian_and_detects_restart(self):
        steps = 6
        model = AffineGaussianH(steps)
        x = torch.tensor([[2., -2., -2.]], dtype=torch.float64)
        branch = generator(31)
        for start in (1, 3, 4):
            estimates = _continuation_final_probability(model, x.expand(20000, -1), start, steps,
                                                        generator=branch)
            exact = model.analytic_probability(x[0], start)
            self.assertAlmostEqual(float(estimates.mean()), exact, delta=.01)
        wrong = model.analytic_probability(x[0], 3, restart_at_zero=True)
        self.assertGreater(abs(wrong - model.analytic_probability(x[0], 3)), .1)

    def test_averaging_preserves_mean_and_reduces_target_variance(self):
        steps = 6
        model = AffineGaussianH(steps)
        x = torch.tensor([[.4, -.7, 1.1]], dtype=torch.float64)
        groups = 3000
        rows = _continuation_final_probability(model, x.expand(groups * 16, -1), 2, steps,
                                               generator=generator(41))[:, 0]
        single = rows[:groups]
        for branches in (4, 16):
            averaged = rows[:groups * branches].reshape(groups, branches).mean(1)
            self.assertAlmostEqual(float(averaged.mean()), float(single.mean()), delta=.015)
            ratio = float(single.var() / averaged.var())
            self.assertGreater(ratio, branches * .7)
            self.assertLess(ratio, branches * 1.4)
        self.assertAlmostEqual(float(single.mean()), model.analytic_probability(x[0], 2), delta=.015)

    def test_standard_error_from_branch_probabilities(self):
        model, data, _ = continuation_data(window_rows=3, branches=8)
        self.assertEqual(data.soft_target_standard_errors.shape, data.soft_targets.shape)
        self.assertTrue((data.soft_target_standard_errors[:, :-1] > 0).all())
        self.assertTrue((data.soft_target_standard_errors[:, -1] == 0).all())
        _, single, _ = continuation_data(window_rows=3, branches=1)
        self.assertIsNone(single.soft_target_standard_errors)
        torch.testing.assert_close(single.soft_targets[:, -1], single.last_probabilities, rtol=0, atol=0)

    def test_branch_streams_leave_base_data_and_base_stream_unchanged(self):
        model = test_h_network.HNetworkTests().fixture()
        reference = model.sample_trajectories(7, 8, generator=generator(113), batch_size=3)
        outcomes = []
        for branches, branch_batch in ((2, None), (5, 4), (5, 3)):
            base = generator(113)
            window = 3.2 * model.horizon / 8
            data = model.sample_trajectories(
                7, 8, generator=base, batch_size=3, soft_target_mode='continuation',
                soft_target_window=window, branch_count=branches, branch_generator=generator(77),
                branch_batch_size=branch_batch)
            outcomes.append((data, base.get_state()))
            for name in ('features', 'labels', 'last_probabilities', 'last_states', 'times', 'weights'):
                torch.testing.assert_close(getattr(data, name), getattr(reference, name), rtol=0, atol=0)
            self.assertEqual(data.soft_targets.shape, (7, 3, 1))
        plain = generator(113)
        model.sample_trajectories(7, 8, generator=plain, batch_size=3)
        for _, state in outcomes:
            self.assertTrue(torch.equal(state, plain.get_state()))
        # same seeds and batching reproduce the targets; changing M changes them
        repeat = model.sample_trajectories(
            7, 8, generator=generator(113), batch_size=3, soft_target_mode='continuation',
            soft_target_window=3.2 * model.horizon / 8, branch_count=2, branch_generator=generator(77))
        torch.testing.assert_close(repeat.soft_targets, outcomes[0][0].soft_targets, rtol=0, atol=0)
        self.assertFalse(torch.equal(outcomes[0][0].soft_targets, outcomes[1][0].soft_targets))

    def test_chunks_are_assembled_in_parent_path_order_with_one_index_set(self):
        model = test_h_network.HNetworkTests().fixture()
        window = 3.2 * model.horizon / 8
        options = dict(soft_target_mode='continuation', soft_target_window=window, branch_count=3)
        chunked = model.sample_trajectories(5, 8, generator=generator(9), batch_size=2,
                                            branch_generator=generator(4), **options)
        pieces, base, branch = [], generator(9), generator(4)
        for count in (2, 2, 1):
            pieces.append(model.sample_trajectories(count, 8, generator=base,
                                                    branch_generator=branch, **options))
        for name in ('soft_targets', 'soft_target_standard_errors'):
            torch.testing.assert_close(getattr(chunked, name),
                                       torch.cat([getattr(p, name) for p in pieces]), rtol=0, atol=0)
        self.assertEqual(chunked.soft_target_indices.tolist(), [5, 6, 7])
        self.assertEqual(len(chunked.soft_targets), 5)

    def test_frozen_reference_modes_restored_after_success_and_failure(self):
        model, data, window = continuation_data()
        reference = model.reference
        before = {k: v.clone() for k, v in reference.state_dict().items()}
        reference._score.g_zeta[0].train()                 # mixed modes
        modes = [m.training for m in reference.modules()]
        _continuation_final_probability(model, data.last_states, 3, 8, generator=generator(1))
        self.assertEqual([m.training for m in reference.modules()], modes)
        with patch.object(HNetwork, 'exact_last_step_probability', side_effect=RuntimeError('boom')):
            with self.assertRaises(RuntimeError):
                _continuation_final_probability(model, data.last_states, 3, 8, generator=generator(1))
        self.assertEqual([m.training for m in reference.modules()], modes)
        model.sample_trajectories(3, 8, generator=generator(2), soft_target_mode='continuation',
                                  soft_target_window=window, branch_count=2, branch_generator=generator(3))
        self.assertEqual([m.training for m in reference.modules()], modes)
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in reference.parameters()))
        with self.assertRaises(ValueError):
            _continuation_final_probability(model, data.last_states, 8, 8)
        with self.assertRaises(ValueError):
            _continuation_final_probability(model, data.last_states, 3, 8,
                                            increments=torch.zeros(1, 6, 4, dtype=torch.float64))


class DataValidityTests(unittest.TestCase):
    def test_invalid_arrays_rejected_before_optimizer_or_sampling_side_effects(self):
        model, data, window = continuation_data()
        bad = {
            'indices dtype': replace(data, soft_target_indices=data.soft_target_indices.int()),
            'indices not last': replace(data, soft_target_indices=data.soft_target_indices - 1),
            'indices unsorted': replace(data, soft_target_indices=data.soft_target_indices.flip(0)),
            'indices duplicated': replace(data, soft_target_indices=data.soft_target_indices.clone().index_fill_(
                0, torch.tensor([0]), 6)),
            'indices matrix': replace(data, soft_target_indices=data.soft_target_indices[None]),
            'missing indices': replace(data, soft_target_indices=None),
            'wrong window': replace(data, soft_target_indices=data.soft_target_indices[1:],
                                    soft_targets=data.soft_targets[:, 1:],
                                    soft_target_standard_errors=data.soft_target_standard_errors[:, 1:]),
            'shape': replace(data, soft_targets=data.soft_targets[:, :-1]),
            'dtype': replace(data, soft_targets=data.soft_targets.float()),
            'nan': replace(data, soft_targets=data.soft_targets * float('nan')),
            'range': replace(data, soft_targets=data.soft_targets + 1.5),
            'grad': replace(data, soft_targets=data.soft_targets.clone().requires_grad_()),
            'final row': replace(data, soft_targets=torch.cat(
                (data.soft_targets[:, :-1], data.soft_targets[:, -1:] * .5), 1)),
            'se negative': replace(data, soft_target_standard_errors=-data.soft_target_standard_errors - 1),
            'se nan': replace(data, soft_target_standard_errors=data.soft_target_standard_errors * float('nan')),
            'se final nonzero': replace(data, soft_target_standard_errors=data.soft_target_standard_errors + .1),
            'se grad': replace(data, soft_target_standard_errors=data.soft_target_standard_errors.clone().requires_grad_()),
            'no last probabilities': replace(data, last_probabilities=None),
        }
        rng = generator(5)
        state = rng.get_state().clone()
        with patch('h_model.torch.optim.Adam') as optimizer:
            for name, invalid in bad.items():
                with self.subTest(name), self.assertRaises(ValueError):
                    train_h(model, invalid, epochs=1, generator=rng, loss='bce',
                            use_last_step_targets=True, soft_target_mode='continuation',
                            soft_target_window=window)
            # a bad validation split is rejected too, as is a window that disagrees with the data
            with self.assertRaises(ValueError):
                train_h(model, data, epochs=1, generator=rng, validation_data=bad['shape'], loss='bce',
                        use_last_step_targets=True, soft_target_mode='continuation',
                        soft_target_window=window)
            with self.assertRaisesRegex(ValueError, 'do not match'):
                train_h(model, data, epochs=1, generator=rng, loss='bce', use_last_step_targets=True,
                        soft_target_mode='continuation', soft_target_window=window * 2)
            with self.assertRaisesRegex(ValueError, 'continuation targets'):
                train_h(model, replace(data, soft_target_indices=None, soft_targets=None,
                                       soft_target_standard_errors=None), epochs=1, generator=rng,
                        loss='bce', use_last_step_targets=True, soft_target_mode='continuation',
                        soft_target_window=window)
            optimizer.assert_not_called()
        self.assertTrue(torch.equal(rng.get_state(), state))
        # reuse mode needs no extra arrays but still needs the final probabilities
        with self.assertRaises(ValueError):
            train_h(model, replace(data, last_probabilities=None, soft_target_indices=None,
                                   soft_targets=None, soft_target_standard_errors=None),
                    epochs=1, use_last_step_targets=True, soft_target_mode='reuse_final',
                    soft_target_window=window)

    def test_training_runs_in_all_modes_and_scores_are_separate(self):
        model, data, window = continuation_data(n=12)
        validation = model.sample_trajectories(
            8, 8, generator=generator(114), soft_target_mode='continuation', soft_target_window=window,
            branch_count=3, branch_generator=generator(8))
        for mode in ('off', 'reuse_final', 'continuation'):
            with self.subTest(mode=mode):
                fresh = test_h_network.HNetworkTests().fixture()
                options = dict(soft_target_mode=mode, soft_target_window=None if mode == 'off' else window)
                selection = {}
                history = train_h(fresh, data, epochs=3, batch_size=5, validation_data=validation,
                                  selection=selection, loss='bce', normalize_inputs=True,
                                  late_time_fraction=.4, last_step_fraction=.1, times_per_trajectory=4,
                                  use_last_step_targets=True, generator=generator(15), **options)
                self.assertEqual(len(history), 3)
                scores = evaluate_h_scores(fresh, validation, late_time_fraction=.4, last_step_fraction=.1,
                                           use_last_step_targets=True, **options)
                self.assertEqual(set(scores), {'binary_brier', 'binary_bce', 'binary_brier_weighted',
                                               'binary_bce_weighted', 'configured_brier', 'configured_bce'})
                self.assertAlmostEqual(
                    selection['best_loss'], min(selection['validation_history']))
                diagnostic = diagnose_h_terminal(fresh, validation, late_time_fraction=.4,
                                                 last_step_fraction=.1, **options)
                kinds = {row['target'] for row in diagnostic['late_rows']}
                self.assertIn('exact_euler_probability', kinds)
                if mode != 'off':
                    self.assertTrue(kinds <= {mode, 'terminal_label', 'exact_euler_probability'})
        # the configured score changes with the target; the binary score does not
        off = evaluate_h_scores(fresh, validation, use_last_step_targets=True)
        soft = evaluate_h_scores(fresh, validation, use_last_step_targets=True,
                                 soft_target_mode='continuation', soft_target_window=window)
        self.assertEqual(off['binary_brier'], soft['binary_brier'])
        self.assertNotEqual(off['configured_bce'], soft['configured_bce'])


if __name__ == '__main__':
    unittest.main()
