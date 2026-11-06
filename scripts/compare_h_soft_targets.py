"""Target-only comparison of H probability targets on an existing frozen DFM.

Stages (all load the saved DFM; none retrains it or touches the source run):

  estimate  Time small probes and print the expected cost; trains nothing.
  pilot     2,048/512/512 H paths, 25 epochs. Finds runtime and integration
            problems only; it does not establish convergence.
  full      The original budgets: 10,000/2,000/2,000 paths, 200 epochs.

Examples (run from the repository root):

  SRC=artifacts/synthetic_unconditional_benchmark/standard_20261106T082150669095Z
  python3.12 scripts/compare_h_soft_targets.py --source-run $SRC --stage estimate
  python3.12 scripts/compare_h_soft_targets.py --source-run $SRC --stage pilot
  python3.12 scripts/compare_h_soft_targets.py --source-run $SRC --stage full \\
      --init-seeds 2027 2028 2029 --data-offsets 0

Three init seeds on fixed data measure optimization variability only; add
--data-offsets 0 1 2 for independent trajectory data. Results go to a fresh
artifacts/h_soft_target_ablation/<timestamp>/ directory.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from datetime import datetime, timezone

import matplotlib
matplotlib.use(__import__('os').environ.get('MPLBACKEND', 'Agg'))

from synthetic_benchmark.configuration import ConditionalAssessmentConfig
from synthetic_benchmark.reporting import plot_soft_target_comparison
from synthetic_benchmark.soft_target_experiment import (
    default_arms, load_soft_target_results, run_soft_target_experiment,
)

STAGES = {
    'pilot': dict(n_train=2048, n_validation=512, n_test=512, epochs=25, n_samples=500,
                  rejection_n=200, rejection_max=20_000, residual_states=32, residual_branches=64),
    'full': dict(n_train=10_000, n_validation=2_000, n_test=2_000, epochs=200, n_samples=3000,
                 rejection_n=1000, rejection_max=50_000, residual_states=64, residual_branches=128),
}
STAGES['estimate'] = STAGES['full']


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source-run', required=True, type=Path, help='saved unconditional run with the frozen DFM')
    parser.add_argument('--stage', choices=sorted(STAGES), default='pilot')
    parser.add_argument('--output', type=Path, default=None, help='fresh output directory')
    parser.add_argument('--source-h-run', type=Path, default=None,
                        help='saved H run whose settings to confirm (training.json)')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--tail', type=float, default=0.05)
    parser.add_argument('--widths', type=int, nargs='+', default=[64, 64])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--path-batch-size', type=int, default=256)
    parser.add_argument('--branch-count', type=int, default=8)
    parser.add_argument('--branch-batch-size', type=int, default=2048)
    parser.add_argument('--late-time-fraction', type=float, default=0.4)
    parser.add_argument('--last-step-fraction', type=float, default=0.1)
    parser.add_argument('--init-seeds', type=int, nargs='+', default=[2027])
    parser.add_argument('--data-offsets', type=int, nargs='+', default=[0])
    parser.add_argument('--arms', nargs='+', default=None,
                        help='subset of arm names (default: all four)')
    for name in ('n_train', 'n_validation', 'n_test', 'epochs', 'n_samples', 'rejection_n',
                 'rejection_max', 'residual_states', 'residual_branches'):
        parser.add_argument('--' + name.replace('_', '-'), type=int, default=None)
    parser.add_argument('--reuse-reference-from', type=Path, default=None,
                        help='saved samples_* directory with truth/rejection draws; verified before use')
    parser.add_argument('--target-cache', type=Path, default=None,
                        help='earlier data_offset*/target_cache directory; rejected if stale')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    stage = dict(STAGES[args.stage])
    for key in stage:
        override = getattr(args, key)
        if override is not None:
            stage[key] = override
    arms = default_arms(args.branch_count)
    if args.arms:
        unknown = set(args.arms) - set(arms)
        if unknown:
            raise SystemExit(f'Unknown arms {sorted(unknown)}; choose from {sorted(arms)}')
        arms = {name: arms[name] for name in args.arms}
    assessment = ConditionalAssessmentConfig(
        rejection_n=stage['rejection_n'], rejection_batch_size=256,
        rejection_max_proposals=stage['rejection_max'],
        rejection_reference_max_proposals=stage['rejection_max'],
        rejection_seed=62003, rejection_reference_seed=62004, truth_reference_seed=62005,
        projection_seed=62006, matching_seed=62007, projections=64)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output or ROOT / 'artifacts' / 'h_soft_target_ablation' / stamp
    result = run_soft_target_experiment(
        args.source_run, output, arms=arms, device=args.device, n_train=stage['n_train'],
        n_validation=stage['n_validation'], n_test=stage['n_test'], epochs=stage['epochs'],
        tail=args.tail, widths=tuple(args.widths), lr=args.lr, batch_size=args.batch_size,
        path_batch_size=args.path_batch_size, branch_batch_size=args.branch_batch_size,
        late_time_fraction=args.late_time_fraction, last_step_fraction=args.last_step_fraction,
        init_seeds=tuple(args.init_seeds), data_offsets=tuple(args.data_offsets),
        n_samples=stage['n_samples'], assessment_settings=assessment,
        reuse_reference_from=args.reuse_reference_from, target_cache=args.target_cache,
        residual_states=stage['residual_states'], residual_branches=stage['residual_branches'],
        source_h_run=args.source_h_run, estimate_only=args.stage == 'estimate')
    if args.stage == 'estimate':
        return
    loaded = load_soft_target_results(output)
    labels = {name: arm['label'] for name, arm in arms.items()}
    figure = plot_soft_target_comparison(
        loaded['portfolios'], loaded['truth'], loaded['rejection'], loaded['population'],
        loaded['threshold'], loaded['table'], loaded['residuals'], loaded['curves'], output,
        labels=labels, show=False)
    print('Figure:', figure)
    print(loaded['summary'].filter(regex='^(arm|n_runs|.*_mean)$').round(4).to_string())


if __name__ == '__main__':
    main()
