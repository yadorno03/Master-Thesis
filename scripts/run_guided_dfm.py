"""Train or reuse H and compare guided draws with independent reference rejection."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone

import numpy as np

from guided_benchmark import benchmark_guidance
from h_model import HNetwork
from h_setup import train_h_from_checkpoint


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--h-checkpoint', type=Path, help='Reuse a fixed H/reference/event bundle')
    source.add_argument('--reference-checkpoint', type=Path, help='Train H on a frozen equity DFM')
    p.add_argument('--weights', type=Path, help='JSON mapping every asset to its raw-return weight')
    p.add_argument('--threshold', type=float, help='Fixed raw-return portfolio threshold')
    p.add_argument('--train-trajectories', type=int, default=4096)
    p.add_argument('--held-out-trajectories', type=int, default=1024)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden-dims', type=int, nargs='+', default=[64, 64])
    p.add_argument('--rollout-batch-size', type=int, default=32)
    p.add_argument('--output', type=Path, required=True, help='New run directory')
    p.add_argument('--samples', type=int, default=512)
    p.add_argument('--steps', type=int, default=128, help='Common H-label and generation grid')
    p.add_argument('--batch-size', type=int, default=128, help='Sampling and H optimization batch size')
    p.add_argument('--max-proposals', type=int, default=100000)
    p.add_argument('--timing-repeats', type=int, default=5)
    p.add_argument('--timing-reference-samples', type=int, default=2048)
    p.add_argument('--seed', type=int, default=74000)
    return p


def run(args):
    if args.h_checkpoint and (args.weights is not None or args.threshold is not None):
        raise ValueError('An H checkpoint fixes the event; do not override weights or threshold')
    if args.reference_checkpoint and (args.weights is None or args.threshold is None):
        raise ValueError('Reference training requires --weights and --threshold')
    source_hash = sha256(args.h_checkpoint or args.reference_checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
        'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'source_checkpoint_sha256': source_hash,
        'platform': platform.platform(), 'python': platform.python_version(),
        'versions': {name: importlib.metadata.version(name)
                     for name in ('torch', 'numpy', 'scipy', 'threadpoolctl')},
    }
    summary_path = args.output / 'run_summary.json'
    write_json(summary_path, summary)
    try:
        artifact_names = ['timing_benchmark.json', 'timing_samples.npz']
        if args.h_checkpoint:
            h, metadata = HNetwork.load_checkpoint(args.h_checkpoint, device='cpu')
        else:
            h, metadata = train_h_from_checkpoint(
                args.reference_checkpoint, weights=json.loads(args.weights.read_text()),
                threshold=args.threshold, num_steps=args.steps,
                train_trajectories=args.train_trajectories,
                held_out_trajectories=args.held_out_trajectories, epochs=args.epochs,
                batch_size=args.batch_size, rollout_batch_size=args.rollout_batch_size,
                hidden_dims=args.hidden_dims, lr=args.lr, seed=args.seed + 4)
            # H uses offsets 4..7; benchmark streams use 0..3 and 10 onward.
            h.save_checkpoint(args.output / 'h_checkpoint.pt', metadata=metadata)
            artifact_names.append('h_checkpoint.pt')
        setup_cost = metadata.get('h_setup_cost')
        summary.update(
            original_metadata=metadata, h_setup_cost=setup_cost,
            event_model={'weights': list(h.config.w), 'threshold': h.C,
                         'T': h.reference.T, 't0': h.reference.t0})
        timing, samples = benchmark_guidance(
            h, num_samples=args.samples, num_steps=args.steps,
            repeats=args.timing_repeats, validation_samples=args.timing_reference_samples,
            batch_size=args.batch_size, max_proposals=args.max_proposals, seed=args.seed,
            setup_seconds=setup_cost['total_seconds'] if setup_cost is not None else None)
        write_json(args.output / 'timing_benchmark.json', timing)
        np.savez_compressed(args.output / 'timing_samples.npz', **samples)
        summary['artifacts_sha256'] = {
            name: sha256(args.output / name)
            for name in artifact_names}
        summary.update(status='complete', completed_at_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        summary.update(status='failed', error=str(exc))
        write_json(summary_path, summary)
        raise
    write_json(summary_path, summary)
    print(f'Complete: {args.output}', flush=True)
    return summary


if __name__ == '__main__':
    run(parser().parse_args())
