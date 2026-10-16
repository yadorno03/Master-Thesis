"""Compare a saved H/reference/event bundle with independent reference rejection."""

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


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--h-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New run directory')
    p.add_argument('--samples', type=int, default=512)
    p.add_argument('--steps', type=int, default=128, help='Common generation grid')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--max-proposals', type=int, default=100000)
    p.add_argument('--timing-repeats', type=int, default=5)
    p.add_argument('--timing-reference-samples', type=int, default=2048)
    p.add_argument('--seed', type=int, default=74000)
    return p


def run(args):
    source_hash = sha256(args.h_checkpoint)
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
        h, metadata = HNetwork.load_checkpoint(args.h_checkpoint, device='cpu')
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
            for name in ('timing_benchmark.json', 'timing_samples.npz')}
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
