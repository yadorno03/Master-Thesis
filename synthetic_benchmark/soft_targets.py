"""Provenance and cache identity for window-based H probability targets.

A construction record lists everything that determines the target arrays: the
frozen DFM checkpoint, event, grid, base split streams and batching, target
mode and resolved rows, and the continuation-branch streams. Its hash is the
cache identity, so a matching filename alone never revives stale arrays.
"""
import hashlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from h_model import TerminalLabelData, _validate_data, soft_target_indices

CONSTRUCTION_VERSION = 'h-soft-targets-v1'
ARRAY_FIELDS = tuple(f.name for f in fields(TerminalLabelData))


def file_sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def array_digest(array):
    """SHA-256 of a tensor/array's dtype, shape and raw bytes."""
    value = array.detach().cpu().contiguous().numpy() if isinstance(array, torch.Tensor) else np.ascontiguousarray(array)
    digest = hashlib.sha256(f'{value.dtype}|{value.shape}|'.encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def resolve_target_rows(model, steps, mode, window):
    """Rows and remaining times the target mode softens, from the original grid."""
    if mode == 'off':
        return [], []
    dtype = next(model.network.parameters()).dtype
    dt = model.horizon / steps
    times = torch.arange(steps, dtype=dtype) * dt
    rows = soft_target_indices(times, model.horizon, window)
    return rows.tolist(), [float(model.horizon - times[j].double()) for j in rows.tolist()]


def target_construction(model, *, mode, window, steps, split_counts, stream_seeds, path_batch_size,
                        dfm_checkpoint, branch_count=None, branch_seeds=None, branch_batch_size=None):
    """JSON-compatible record of how targets are (re)constructed; no timings or results."""
    parameter = next(model.network.parameters())
    rows, remaining = resolve_target_rows(model, steps, mode, window)
    continuation = mode == 'continuation'
    return json.loads(json.dumps(dict(
        version=CONSTRUCTION_VERSION, mode=mode, window=window, original_steps=steps,
        step_size=model.horizon / steps, horizon=model.horizon, time_indices=rows,
        remaining_times=remaining,
        branch_count=branch_count if continuation else None,
        branch_seeds=dict(branch_seeds) if continuation else None,
        branch_batch_size=branch_batch_size if continuation else None,
        branch_final_step='analytic exact_last_step_probability; no Bernoulli draw',
        stream_seeds=dict(stream_seeds), split_counts=[[name, int(n)] for name, n in split_counts],
        path_batch_size=path_batch_size, dtype=str(parameter.dtype), device=str(parameter.device),
        event=dict(n_assets=len(model.config.w), threshold=float(model.C),
                   weights_sha256=array_digest(np.asarray(model.config.w, dtype=np.float64))),
        dfm=dict(path=str(dfm_checkpoint), sha256=file_sha256(dfm_checkpoint)),
    ), allow_nan=False))


def construction_identity(construction):
    """Hash of every input that determines the arrays (the DFM path is not an input)."""
    record = json.loads(json.dumps(construction))
    record['dfm'] = dict(sha256=record['dfm']['sha256'])
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def split_array_digests(data_by_split):
    """Per-split digests of every stored target/base array (the target-array identity)."""
    return {split: {name: array_digest(getattr(data, name)) for name in ARRAY_FIELDS
                    if getattr(data, name) is not None}
            for split, data in data_by_split.items()}


def save_target_cache(directory, construction, data_by_split):
    """Write splits and their identity; refuse to overwrite an existing cache."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    torch.save({split: {name: getattr(data, name).detach().cpu() for name in ARRAY_FIELDS
                        if getattr(data, name) is not None}
                for split, data in data_by_split.items()}, directory / 'splits.pt')
    identity = dict(identity=construction_identity(construction), construction=construction,
                    arrays=split_array_digests(data_by_split))
    (directory / 'identity.json').write_text(json.dumps(identity, indent=2, allow_nan=False))
    return identity


def load_target_cache(directory, construction, model):
    """Return cached splits only if the saved identity equals this construction.

    Compares every construction field (not just the hash), verifies the stored
    arrays against their digests, and validates them against the H model.
    """
    directory = Path(directory)
    saved = json.loads((directory / 'identity.json').read_text())
    expected = json.loads(json.dumps(construction))
    stale = [key for key in sorted(set(saved['construction']) | set(expected))
             if key != 'dfm' and saved['construction'].get(key) != expected.get(key)]
    if saved['construction'].get('dfm', {}).get('sha256') != expected['dfm']['sha256']:
        stale.append('dfm.sha256')
    if stale or saved['identity'] != construction_identity(construction):
        raise ValueError('Stale target cache; differs in: ' + ', '.join(stale or ['identity']))
    arrays = torch.load(directory / 'splits.pt', map_location=next(model.network.parameters()).device,
                        weights_only=True)
    data = {split: TerminalLabelData(**tensors) for split, tensors in arrays.items()}
    if split_array_digests(data) != saved['arrays']:
        raise ValueError('Cached target arrays do not match their saved digests')
    for split in data.values():
        _validate_data(model, split)
    return data
