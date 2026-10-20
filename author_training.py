"""Train the numerical DFM with the pinned authors' Trainer and recipe."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset
from author_dfm import AuthorDFMConfig, author_module, build_author_dfm
from author_data import ReturnTransform
from author_checkpoint import AuthorModel


def train_author_dfm(returns, assets, output_dir, *, config=AuthorDFMConfig(), amp=True,
                     transform=None):
    """Train on observed returns only and persist the ordinary final model.

    Asset columns retain their supplied order. This mirrors upstream train.py,
    including standardization, drop-last batches, AdamW, scheduler, EMA updates
    and clipping. The final sampling entry point uses ordinary weights, not EMA.
    Explicit small configs/amp=False are useful for CPU checks, not replication.
    Pass transform= to reuse the exact training statistics used by another model.
    Float64 shared coordinates are cast once at the U-Net input boundary; the
    saved buffers retain their precision for inverse/event conversion.
    """
    if not isinstance(amp, bool):
        raise ValueError('amp must be boolean')
    if transform is None:
        transform = ReturnTransform.fit(returns, assets)
    elif not isinstance(transform, ReturnTransform):
        raise TypeError('Expected a fitted ReturnTransform')
    transform.check_assets(assets)
    # Copy buffers before moving the model, preserving caller device/ownership.
    transform = deepcopy(transform)
    if len(transform.assets) != config.d:
        raise ValueError('Return columns must equal the configured image area')
    standardized = transform(returns, assets=assets)
    if standardized.ndim != 2:
        raise ValueError('Training returns must have shape [N,d]')
    data = standardized.reshape(-1, 1, *config.image_size).detach().to(device='cpu', dtype=torch.float32)
    if not torch.isfinite(data).all():
        raise ValueError('Standardized training data must remain finite in float32')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    diffusion = build_author_dfm(config)
    trainer = author_module().Trainer(diffusion, TensorDataset(data),
        train_batch_size=min(config.batch_size, len(data)), train_lr=1e-4,
        train_epochs=config.epochs, adamw_weight_decay=.01, cosine_scheduler=True,
        warm_up=True, warmup_iters=20, T_0=400, T_mult=1, eta_min=1e-5,
        cosine_steps=400, gradient_accumulate_every=1, ema_decay=.999,
        split_batches=False, save_and_sample_every=1000, results_folder=output_dir,
        param_path='', amp=amp, save_timesteps=[config.save_index])
    trainer.train()
    diffusion = trainer.accelerator.unwrap_model(trainer.model)
    device = next(diffusion.parameters()).device
    model = AuthorModel(diffusion, transform.to(device), config).eval()
    metadata = dict(amp=amp, training_observations=len(data),
        batch_size=min(config.batch_size, len(data)), trained_epochs=config.epochs,
        optimizer_updates=trainer.step, selected_weights='ordinary final weights',
        coordinate_order='input columns, unchanged', checkpoint_selection_observations=0,
        preprocessing_dtype=str(transform.mean.dtype))
    model.save_checkpoint(output_dir / 'dfm.pt', metadata=metadata)
    (output_dir / 'training.json').write_text(json.dumps(
        dict(**config.metadata(), **metadata), indent=2) + '\n')
    return model
