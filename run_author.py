"""Train, sample and condition the pinned author DFM on observed return arrays."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from author_dfm import AuthorDFMConfig
from author_checkpoint import AuthorModel
from author_training import train_author_dfm
from author_reference import FrozenAuthorReference
from author_sampling import sample_author_ddpm
from author_h import DDPMHNetwork
from author_h_training import collect_ddpm_labels, train_ddpm_h, evaluate_ddpm_h
from author_h_checkpoint import save_ddpm_h, load_ddpm_h
from author_guidance import sample_guided_author


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train')
    train.add_argument('--returns', type=Path, required=True)
    train.add_argument('--assets', type=Path, required=True)
    for name, default in [('height',32), ('width',64), ('model-dim',256),
                          ('timesteps',200), ('save-index',20), ('epochs',600), ('batch-size',32)]:
        train.add_argument('--'+name, type=int, default=default)
    train.add_argument('--no-amp', action='store_true')
    sample = commands.add_parser('sample')
    sample.add_argument('--dfm', type=Path, required=True)
    train_h = commands.add_parser('train-h')
    train_h.add_argument('--dfm', type=Path, required=True)
    train_h.add_argument('--weights', type=Path, required=True)
    train_h.add_argument('--threshold', type=float, required=True)
    train_h.add_argument('--trajectories', type=int, default=1024)
    train_h.add_argument('--epochs', type=int, default=100)
    train_h.add_argument('--batch-size', type=int, default=32)
    train_h.add_argument('--lr', type=float, default=1e-3)
    guide = commands.add_parser('guide')
    guide.add_argument('--h', type=Path, required=True)
    guide.add_argument('--guidance-scale', type=float, default=1.)
    guide.add_argument('--initialization', choices=['h_rejection','gaussian'], default='h_rejection')
    guide.add_argument('--max-proposals', type=int, default=100000)
    for cmd in [sample, guide]:
        cmd.add_argument('--count', type=int, default=8192)
        cmd.add_argument('--batch-size', type=int, default=128)
    for cmd in [train, sample, train_h, guide]:
        cmd.add_argument('--seed', type=int, default=3407)
        cmd.add_argument('--output', type=Path, required=True)
    for cmd in [sample, train_h, guide]:
        cmd.add_argument('--device', choices=['cpu','cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError('Output or metadata already exists')
    if args.command == 'train':
        cfg = AuthorDFMConfig(image_size=(args.height,args.width), model_dim=args.model_dim,
            timesteps=args.timesteps, save_index=args.save_index, epochs=args.epochs,
            batch_size=args.batch_size, seed=args.seed)
        train_author_dfm(np.load(args.returns, allow_pickle=False), json.loads(args.assets.read_text()),
                         args.output, config=cfg, amp=not args.no_amp)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    if args.command == 'guide':
        h, _ = load_ddpm_h(args.h, device=args.device)
        result = sample_guided_author(h, args.count, guidance_scale=args.guidance_scale,
            initialization=args.initialization, max_proposals=args.max_proposals,
            batch_size=args.batch_size, generator=generator)
        model, standardized, info = h.reference.model, result.samples, result.diagnostics
    else:
        model, _ = AuthorModel.load_checkpoint(args.dfm, device=args.device)
        reference = FrozenAuthorReference(model, take_ownership=True)
        if args.command == 'train-h':
            event = model.transform.event(json.loads(args.weights.read_text()), args.threshold)
            h = DDPMHNetwork(reference, event)
            data = collect_ddpm_labels(h, args.trajectories, batch_size=args.batch_size, generator=generator)
            history = train_ddpm_h(h, data, epochs=args.epochs, batch_size=args.batch_size,
                                   lr=args.lr, generator=generator)
            save_ddpm_h(h, args.output, metadata=dict(seed=args.seed, loss_history=history,
                training_diagnostics=evaluate_ddpm_h(h, data), raw_weights=json.loads(args.weights.read_text()),
                raw_threshold=args.threshold, note='training diagnostics, not held-out evaluation'))
            return
        if args.count < 1 or args.batch_size < 1:
            raise ValueError('count and batch_size must be positive')
        standardized = torch.cat([sample_author_ddpm(reference, min(args.batch_size,args.count-start),
            generator=generator) for start in range(0,args.count,args.batch_size)])
        info = {'target':'unconditional author DDPM at saved endpoint'}
    raw = model.transform.inverse(standardized).detach().cpu().numpy()
    with args.output.open('xb') as handle:
        np.save(handle, raw, allow_pickle=False)
    metadata = model.config.metadata() | info | dict(seed=args.seed, samples=len(raw), assets=model.transform.assets)
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()
