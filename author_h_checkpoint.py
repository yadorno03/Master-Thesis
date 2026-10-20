"""Bundle H with its exact author DDPM, endpoint, event and preprocessing."""
from dataclasses import asdict
import json
from pathlib import Path
import torch
from author_checkpoint import AuthorModel
from author_reference import FrozenAuthorReference
from author_h import DDPMHNetwork
from conditioning import ConditioningConfig


def save_ddpm_h(h, path, *, metadata=None):
    if not isinstance(h, DDPMHNetwork):
        raise TypeError('Expected DDPMHNetwork')
    state = {k: v.detach().cpu().clone() for k,v in h.network.state_dict().items()}
    if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('H weights must be finite float32')
    expected_w = h.w.new_tensor(h.event.w)
    if not torch.equal(h.w, expected_w) or h.threshold != h.threshold.new_tensor(h.event.C):
        raise ValueError('H event buffers differ from the immutable event configuration')
    if any(p.requires_grad or p.grad is not None for p in h.reference.parameters()):
        raise ValueError('Reference must remain frozen')
    payload = dict(format='author-ddpm-h-v1', reference=h.reference.model.checkpoint_payload(),
        event=asdict(h.event), hidden_dims=h.hidden_dims, h_state=state,
        metadata=json.loads(json.dumps(metadata or {}, allow_nan=False)))
    with Path(path).open('xb') as handle:
        torch.save(payload, handle)


def load_ddpm_h(path, *, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('format') != 'author-ddpm-h-v1':
        raise ValueError('Unsupported DDPM H checkpoint format')
    if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in saved['h_state'].values()):
        raise ValueError('H checkpoint tensors must be finite float32')
    with torch.random.fork_rng(devices=[]):
        model = AuthorModel.from_payload(saved['reference'])
        reference = FrozenAuthorReference(model, take_ownership=True)
        h = DDPMHNetwork(reference, ConditioningConfig(**saved['event']), saved['hidden_dims'])
        h.network.load_state_dict(saved['h_state'], strict=True)
    return h.to(device).eval(), saved['metadata']
