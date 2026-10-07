"""Model loading, checkpoint management, and batch collation utilities."""

from pathlib import Path
from typing import Dict, List, Any, Union
import torch
import torch.nn as nn
import numpy as np


def load_model_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    device: Union[str, torch.device] = 'cpu'
) -> nn.Module:
    """Loads PyTorch checkpoint weights into model, cleans _orig_mod prefixes, and sets eval mode."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    raw_ckpt = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    clean_ckpt = {k.replace('_orig_mod.', ''): v for k, v in raw_ckpt.items()}
    model.load_state_dict(clean_ckpt, strict=True)
    model.to(device)
    model.eval()
    return model


def collate_multimet_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Standard PyTorch DataLoader collate function for Multimet dataset samples."""
    batch = {}
    for k in samples[0].keys():
        if k == 'date':
            batch[k] = [s[k] for s in samples]
        elif k == 'x_d':
            batch[k] = {
                var: torch.stack([torch.tensor(s[k][var], dtype=torch.float32) for s in samples])
                for var in samples[0][k].keys()
            }
        elif isinstance(samples[0][k], np.ndarray):
            batch[k] = torch.stack([torch.tensor(s[k], dtype=torch.float32) for s in samples])
        elif isinstance(samples[0][k], torch.Tensor):
            batch[k] = torch.stack([s[k] for s in samples])
        else:
            batch[k] = [s[k] for s in samples]
    return batch
