"""Generic checkpoint save/load, shared by the VAE and DDPM/LDM training loops.

Ported from the ``save_checkpoint``/``load_checkpoint`` pair in
``ddpm_ffhq_cosine_ema_pipeline.ipynb`` (optional ``ema_model``/
``repa_module`` payload keys) and generalized slightly so the same
functions also cover the VAE's simpler ``checkpoint_last.pt``/
``checkpoint_best.pt`` format (which has no EMA/REPA concept). Every
checkpoint is self-describing: it always carries the run's ``config`` dict
alongside the weights, which is what lets the evaluation scripts print
"Saved training config: ..." before doing anything else.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

__all__ = ["save_checkpoint", "load_checkpoint"]


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer,
    epoch: int,
    history,
    config: dict,
    repa_module: Optional[nn.Module] = None,
    ema_model: Optional[nn.Module] = None,
    extra: Optional[dict] = None,
) -> None:
    payload = dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict() if optimizer is not None else None,
        epoch=epoch,
        history=history,
        config=config,
    )
    if repa_module is not None:
        payload["repa_projection"] = repa_module.projection.state_dict()
    if ema_model is not None:
        payload["ema_model"] = ema_model.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer=None,
    device=None,
    repa_module: Optional[nn.Module] = None,
    ema_model: Optional[nn.Module] = None,
) -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if repa_module is not None and "repa_projection" in ckpt:
        repa_module.projection.load_state_dict(ckpt["repa_projection"])
    if ema_model is not None and "ema_model" in ckpt:
        ema_model.load_state_dict(ckpt["ema_model"])
    return ckpt
