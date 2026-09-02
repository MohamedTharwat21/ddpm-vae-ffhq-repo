"""Exponential Moving Average (EMA) of the DDPM U-Net weights.

Used only by the cosine+EMA (v1/65-epoch, v2/100-epoch) and linear+EMA pixel DDPM
experiments; the earlier linear+logit-normal and linear+logit-normal+REPA
experiments predate EMA and were trained without it (``use_ema=False``).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

__all__ = ["update_ema", "build_ema_model"]


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """In-place EMA update of ema_model's parameters toward model's current parameters.

    Buffers (none in this GroupNorm-based U-Net - it has no BatchNorm
    running-stat buffers) are copied directly, not EMA-averaged, which is
    the generically correct behavior for any architecture that does carry
    such buffers.
    """
    with torch.no_grad():
        ema_params = dict(ema_model.named_parameters())
        for name, p in model.named_parameters():
            ema_params[name].mul_(decay).add_(p.detach(), alpha=1 - decay)
        ema_buffers = dict(ema_model.named_buffers())
        for name, b in model.named_buffers():
            ema_buffers[name].copy_(b)


def build_ema_model(model: nn.Module, device) -> nn.Module:
    """Construct a frozen, eval-mode copy of ``model`` to serve as the EMA target.

    Uses ``copy.deepcopy`` so the EMA model is architecturally identical to
    ``model`` without re-reading the config, then loads the current
    (typically freshly-initialized) weights and freezes every parameter.
    """
    ema_model = copy.deepcopy(model).to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema_model.eval()
    return ema_model
