#!/usr/bin/env python3
"""Train a pixel-space DDPM on FFHQ from a YAML config.

Usage:
    python scripts/train_ddpm.py --config configs/ddpm/cosine_ema_logit_v2_100.yaml

Covers every pixel-DDPM experiment in this repository (linear+logit-normal,
+REPA, cosine+EMA 65/100 epoch, linear+EMA) purely through the config file
- see configs/ddpm/*.yaml. The training loop itself lives in
_diffusion_common.py, shared with train_ldm.py.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from ffhq_repo.config import load_config
from ffhq_repo.utils.paths import resolve_device
from _diffusion_common import run_training_loop


def get_x0(raw_batch, device):
    """Pixel-space DDPM: x0 is just the [-1, 1]-normalized image batch;
    REPA's target is the same tensor (its own target_features() rescales
    to [0, 1] and ImageNet-normalizes internally)."""
    x0 = raw_batch.to(device, non_blocking=True)
    return x0, x0


def decode_fn(x):
    """Pixel-space: no VAE decode needed - just undo the [-1, 1] DDPM convention."""
    return (x.clamp(-1, 1) + 1) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a configs/ddpm/*.yaml file")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device)
    print("Using device:", device)

    run_training_loop(
        cfg, device, experiment_name_suffix="",
        in_channels=3, img_size=cfg.data.img_size,
        get_x0=get_x0, decode_fn=decode_fn,
        normalize_to_unit_range=True,  # pixel DDPM convention: [-1, 1]
    )


if __name__ == "__main__":
    main()
