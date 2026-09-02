#!/usr/bin/env python3
"""Train a latent-space DDPM (LDM) on FFHQ, diffusing a frozen pretrained VAE's latents.

Usage:
    python scripts/train_ldm.py --config configs/ldm/linear.yaml

Loads and freezes the VAE checkpoint named by ``config.model.vae_checkpoint_path``
(or auto-detects one, see ffhq_repo.data.pretrained_vae), derives the DDPM
U-Net's latent input shape from it, and reuses the exact same training loop
as train_ddpm.py (_diffusion_common.py) - only ``get_x0`` (encode through
the frozen VAE instead of using raw pixels) and ``decode_fn`` (decode
through the frozen VAE instead of an identity un-normalize) differ.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from ffhq_repo.config import load_config
from ffhq_repo.data.dataset import find_data_root
from ffhq_repo.data.pretrained_vae import find_vae_checkpoint, load_pretrained_vae, factor_latent_shape, decode_latent
from ffhq_repo.utils.paths import resolve_device
from _diffusion_common import run_training_loop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a configs/ldm/*.yaml file")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device)
    print("Using device:", device)

    vae_checkpoint_path = cfg.model.vae_checkpoint_path
    if vae_checkpoint_path is None:
        vae_checkpoint_path = find_vae_checkpoint(cfg.data.data_root_search)
    print("VAE checkpoint:", vae_checkpoint_path)

    vae, vae_img_size, latent_dim, vae_base_channels = load_pretrained_vae(
        vae_checkpoint_path, cfg.data.img_size, device,
    )

    min_spatial = 2 ** (len(cfg.model.channel_mults) - 1)
    latent_channels, latent_spatial = factor_latent_shape(latent_dim, min_spatial)
    print(f"Latent shape: flat=({latent_dim},) -> reshaped to "
          f"(C={latent_channels}, H={latent_spatial}, W={latent_spatial}) for the U-Net.")

    def get_x0(raw_batch, device):
        images = raw_batch.to(device, non_blocking=True)
        bs = images.size(0)
        with torch.no_grad():
            mu, _ = vae.encoder(images)
            x0 = mu.reshape(bs, latent_channels, latent_spatial, latent_spatial)
        repa_target = images * 2 - 1  # REPA's ResNet-18 target expects [-1, 1]; dataset here is [0, 1]
        return x0, repa_target

    def decode_fn(z_spatial):
        return decode_latent(vae, z_spatial, latent_dim)

    run_training_loop(
        cfg, device, experiment_name_suffix="",
        in_channels=latent_channels, img_size=latent_spatial,
        get_x0=get_x0, decode_fn=decode_fn,
        normalize_to_unit_range=False,  # VAE convention: images stay in [0, 1]
    )


if __name__ == "__main__":
    main()
