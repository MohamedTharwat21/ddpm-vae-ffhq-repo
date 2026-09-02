#!/usr/bin/env python3
"""Reconstruct a batch of real validation images through a trained VAE
checkpoint and save an Original (top) / Reconstruction (bottom) grid.

Usage:
    python scripts/reconstruct.py --config configs/vae/v2.yaml \\
        --checkpoint experiments/vae/v2/checkpoints/checkpoint_best.pt \\
        --out reconstructions.png
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ffhq_repo.config import load_config
from ffhq_repo.data.dataset import discover_all_images, find_data_root, generate_or_load_split, ManifestImageDataset
from ffhq_repo.evaluation.metrics import psnr_from_mse
from ffhq_repo.models.vae import VAE
from ffhq_repo.utils.paths import resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--out", default="reconstructions.png")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.model.type != "vae":
        raise ValueError("reconstruct.py only applies to VAE configs (config.model.type == 'vae').")
    device = resolve_device(args.device)

    model = VAE(cfg.data.img_size, cfg.model.latent_dim, cfg.model.base_channels).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("Loaded checkpoint:", args.checkpoint, "| epoch:", ckpt.get("epoch", "unknown"))

    data_root = cfg.data.data_root or find_data_root(cfg.data.data_root_search)
    all_images = discover_all_images(data_root)
    split = generate_or_load_split(
        cfg.data.split_file, all_images, cfg.data.train_count, cfg.data.val_count, cfg.data.split_seed,
    )
    manifest = pd.DataFrame({
        "filepath": split["train"] + split["val"],
        "vae_split": (["train"] * len(split["train"])) + (["val"] * len(split["val"])),
    })
    val_ds = ManifestImageDataset(manifest, data_root, "val", cfg.data.img_size, normalize_to_unit_range=False)
    loader = DataLoader(val_ds, batch_size=args.num_images, shuffle=True)
    batch = next(iter(loader))
    if isinstance(batch, (tuple, list)):
        batch = batch[0]
    batch = batch.to(device)

    with torch.no_grad():
        recon, _, _ = model(batch)
        recon = recon.clamp(0, 1)

    mse = (recon - batch).pow(2).mean().item()
    print(f"Batch MSE: {mse:.6f} | PSNR: {psnr_from_mse(mse):.3f} dB")

    comparison = torch.cat([batch.cpu(), recon.cpu()], dim=0)
    grid = vutils.make_grid(comparison, nrow=batch.size(0))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    vutils.save_image(grid, args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
