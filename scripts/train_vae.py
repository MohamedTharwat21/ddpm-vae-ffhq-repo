#!/usr/bin/env python3
"""Train a VAE on FFHQ from a YAML config.

Usage:
    python scripts/train_vae.py --config configs/vae/v2.yaml

Mirrors ``vae_ffhq_pipeline.ipynb``'s training loop (section 9) exactly -
same optimizer, same AMP usage, same checkpoint cadence
(``checkpoint_last.pt`` every epoch, ``checkpoint_best.pt`` on val
improvement), same fixed-z/fixed-recon monitoring artifacts - just driven
by a config file and importing from ``ffhq_repo`` instead of being copied
into a notebook cell per experiment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ffhq_repo.config import load_config
from ffhq_repo.data.dataset import (
    discover_all_images, find_data_root, generate_or_load_split,
    verify_split, stable_seed, ManifestImageDataset,
)
from ffhq_repo.models.vae import VAE, VGGPerceptualLoss, vae_loss, get_beta
from ffhq_repo.training.checkpoint import save_checkpoint, load_checkpoint
from ffhq_repo.training.seeding import set_all_seeds
from ffhq_repo.utils.paths import resolve_device


def build_dataloaders(cfg, device):
    data_root = cfg.data.data_root or find_data_root(cfg.data.data_root_search)
    print("Dataset root:", data_root)

    all_images = discover_all_images(data_root)
    split = generate_or_load_split(
        cfg.data.split_file, all_images, cfg.data.train_count, cfg.data.val_count, cfg.data.split_seed,
    )
    verify_split(split, all_images)

    manifest = pd.DataFrame({
        "filepath": split["train"] + split["val"],
        "vae_split": (["train"] * len(split["train"])) + (["val"] * len(split["val"])),
    })

    train_ds = ManifestImageDataset(manifest, data_root, "train", cfg.data.img_size, normalize_to_unit_range=False)
    val_ds = ManifestImageDataset(manifest, data_root, "val", cfg.data.img_size, normalize_to_unit_range=False)
    print(f"train: {len(train_ds)} images | val: {len(val_ds)} images")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.training.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers, pin_memory=True,
    )
    return train_loader, val_loader, val_ds


def train_one_epoch(model, loader, optimizer, beta, scaler, device, use_amp, perceptual_fn, perceptual_weight):
    model.train()
    total = total_recon = total_perc = total_kld = 0.0
    n = 0
    for x in loader:
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = x.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            recon, mu, logvar = model(x)
            loss, recon_loss, perc_loss, kld = vae_loss(
                recon, x, mu, logvar, beta, perceptual_fn, perceptual_weight,
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = x.size(0)
        total += loss.item() * bs
        total_recon += recon_loss.item() * bs
        total_perc += perc_loss.item() * bs
        total_kld += kld.item() * bs
        n += bs
    return total / n, total_recon / n, total_perc / n, total_kld / n


@torch.no_grad()
def evaluate(model, loader, beta, device, perceptual_fn, perceptual_weight):
    model.eval()
    total = total_recon = total_perc = total_kld = 0.0
    n = 0
    for x in loader:
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = x.to(device, non_blocking=True)
        recon, mu, logvar = model(x)
        loss, recon_loss, perc_loss, kld = vae_loss(recon, x, mu, logvar, beta, perceptual_fn, perceptual_weight)
        bs = x.size(0)
        total += loss.item() * bs
        total_recon += recon_loss.item() * bs
        total_perc += perc_loss.item() * bs
        total_kld += kld.item() * bs
        n += bs
    return total / n, total_recon / n, total_perc / n, total_kld / n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a configs/vae/*.yaml file")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device)
    print("Using device:", device)

    set_all_seeds(cfg.data.global_seed)

    output_dir = cfg.paths.output_dir
    for sub in ["", "generations", "reconstructions"]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    train_loader, val_loader, val_ds = build_dataloaders(cfg, device)

    model = VAE(cfg.data.img_size, cfg.model.latent_dim, cfg.model.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters")

    perceptual_fn = None
    perceptual_weight = 0.0
    if cfg.perceptual_loss.enabled:
        perceptual_fn = VGGPerceptualLoss(
            layers=cfg.perceptual_loss.vgg_layers, pretrained=cfg.perceptual_loss.vgg_pretrained,
        ).to(device)
        perceptual_weight = cfg.perceptual_loss.weight
        print(f"Perceptual loss enabled: {cfg.perceptual_loss.vgg_model}, "
              f"layers={cfg.perceptual_loss.vgg_layers}, weight={perceptual_weight}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.use_amp and device.type == "cuda")

    # --- fixed monitoring artifacts (same-seed generation grid + reconstruction grid) ---
    fixed_z_path = os.path.join(output_dir, "fixed_z.pt")
    if os.path.exists(fixed_z_path):
        fixed_z = torch.load(fixed_z_path)
    else:
        g = torch.Generator().manual_seed(stable_seed(cfg.data.global_seed, "fixed_z"))
        fixed_z = torch.randn(cfg.monitoring.num_generation_samples, cfg.model.latent_dim, generator=g)
        torch.save(fixed_z, fixed_z_path)
    fixed_z = fixed_z.to(device)

    fixed_recon_path = os.path.join(output_dir, "fixed_recon_indices.json")
    if os.path.exists(fixed_recon_path):
        with open(fixed_recon_path) as f:
            fixed_recon_idx = json.load(f)
    else:
        rng = np.random.RandomState(stable_seed(cfg.data.global_seed, "fixed_recon"))
        n = min(cfg.monitoring.num_recon_samples, len(val_ds))
        fixed_recon_idx = rng.choice(len(val_ds), size=n, replace=False).tolist()
        with open(fixed_recon_path, "w") as f:
            json.dump(fixed_recon_idx, f)
    fixed_recon_batch = torch.stack([val_ds[i] for i in fixed_recon_idx]).to(device)

    # --- resume ---
    history = []
    start_epoch = 1
    ckpt_last_path = os.path.join(output_dir, "checkpoint_last.pt")
    if cfg.training.resume and os.path.exists(ckpt_last_path):
        ckpt = load_checkpoint(ckpt_last_path, model, optimizer, device)
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", [])
        print(f"Resumed from checkpoint at epoch {ckpt['epoch']}; continuing at epoch {start_epoch}.")
    else:
        print("Starting training from scratch.")

    best_val = min((h["val_loss"] for h in history), default=float("inf"))

    for epoch in range(start_epoch, cfg.training.epochs + 1):
        t0 = time.time()
        beta = get_beta(epoch, cfg.training.beta_warmup_epochs, cfg.training.beta)

        train_loss, train_recon, train_perc, train_kld = train_one_epoch(
            model, train_loader, optimizer, beta, scaler, device, cfg.training.use_amp,
            perceptual_fn, perceptual_weight,
        )
        val_loss, val_recon, val_perc, val_kld = evaluate(
            model, val_loader, beta, device, perceptual_fn, perceptual_weight,
        )

        model.eval()
        with torch.no_grad():
            gen_grid = vutils.make_grid(model.decoder(fixed_z).cpu(), nrow=8)
            vutils.save_image(gen_grid, os.path.join(output_dir, "generations", f"epoch_{epoch:03d}.png"))
            recon, _, _ = model(fixed_recon_batch)
            comparison = torch.cat([fixed_recon_batch.cpu(), recon.cpu()], dim=0)
            recon_grid = vutils.make_grid(comparison, nrow=fixed_recon_batch.size(0))
            vutils.save_image(recon_grid, os.path.join(output_dir, "reconstructions", f"epoch_{epoch:03d}.png"))

        history.append(dict(
            epoch=epoch, beta=beta,
            train_loss=train_loss, train_recon=train_recon, train_perceptual=train_perc, train_kld=train_kld,
            val_loss=val_loss, val_recon=val_recon, val_perceptual=val_perc, val_kld=val_kld,
            seconds=time.time() - t0,
        ))
        pd.DataFrame(history).to_csv(os.path.join(output_dir, "history.csv"), index=False)

        print(f"Epoch {epoch:3d}/{cfg.training.epochs} | beta={beta:.3f} | "
              f"train {train_loss:8.2f} (recon {train_recon:8.2f}, kld {train_kld:7.2f}) | "
              f"val {val_loss:8.2f} (recon {val_recon:8.2f}, kld {val_kld:7.2f}) | "
              f"{time.time() - t0:.1f}s")

        save_checkpoint(ckpt_last_path, model, optimizer, epoch, history, cfg)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(os.path.join(output_dir, "checkpoint_best.pt"), model, None, epoch, history, cfg)

    print("Training complete. Best val loss:", best_val)


if __name__ == "__main__":
    main()
