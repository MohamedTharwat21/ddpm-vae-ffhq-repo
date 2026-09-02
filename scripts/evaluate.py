#!/usr/bin/env python3
"""Evaluate a trained checkpoint: FID / Inception Score (all model types),
plus reconstruction MSE/PSNR/SSIM/LPIPS (VAE only).

Usage:
    python scripts/evaluate.py --config configs/vae/v2.yaml \\
        --checkpoint experiments/vae/v2/checkpoints/checkpoint_best.pt

    python scripts/evaluate.py --config configs/ddpm/cosine_ema_logit_v2_100.yaml \\
        --checkpoint experiments/pixel_ddpm/cosine_ema_logit_v2_100/checkpoints/latest_model.pt \\
        --num-samples 2048

    python scripts/evaluate.py --config configs/ldm/linear.yaml \\
        --checkpoint experiments/latent_ddpm/linear/checkpoints/latest_model.pt \\
        --vae-checkpoint experiments/vae/v2/checkpoints/checkpoint_best.pt

This is the same evaluation methodology (torchmetrics FID/IS, image range
handling, sample counts) reported throughout the README's Quantitative
Results tables - see README.md's "Evaluation Methodology" section for the
exact settings (number of generated images, reference set size, sampler,
timestep count, checkpoint, seed) that produced each reported number.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from ffhq_repo.config import load_config
from ffhq_repo.data.dataset import (
    discover_all_images, find_data_root, generate_or_load_split, ManifestImageDataset,
)
from ffhq_repo.data.pretrained_vae import find_vae_checkpoint, load_pretrained_vae, factor_latent_shape, decode_latent
from ffhq_repo.diffusion.scheduler import DiffusionScheduler
from ffhq_repo.evaluation.metrics import evaluate_generation_fid_is, evaluate_vae_reconstruction
from ffhq_repo.models.unet import UNet
from ffhq_repo.models.vae import VAE
from ffhq_repo.utils.paths import resolve_device


def build_val_loader(cfg, normalize_to_unit_range, batch_size):
    data_root = cfg.data.data_root or find_data_root(cfg.data.data_root_search)
    all_images = discover_all_images(data_root)
    split = generate_or_load_split(
        cfg.data.split_file, all_images, cfg.data.train_count, cfg.data.val_count, cfg.data.split_seed,
    )
    import pandas as pd
    manifest = pd.DataFrame({
        "filepath": split["train"] + split["val"],
        "vae_split": (["train"] * len(split["train"])) + (["val"] * len(split["val"])),
    })
    val_ds = ManifestImageDataset(manifest, data_root, "val", cfg.data.img_size, normalize_to_unit_range)
    num_workers = cfg.get("training", {}).get("num_workers", 2)
    return DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def evaluate_vae(cfg, checkpoint_path, num_samples, device):
    model = VAE(cfg.data.img_size, cfg.model.latent_dim, cfg.model.base_channels).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("Loaded checkpoint:", checkpoint_path, "| epoch:", ckpt.get("epoch", "unknown"))

    val_loader = build_val_loader(cfg, normalize_to_unit_range=False, batch_size=cfg.evaluation.eval_batch_size)
    recon_result = evaluate_vae_reconstruction(
        model, val_loader, device, max_samples=cfg.evaluation.reconstruction_samples,
    )
    print(f"Reconstruction samples: {recon_result['num_samples']:,}")
    print(f"MSE:   {recon_result['mse']:.6f}")
    print(f"PSNR:  {recon_result['psnr']:.3f} dB")
    print(f"SSIM:  {recon_result['ssim']:.4f}")
    if "lpips" in recon_result:
        print(f"LPIPS: {recon_result['lpips']:.4f}")

    real_loader = build_val_loader(cfg, normalize_to_unit_range=False, batch_size=cfg.evaluation.eval_batch_size)

    def generate_fn(n):
        z = torch.randn(n, cfg.model.latent_dim, device=device)
        return model.decoder(z).clamp(0, 1)

    gen_result = evaluate_generation_fid_is(
        real_loader, generate_fn, num_samples or cfg.evaluation.generation_samples, device,
        batch_size=cfg.evaluation.eval_batch_size,
    )
    print(f"Real images used for FID: {gen_result['num_real']:,}")
    print(f"Generated samples: {gen_result['num_generated']:,}")
    print(f"FID: {gen_result['fid']:.3f}")
    print(f"Inception Score: {gen_result['is_mean']:.3f} +/- {gen_result['is_std']:.3f}")


def evaluate_diffusion(cfg, checkpoint_path, num_samples, device, vae_checkpoint_override, is_latent):
    vae, latent_channels, latent_spatial, latent_dim, decode_fn = None, None, None, None, None
    in_channels, img_size, real_normalize = 3, cfg.data.img_size, False

    if is_latent:
        vae_checkpoint_path = vae_checkpoint_override or cfg.model.vae_checkpoint_path or find_vae_checkpoint(cfg.data.data_root_search)
        vae, _, latent_dim, _ = load_pretrained_vae(vae_checkpoint_path, cfg.data.img_size, device)
        min_spatial = 2 ** (len(cfg.model.channel_mults) - 1)
        latent_channels, latent_spatial = factor_latent_shape(latent_dim, min_spatial)
        in_channels, img_size = latent_channels, latent_spatial
        real_normalize = False  # dataset stays [0, 1] for LDM

        def decode_fn(z):
            return decode_latent(vae, z, latent_dim)
    else:
        real_normalize = True  # DDPM convention: [-1, 1]

        def decode_fn(x):
            return (x.clamp(-1, 1) + 1) / 2

    model = UNet(
        img_size=img_size, base_channels=cfg.model.base_channels, channel_mults=cfg.model.channel_mults,
        num_res_blocks=cfg.model.num_res_blocks, attention_resolutions=cfg.model.attention_resolutions,
        time_emb_dim=cfg.model.time_emb_dim, num_heads=cfg.model.num_heads, dropout=cfg.model.dropout,
        in_channels=in_channels,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    print("Checkpoint:", checkpoint_path)
    print("Checkpoint keys:", sorted(ckpt.keys()))
    print("Saved epoch:", ckpt.get("epoch", "unknown"))
    model.load_state_dict(ckpt["model"])
    if "ema_model" in ckpt:
        model.load_state_dict(ckpt["ema_model"])
        print("Loaded EMA weights for evaluation and sample generation.")
    else:
        print("No EMA weights found; evaluating the raw model weights.")
    model.eval()

    scheduler = DiffusionScheduler(
        timesteps=cfg.diffusion.timesteps, schedule=cfg.diffusion.schedule,
        beta_start=cfg.diffusion.beta_start, beta_end=cfg.diffusion.beta_end,
        device=device, cosine_s=cfg.diffusion.cosine_s,
    )

    real_loader = build_val_loader(cfg, normalize_to_unit_range=real_normalize, batch_size=cfg.training.batch_size)

    def generate_fn(n):
        shape = (n, in_channels, img_size, img_size)
        samples = scheduler.generate(
            model, shape, device, sampler=cfg.diffusion.sampler, ddim_steps=cfg.diffusion.ddim_steps,
            ddim_eta=cfg.diffusion.ddim_eta, clip_denoised=cfg.diffusion.clip_denoised,
        )
        return decode_fn(samples)

    def real_to_unit_range(x):
        return (x.clamp(-1, 1) + 1) / 2 if real_normalize else x

    n = num_samples or cfg.evaluation.num_samples
    result = evaluate_generation_fid_is(
        real_loader, generate_fn, n, device, real_to_unit_range=real_to_unit_range,
        batch_size=cfg.training.batch_size,
    )
    print(f"FID ({n} generated samples, sampler={cfg.diffusion.sampler}): {result['fid']:.4f}")
    print(f"Inception Score ({n} generated samples): {result['is_mean']:.4f} +/- {result['is_std']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vae-checkpoint", default=None, help="Override for LDM's frozen VAE checkpoint")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device)
    print("Using device:", device)

    model_type = cfg.model.type
    if model_type == "vae":
        evaluate_vae(cfg, args.checkpoint, args.num_samples, device)
    elif model_type == "ddpm":
        evaluate_diffusion(cfg, args.checkpoint, args.num_samples, device, args.vae_checkpoint, is_latent=False)
    elif model_type == "ldm":
        evaluate_diffusion(cfg, args.checkpoint, args.num_samples, device, args.vae_checkpoint, is_latent=True)
    else:
        raise ValueError(f"Unknown config.model.type: {model_type!r}")


if __name__ == "__main__":
    main()
