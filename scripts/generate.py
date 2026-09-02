#!/usr/bin/env python3
"""Generate a grid of unconditional samples from a trained checkpoint (VAE, pixel DDPM, or LDM).

Usage:
    python scripts/generate.py --config configs/vae/v2.yaml \\
        --checkpoint experiments/vae/v2/checkpoints/checkpoint_best.pt --out samples.png

    python scripts/generate.py --config configs/ddpm/cosine_ema_logit_v2_100.yaml \\
        --checkpoint experiments/pixel_ddpm/cosine_ema_logit_v2_100/checkpoints/latest_model.pt \\
        --sampler ddim --out samples_ddim.png
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torchvision.utils as vutils

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ffhq_repo.config import load_config
from ffhq_repo.data.pretrained_vae import find_vae_checkpoint, load_pretrained_vae, factor_latent_shape, decode_latent
from ffhq_repo.diffusion.scheduler import DiffusionScheduler
from ffhq_repo.models.unet import UNet
from ffhq_repo.models.vae import VAE
from ffhq_repo.utils.paths import resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vae-checkpoint", default=None)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--sampler", default=None, choices=["ddpm", "ddim"], help="Override config.diffusion.sampler")
    parser.add_argument("--out", default="generated_samples.png")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    if cfg.model.type == "vae":
        model = VAE(cfg.data.img_size, cfg.model.latent_dim, cfg.model.base_channels).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        with torch.no_grad():
            z = torch.randn(args.num_samples, cfg.model.latent_dim, device=device)
            samples = model.decoder(z).clamp(0, 1)

    else:
        is_latent = cfg.model.type == "ldm"
        decode_fn = lambda x: (x.clamp(-1, 1) + 1) / 2
        in_channels, img_size = 3, cfg.data.img_size

        if is_latent:
            vae_checkpoint_path = args.vae_checkpoint or cfg.model.vae_checkpoint_path or find_vae_checkpoint(cfg.data.data_root_search)
            vae, _, latent_dim, _ = load_pretrained_vae(vae_checkpoint_path, cfg.data.img_size, device)
            min_spatial = 2 ** (len(cfg.model.channel_mults) - 1)
            in_channels, img_size = factor_latent_shape(latent_dim, min_spatial)
            decode_fn = lambda z: decode_latent(vae, z, latent_dim)

        model = UNet(
            img_size=img_size, base_channels=cfg.model.base_channels, channel_mults=cfg.model.channel_mults,
            num_res_blocks=cfg.model.num_res_blocks, attention_resolutions=cfg.model.attention_resolutions,
            time_emb_dim=cfg.model.time_emb_dim, num_heads=cfg.model.num_heads, dropout=cfg.model.dropout,
            in_channels=in_channels,
        ).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "ema_model" in ckpt:
            model.load_state_dict(ckpt["ema_model"])
            print("Using EMA weights.")
        model.eval()

        scheduler = DiffusionScheduler(
            timesteps=cfg.diffusion.timesteps, schedule=cfg.diffusion.schedule,
            beta_start=cfg.diffusion.beta_start, beta_end=cfg.diffusion.beta_end,
            device=device, cosine_s=cfg.diffusion.cosine_s,
        )
        sampler = args.sampler or cfg.diffusion.sampler
        shape = (args.num_samples, in_channels, img_size, img_size)
        with torch.no_grad():
            raw = scheduler.generate(
                model, shape, device, sampler=sampler, ddim_steps=cfg.diffusion.ddim_steps,
                ddim_eta=cfg.diffusion.ddim_eta, clip_denoised=cfg.diffusion.clip_denoised,
            )
            samples = decode_fn(raw)
        print(f"Generated {args.num_samples} samples with sampler={sampler}.")

    nrow = max(1, int(args.num_samples ** 0.5))
    grid = vutils.make_grid(samples.cpu(), nrow=nrow)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    vutils.save_image(grid, args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
