"""Shared DDPM/LDM training loop, imported by both train_ddpm.py and
train_ldm.py so the two CLI entry points do not duplicate the training
loop, sampling/visualization helpers, or the main epoch loop - only the
model-specific "how do I get a clean x0 batch, and how do I turn a
generated latent/pixel batch into a displayable [0,1] image" pieces differ,
and those are passed in as plain callables. See
ffhq_repo/diffusion/scheduler.py and ffhq_repo/models/unet.py for the
actual model/scheduler code this loop calls into.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import DataLoader

from ffhq_repo.data.dataset import (
    discover_all_images, find_data_root, generate_or_load_split,
    verify_split, stable_seed, ManifestImageDataset,
)
from ffhq_repo.diffusion.scheduler import DiffusionScheduler
from ffhq_repo.diffusion.timesteps import sample_timesteps
from ffhq_repo.models.unet import UNet
from ffhq_repo.training.checkpoint import save_checkpoint, load_checkpoint
from ffhq_repo.training.ema import build_ema_model, update_ema
from ffhq_repo.training.repa import RepaAlignment
from ffhq_repo.training.seeding import set_all_seeds
from ffhq_repo.utils.paths import make_experiment_dir


def build_dataloaders(cfg, normalize_to_unit_range: bool):
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

    train_ds = ManifestImageDataset(
        manifest, data_root, "train", cfg.data.img_size, normalize_to_unit_range=normalize_to_unit_range,
    )
    val_ds = ManifestImageDataset(
        manifest, data_root, "val", cfg.data.img_size, normalize_to_unit_range=normalize_to_unit_range,
    )
    print(f"train: {len(train_ds)} images | val: {len(val_ds)} images")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.training.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=cfg.training.persistent_workers and cfg.training.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers, pin_memory=True,
        persistent_workers=cfg.training.persistent_workers and cfg.training.num_workers > 0,
    )
    return train_loader, val_loader


def build_model_and_scheduler(cfg, device, in_channels: int, img_size: int):
    model = UNet(
        img_size=img_size,
        base_channels=cfg.model.base_channels,
        channel_mults=cfg.model.channel_mults,
        num_res_blocks=cfg.model.num_res_blocks,
        attention_resolutions=cfg.model.attention_resolutions,
        time_emb_dim=cfg.model.time_emb_dim,
        num_heads=cfg.model.num_heads,
        dropout=cfg.model.dropout,
        in_channels=in_channels,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net has {n_params:,} parameters")

    scheduler = DiffusionScheduler(
        timesteps=cfg.diffusion.timesteps, schedule=cfg.diffusion.schedule,
        beta_start=cfg.diffusion.beta_start, beta_end=cfg.diffusion.beta_end,
        device=device, cosine_s=cfg.diffusion.cosine_s,
    )
    print(f"Diffusion scheduler ready: {cfg.diffusion.timesteps} timesteps, "
          f"'{cfg.diffusion.schedule}' schedule, clip_denoised={cfg.diffusion.clip_denoised}, "
          f"sampler='{cfg.diffusion.sampler}'.")

    ema_model = build_ema_model(model, device) if cfg.ema.enabled else None
    if ema_model is not None:
        print(f"EMA model initialized (decay={cfg.ema.decay}).")

    repa_module = None
    if cfg.repa.enabled:
        repa_module = RepaAlignment(model.bottleneck_channels, pretrained=cfg.repa.pretrained).to(device)
        print(f"REPA enabled (weight={cfg.repa.weight}).")

    return model, scheduler, ema_model, repa_module


def train_one_epoch(model, loader, scheduler, optimizer, scaler, device, cfg, get_x0, ema_model, repa_module):
    model.train()
    total_loss = total_mse = total_repa = 0.0
    n = 0
    for raw_batch in loader:
        x0, repa_target = get_x0(raw_batch, device)
        bs = x0.size(0)

        t = sample_timesteps(
            bs, cfg.diffusion.timesteps, cfg.diffusion.timestep_sampling, device,
            cfg.diffusion.logit_normal_mean, cfg.diffusion.logit_normal_std,
        )
        noise = torch.randn_like(x0)
        x_t = scheduler.q_sample(x0, t, noise)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=cfg.training.use_amp and device.type == "cuda"):
            if repa_module is not None:
                pred_noise, bottleneck_features = model(x_t, t, return_features=True)
            else:
                pred_noise = model(x_t, t)
            mse_loss = F.mse_loss(pred_noise, noise)

            repa_loss = torch.tensor(0.0, device=device)
            if repa_module is not None:
                repa_loss = repa_module(bottleneck_features.float(), repa_target.float())

            loss = mse_loss + cfg.repa.weight * repa_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if ema_model is not None:
            update_ema(ema_model, model, cfg.ema.decay)

        total_loss += loss.item() * bs
        total_mse += mse_loss.item() * bs
        total_repa += repa_loss.item() * bs
        n += bs

    return total_loss / n, total_mse / n, total_repa / n


@torch.no_grad()
def evaluate(model, loader, scheduler, device, cfg, get_x0):
    model.eval()
    total_mse = 0.0
    n = 0
    for raw_batch in loader:
        x0, _ = get_x0(raw_batch, device)
        bs = x0.size(0)
        t = sample_timesteps(bs, cfg.diffusion.timesteps, cfg.diffusion.timestep_sampling, device,
                              cfg.diffusion.logit_normal_mean, cfg.diffusion.logit_normal_std)
        noise = torch.randn_like(x0)
        x_t = scheduler.q_sample(x0, t, noise)
        pred_noise = model(x_t, t)
        mse_loss = F.mse_loss(pred_noise, noise)
        total_mse += mse_loss.item() * bs
        n += bs
    return total_mse / n


def save_sample_grid(sampling_model, scheduler, fixed_noise, epoch, experiment_dir, cfg, device, decode_fn):
    sampling_model.eval()
    samples = scheduler.generate(
        sampling_model, fixed_noise.shape, device, sampler=cfg.diffusion.sampler,
        ddim_steps=cfg.diffusion.ddim_steps, ddim_eta=cfg.diffusion.ddim_eta,
        clip_denoised=cfg.diffusion.clip_denoised, start_noise=fixed_noise,
    )
    images = decode_fn(samples)  # -> [0, 1]
    grid = vutils.make_grid(images.cpu(), nrow=4)
    sampler_tag = "ddpm" if cfg.diffusion.sampler == "ddpm" else f"ddim{cfg.diffusion.ddim_steps}"
    path = os.path.join(experiment_dir, "generated_samples", f"epoch_{epoch:04d}_{sampler_tag}.png")
    vutils.save_image(grid, path)
    return path


def make_denoising_gif(sampling_model, scheduler, fixed_noise_single, out_path, trajectory_every, cfg, device,
                        decode_fn, duration_ms: int = 150):
    sampling_model.eval()
    _, trajectory = scheduler.generate(
        sampling_model, fixed_noise_single.shape, device, sampler=cfg.diffusion.sampler,
        ddim_steps=cfg.diffusion.ddim_steps, ddim_eta=cfg.diffusion.ddim_eta,
        clip_denoised=cfg.diffusion.clip_denoised, start_noise=fixed_noise_single,
        return_trajectory=True, trajectory_every=trajectory_every,
    )
    frames = []
    for x in trajectory:
        img01 = decode_fn(x.to(device)).cpu()
        arr = (img01[0].permute(1, 2, 0).numpy() * 255).astype("uint8")
        frames.append(Image.fromarray(arr).resize((256, 256), Image.NEAREST))
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    print(f"Saved denoising GIF ({cfg.diffusion.sampler.upper()} sampler) to {out_path} ({len(frames)} frames)")
    return trajectory


def make_denoising_grid(trajectory, out_path, device, decode_fn, num_snapshots: int = 10):
    idxs = np.linspace(0, len(trajectory) - 1, num_snapshots).round().astype(int)
    idxs = sorted(set(idxs.tolist()))
    frames01 = [decode_fn(trajectory[i].to(device)).cpu()[0] for i in idxs]
    grid = vutils.make_grid(torch.stack(frames01), nrow=len(frames01))
    vutils.save_image(grid, out_path)
    print(f"Saved denoising snapshot grid to {out_path} ({len(frames01)} snapshots)")


def run_training_loop(cfg, device, experiment_name_suffix, in_channels, img_size, get_x0, decode_fn,
                       normalize_to_unit_range: bool = True):
    """The shared main loop: resume, per-epoch train/val, checkpointing,
    periodic sample grid + denoising GIF/grid. Used by both train_ddpm.py
    (pixel-space, decode_fn = identity clamp-to-[0,1]) and train_ldm.py
    (latent-space, decode_fn = frozen-VAE decode)."""
    set_all_seeds(cfg.data.global_seed)

    experiment_dir = make_experiment_dir(cfg.paths.output_root, cfg.paths.experiment_name + experiment_name_suffix)
    print("Experiment directory:", experiment_dir)
    with open(os.path.join(experiment_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    model, scheduler, ema_model, repa_module = build_model_and_scheduler(cfg, device, in_channels, img_size)

    fixed_noise_path = os.path.join(experiment_dir, "fixed_noise.pt")
    g = torch.Generator().manual_seed(stable_seed(cfg.data.global_seed, "fixed_noise"))
    fixed_noise = torch.randn(cfg.monitoring.num_monitor_samples, in_channels, img_size, img_size, generator=g).to(device)
    torch.save(fixed_noise.cpu(), fixed_noise_path)

    g2 = torch.Generator().manual_seed(stable_seed(cfg.data.global_seed, "fixed_noise_single"))
    fixed_noise_single = torch.randn(1, in_channels, img_size, img_size, generator=g2).to(device)
    torch.save(fixed_noise_single.cpu(), os.path.join(experiment_dir, "fixed_noise_single.pt"))

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.use_amp and device.type == "cuda")

    history = []
    start_epoch = 1
    latest_path = os.path.join(experiment_dir, "latest_model.pt")
    best_path = os.path.join(experiment_dir, "best_model.pt")
    history_path = os.path.join(experiment_dir, "training_history.csv")

    if cfg.training.resume and os.path.exists(latest_path):
        ckpt = load_checkpoint(latest_path, model, optimizer, device, repa_module=repa_module, ema_model=ema_model)
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", [])
        print(f"Resumed from epoch {ckpt['epoch']}; continuing at epoch {start_epoch}.")
    else:
        print("Starting training from scratch.")

    best_val = min((h["val_mse"] for h in history), default=float("inf"))
    sampling_model = ema_model if ema_model is not None else model

    train_loader, val_loader = build_dataloaders(cfg, normalize_to_unit_range=normalize_to_unit_range)

    for epoch in range(start_epoch, cfg.training.epochs + 1):
        t0 = time.time()
        train_loss, train_mse, train_repa = train_one_epoch(
            model, train_loader, scheduler, optimizer, scaler, device, cfg, get_x0, ema_model, repa_module,
        )
        val_mse = evaluate(model, val_loader, scheduler, device, cfg, get_x0)

        history.append(dict(
            epoch=epoch, train_loss=train_loss, train_mse=train_mse, train_repa=train_repa,
            val_mse=val_mse, seconds=time.time() - t0,
        ))
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(f"Epoch {epoch:4d}/{cfg.training.epochs} | train_mse {train_mse:.5f} | "
              f"val_mse {val_mse:.5f} | {time.time() - t0:.1f}s")

        save_checkpoint(latest_path, model, optimizer, epoch, history, cfg, repa_module=repa_module, ema_model=ema_model)
        if val_mse < best_val:
            best_val = val_mse
            save_checkpoint(best_path, model, optimizer, epoch, history, cfg, repa_module=repa_module, ema_model=ema_model)

        if epoch % cfg.monitoring.sample_every_n_epochs == 0 or epoch == cfg.training.epochs:
            path = save_sample_grid(sampling_model, scheduler, fixed_noise, epoch, experiment_dir, cfg, device, decode_fn)
            print("  saved sample grid:", path)

        if epoch % cfg.monitoring.denoising_vis_every_n_epochs == 0 or epoch == cfg.training.epochs:
            traj_every = (cfg.monitoring.trajectory_snapshot_every if cfg.diffusion.sampler == "ddpm"
                          else max(1, cfg.diffusion.ddim_steps // 10))
            gif_path = os.path.join(experiment_dir, "denoising", f"epoch_{epoch:04d}.gif")
            grid_path = os.path.join(experiment_dir, "denoising", f"epoch_{epoch:04d}_grid.png")
            trajectory = make_denoising_gif(sampling_model, scheduler, fixed_noise_single, gif_path,
                                             traj_every, cfg, device, decode_fn)
            make_denoising_grid(trajectory, grid_path, device, decode_fn)

    print("Training complete. Best val_mse:", best_val)
    return experiment_dir
