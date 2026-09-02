"""Reusable evaluation metrics: reconstruction quality (MSE/PSNR/SSIM/LPIPS)
and generation quality (FID/Inception Score), shared by the VAE, pixel
DDPM, and latent DDPM (LDM) evaluation scripts.

Generalized from the evaluation snippets supplied for VAE v1/v2/v3, DDPM
pixel v1 (linear+logit-normal), DDPM cosine+EMA (65 and 100 epoch), and LDM
linear - all four used essentially the same torchmetrics-based FID/IS
pattern with small, model-specific differences in *how a batch of images is
produced* (VAE: ``decoder(z)``; pixel DDPM: ``scheduler.generate(...)``;
LDM: ``scheduler.generate(...)`` + ``decode_latent(...)``). This module
factors out the shared metric-computation loop and leaves the
model-specific "how do I make N images" part as a plain callable
(``generate_fn``).

All metrics operate on images in [0, 1] (torchmetrics' expected range for
``normalize=True``). Callers are responsible for converting their model's
native range (e.g. DDPM's [-1, 1]) before calling into this module -
exactly as the original evaluation notebooks did.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

__all__ = [
    "psnr_from_mse", "evaluate_vae_reconstruction", "evaluate_generation_fid_is",
]


def psnr_from_mse(mse: float) -> float:
    return 10.0 * math.log10(1.0 / max(mse, 1e-10))


@torch.no_grad()
def evaluate_vae_reconstruction(
    model,
    val_loader: DataLoader,
    device,
    max_samples: Optional[int] = None,
    use_lpips: bool = True,
):
    """Reconstruction MSE / PSNR / SSIM / (optionally) LPIPS over a validation loader.

    ``model(x)`` is expected to return ``(recon, mu, logvar)`` (the VAE
    forward signature); ``recon`` is clamped to [0, 1] before every metric.
    Mirrors the reconstruction-evaluation cell supplied for VAE v1/v2/v3.
    """
    from torchmetrics.image import StructuralSimilarityIndexMeasure

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = None
    if use_lpips:
        import lpips as lpips_lib
        lpips_metric = lpips_lib.LPIPS(net="alex").to(device).eval()

    model.eval()
    total_squared_error = 0.0
    total_pixels = 0
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    total_images = 0
    shown_originals, shown_reconstructions = None, None

    for x in tqdm(val_loader, desc="Evaluating reconstructions"):
        if isinstance(x, (tuple, list)):
            x = x[0]
        if max_samples is not None and total_images >= max_samples:
            break
        if max_samples is not None:
            remaining = max_samples - total_images
            x = x[:remaining]
        x = x.to(device, non_blocking=True)
        recon, _, _ = model(x)
        recon = recon.clamp(0.0, 1.0)

        squared_error = (recon - x).pow(2)
        batch_mse = squared_error.flatten(1).mean(dim=1)
        batch_psnr = 10.0 * torch.log10(1.0 / batch_mse.clamp_min(1e-10))
        batch_ssim = ssim_metric(x, recon)

        batch_size = x.size(0)
        total_squared_error += squared_error.sum().item()
        total_pixels += x.numel()
        total_psnr += batch_psnr.sum().item()
        total_ssim += batch_ssim.item() * batch_size

        if lpips_metric is not None:
            batch_lpips = lpips_metric(x * 2.0 - 1.0, recon * 2.0 - 1.0).flatten()
            total_lpips += batch_lpips.sum().item()

        total_images += batch_size
        if shown_originals is None:
            shown_originals = x[:8].cpu()
            shown_reconstructions = recon[:8].cpu()

    mse = total_squared_error / total_pixels
    result = dict(
        num_samples=total_images,
        mse=mse,
        psnr=psnr_from_mse(mse),
        ssim=total_ssim / total_images,
        originals=shown_originals,
        reconstructions=shown_reconstructions,
    )
    if lpips_metric is not None:
        result["lpips"] = total_lpips / total_images
    return result


@torch.no_grad()
def evaluate_generation_fid_is(
    real_loader: Iterable,
    generate_fn: Callable[[int], torch.Tensor],
    num_generated: int,
    device,
    real_to_unit_range: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    batch_size: int = 64,
    fid_feature_dim: int = 2048,
):
    """FID + Inception Score between ``real_loader``'s images and ``generate_fn``'s output.

    Args:
        real_loader: yields real image batches (any range; converted via
            ``real_to_unit_range``, default identity).
        generate_fn(n): returns a batch of ``n`` generated images already in
            [0, 1] - the model-specific adapter (VAE decoder, DDPM/DDIM
            sampler + optional VAE decode for LDM, ...).
        num_generated: total number of generated images to evaluate on.
        real_to_unit_range: maps a real batch to [0, 1] (e.g. ``(x.clamp(-1,1)+1)/2``
            for DDPM-convention data, identity for VAE-convention [0, 1] data).

    Returns a dict with ``fid``, ``is_mean``, ``is_std``, ``num_real``,
    ``num_generated``, and ``sample_grid`` (the first generated batch, for
    a qualitative preview).
    """
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    fid_metric = FrechetInceptionDistance(feature=fid_feature_dim, normalize=True).to(device)
    is_metric = InceptionScore(normalize=True).to(device)
    fid_metric.eval()
    is_metric.eval()

    num_real = 0
    for real_batch in real_loader:
        if isinstance(real_batch, (tuple, list)):
            real_batch = real_batch[0]
        real_batch = real_to_unit_range(real_batch.to(device, non_blocking=True))
        fid_metric.update(real_batch, real=True)
        num_real += real_batch.size(0)

    sample_grid = None
    generated_count = 0
    while generated_count < num_generated:
        current_bs = min(batch_size, num_generated - generated_count)
        generated = generate_fn(current_bs).clamp(0.0, 1.0)
        fid_metric.update(generated, real=False)
        is_metric.update(generated)
        if sample_grid is None:
            sample_grid = generated[: min(64, current_bs)].cpu()
        generated_count += current_bs

    fid_score = fid_metric.compute().item()
    is_mean, is_std = is_metric.compute()
    return dict(
        fid=fid_score,
        is_mean=is_mean.item(),
        is_std=is_std.item(),
        num_real=num_real,
        num_generated=generated_count,
        sample_grid=sample_grid,
    )
