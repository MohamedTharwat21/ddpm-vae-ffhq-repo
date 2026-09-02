"""REPA — a lightweight, practical take on representation alignment (Yu et al. 2024).

This is NOT a literal reproduction of the
REPA paper (which aligns a DiT against DINOv2 on a specific schedule) - it
is a lightweight, configurable version of the same idea, sized for a small
pixel-space U-Net on a single Kaggle GPU:

- Take the U-Net's bottleneck feature map, global-average-pool it, and
  project it through a small trainable MLP head.
- Take the same clean images x0, resize to 224 and ImageNet-normalize, and
  run them through a frozen pretrained ResNet-18 (avgpool features).
- Loss = 1 - cosine_similarity(projected_unet_features, frozen_target_features),
  averaged over the batch, added to the main epsilon-MSE loss with weight
  ``REPA_WEIGHT``.

Used by the ``pixel_ddpm/linear_logit_repa`` experiment only; every other
experiment in this project has ``USE_REPA=False``. See
``reports/experiments/ddpm_pixel.md`` for the actual (mixed/negative)
experimental result - REPA is not assumed to have improved anything.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RepaAlignment"]


class RepaAlignment(nn.Module):
    def __init__(self, unet_feature_channels: int, pretrained: bool = True):
        super().__init__()
        import torchvision.models as tvm

        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        encoder = tvm.resnet18(weights=weights)
        # Keep everything up to (and including) global average pooling;
        # drop the final classification layer - we just want the 512-d features.
        self.encoder = nn.Sequential(*list(encoder.children())[:-1])
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(unet_feature_channels, 512), nn.SiLU(), nn.Linear(512, 512),
        )

    @torch.no_grad()
    def target_features(self, x0_minus1_to_1: torch.Tensor) -> torch.Tensor:
        """x0_minus1_to_1: clean images in [-1, 1], shape (B, 3, H, W)."""
        x01 = (x0_minus1_to_1.clamp(-1, 1) + 1) / 2  # -> [0, 1]
        x_resized = F.interpolate(x01, size=224, mode="bilinear", align_corners=False)
        x_norm = (x_resized - self.imagenet_mean) / self.imagenet_std
        feats = self.encoder(x_norm).flatten(1)  # (B, 512)
        return feats

    def forward(self, unet_bottleneck_features: torch.Tensor, x0_minus1_to_1: torch.Tensor) -> torch.Tensor:
        target = self.target_features(x0_minus1_to_1)
        pred = self.projection(unet_bottleneck_features)
        cos_sim = F.cosine_similarity(pred, target, dim=-1)
        return (1 - cos_sim).mean()
