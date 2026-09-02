"""Convolutional VAE used by all three FFHQ VAE experiments (v1, v2, v3).

See`reports/experiments/vae.md` ("Architecture note") for a full explanation of
that discrepancy and why this module is the one considered authoritative.

Differences between v1 / v2 / v3 are entirely configuration-driven:

- ``base_channels``: 32 (v1, v2) or 64 (v3).
- ``latent_dim``: 128 for all three supplied configs.
- perceptual loss: off for v1/v3, on for v2 (see :class:`VGGPerceptualLoss`
  and the ``use_perceptual_loss`` flag on :func:`vae_loss`).

Nothing about the Encoder/Decoder/VAE classes themselves changes between
versions - one implementation, three configs, per the project's "do not
duplicate implementations" rule.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Encoder", "Decoder", "VAE", "VGGPerceptualLoss", "vae_loss", "get_beta"]


class Encoder(nn.Module):
    """4x stride-2 conv encoder -> flatten -> linear heads for mu/logvar.

    ``img_size`` must be divisible by 16 (four stride-2 conv layers).
    """

    def __init__(self, img_size: int, latent_dim: int, base_channels: int):
        super().__init__()
        assert img_size % 16 == 0, "IMG_SIZE must be divisible by 16 (4 stride-2 conv layers)"
        c = base_channels

        self.net = nn.Sequential(
            nn.Conv2d(3, c, 4, 2, 1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(0.2, inplace=True),  # -> img_size/2
            nn.Conv2d(c, c * 2, 4, 2, 1),
            nn.BatchNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),  # -> img_size/4
            nn.Conv2d(c * 2, c * 4, 4, 2, 1),
            nn.BatchNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),  # -> img_size/8
            nn.Conv2d(c * 4, c * 8, 4, 2, 1),
            nn.BatchNorm2d(c * 8),
            nn.LeakyReLU(0.2, inplace=True),  # -> img_size/16
        )

        self.reduced_size = img_size // 16
        flat_dim = c * 8 * self.reduced_size * self.reduced_size

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.net(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """Mirrors Encoder with transposed convolutions. No output activation:

    the reconstruction loss is MSE against images in [0, 1], and the
    decoder's raw (unbounded) output is clamped to [0, 1] only at
    evaluation/sampling time - not squashed by a Sigmoid during training.
    This matches the actual v1/v2/v3 training runs, which is why there is
    no ``nn.Sigmoid()`` here (contrast with the BCE-loss prototype in
    ``notebooks/archive/vae_ffhq_pipeline.ipynb``, which does have one).
    """

    def __init__(self, img_size: int, latent_dim: int, base_channels: int):
        super().__init__()

        c = base_channels
        self.reduced_size = img_size // 16
        self.base_channels = c

        flat_dim = c * 8 * self.reduced_size * self.reduced_size
        self.fc = nn.Linear(latent_dim, flat_dim)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(c * 8, c * 4, 4, 2, 1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 2, c, 4, 2, 1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            # No Sigmoid for MSE reconstruction loss.
            nn.ConvTranspose2d(c, 3, 4, 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, self.base_channels * 8, self.reduced_size, self.reduced_size)
        return self.net(h)


class VAE(nn.Module):
    """Convolutional beta-VAE: Encoder -> reparameterize -> Decoder."""

    def __init__(self, img_size: int, latent_dim: int, base_channels: int):
        super().__init__()
        self.encoder = Encoder(img_size, latent_dim, base_channels)
        self.decoder = Decoder(img_size, latent_dim, base_channels)
        self.latent_dim = latent_dim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


class VGGPerceptualLoss(nn.Module):
    """Perceptual (feature-space) reconstruction loss, used only by VAE v2.

    Reconstructed to match the exact configuration v2 was trained with
    (``VGG_MODEL="vgg16"``, pretrained ImageNet weights, feature layers
    ``[3, 8, 15, 22]``, i.e. after the first four ReLU activations of
    ``torchvision.models.vgg16().features``, roughly conv1_2/conv2_2/
    conv3_3/conv4_3) since the original notebook cell that implemented it
    was not part of the material supplied for this consolidation. The loss
    itself - mean absolute error between VGG feature maps of the
    reconstruction and the target, summed across the configured layers - is
    the standard, widely-used formulation (Johnson et al. 2016) and was not
    invented for this port; only the exact original code was unavailable.
    Frozen throughout: no gradient ever flows into the VGG backbone.
    """

    def __init__(self, layers=(3, 8, 15, 22), pretrained: bool = True):
        super().__init__()
        import torchvision.models as tvm

        weights = tvm.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = tvm.vgg16(weights=weights).features
        self.layers = sorted(layers)
        self.blocks = nn.ModuleList()
        prev = 0
        for layer_idx in self.layers:
            self.blocks.append(vgg[prev:layer_idx + 1])
            prev = layer_idx + 1
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def _normalize(self, x01: torch.Tensor) -> torch.Tensor:
        return (x01 - self.mean) / self.std

    def forward(self, recon01: torch.Tensor, target01: torch.Tensor) -> torch.Tensor:
        h_recon = self._normalize(recon01.clamp(0, 1))
        h_target = self._normalize(target01.clamp(0, 1))
        loss = 0.0
        for block in self.blocks:
            h_recon = block(h_recon)
            h_target = block(h_target.detach())
            loss = loss + F.l1_loss(h_recon, h_target)
        return loss


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    perceptual_loss_fn: "VGGPerceptualLoss | None" = None,
    perceptual_weight: float = 0.0,
):
    """beta-VAE loss: MSE reconstruction + beta * KL, optionally + weighted perceptual term.

    Returns ``(total_loss, recon_loss, perceptual_loss, kld)`` - the extra
    ``perceptual_loss`` element is 0.0 (a plain tensor) when
    ``perceptual_loss_fn`` is None, so the same training loop works for
    v1/v3 (no perceptual term) and v2 (perceptual term on) without
    branching in the caller.
    """
    # MSE reconstruction loss, summed over all pixels/channels and averaged
    # over the batch (matches the actual v1/v2/v3 training convention).
    recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)

    perceptual = torch.tensor(0.0, device=x.device)
    if perceptual_loss_fn is not None and perceptual_weight > 0:
        perceptual = perceptual_loss_fn(recon, x)

    # KL(q(z|x) || N(0, I))
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    loss = recon_loss + beta * kld + perceptual_weight * perceptual
    return loss, recon_loss, perceptual, kld


def get_beta(epoch: int, warmup_epochs: int, target_beta: float) -> float:
    """Linear KL warm-up: beta ramps 0 -> target_beta over warmup_epochs."""
    if warmup_epochs <= 0:
        return target_beta
    return target_beta * min(1.0, epoch / warmup_epochs)
