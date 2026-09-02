"""DDPM U-Net (Ho et al. 2020 / OpenAI improved-diffusion style).

Used, unmodified, by every DDPM/LDM experiment in this project (pixel-space
and latent-space alike) - only ``in_channels`` and ``img_size`` differ
between the two (raw RGB channels/resolution for pixel DDPM, the frozen
VAE's latent channels/spatial grid for latent DDPM). This is the single
source of truth for the architecture; do not duplicate it per-experiment.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ResidualBlock", "SelfAttention", "TimeEmbedding", "Downsample", "Upsample", "UNet",
]


class ResidualBlock(nn.Module):
    """Pre-activation (GroupNorm -> SiLU -> Conv) block with additive time conditioning."""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention over spatial positions, applied only at configured resolutions."""

    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv.unbind(1)
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum("bhcn,bhcm->bhnm", q, k) * scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v).reshape(B, C, H, W)
        return x + self.proj(out)


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Standard transformer-style sinusoidal embedding of integer timesteps."""
    import math

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimeEmbedding(nn.Module):
    def __init__(self, base_dim: int, time_emb_dim: int):
        super().__init__()
        self.base_dim = base_dim
        self.mlp = nn.Sequential(
            nn.Linear(base_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_timestep_embedding(t, self.base_dim))


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)


class UNet(nn.Module):
    """Encoder/bottleneck/decoder U-Net with skip connections, driven entirely by config.

    ``in_channels``/``img_size`` are 3/pixel-resolution for pixel-space
    DDPM, or ``LATENT_CHANNELS``/``LATENT_SPATIAL`` (derived from the
    frozen VAE, see :mod:`ffhq_repo.data.pretrained_vae`) for latent-space
    DDPM - the class itself is identical either way.

    ``forward(x, t, return_features=False)``: set ``return_features=True``
    to also return the bottleneck feature map, used only by the optional
    REPA loss (:mod:`ffhq_repo.training.repa`).
    """

    def __init__(
        self,
        img_size: int,
        base_channels: int,
        channel_mults,
        num_res_blocks: int,
        attention_resolutions,
        time_emb_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        in_channels: int = 3,
    ):
        super().__init__()
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks

        self.time_embedding = TimeEmbedding(base_channels, time_emb_dim)
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down_res_blocks = nn.ModuleList()
        self.down_attn_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skip_channels = [base_channels]
        now_ch = base_channels
        cur_res = img_size
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_res_blocks.append(ResidualBlock(now_ch, out_ch, time_emb_dim, dropout))
                use_attn = cur_res in attention_resolutions
                self.down_attn_blocks.append(SelfAttention(out_ch, num_heads) if use_attn else nn.Identity())
                now_ch = out_ch
                skip_channels.append(now_ch)
            if level != len(channel_mults) - 1:
                self.downsamples.append(Downsample(now_ch))
                skip_channels.append(now_ch)
                cur_res //= 2
            else:
                self.downsamples.append(None)

        self.mid_block1 = ResidualBlock(now_ch, now_ch, time_emb_dim, dropout)
        self.mid_attn = SelfAttention(now_ch, num_heads)
        self.mid_block2 = ResidualBlock(now_ch, now_ch, time_emb_dim, dropout)
        self.bottleneck_channels = now_ch

        self.up_res_blocks = nn.ModuleList()
        self.up_attn_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            use_attn = cur_res in attention_resolutions
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                self.up_res_blocks.append(ResidualBlock(now_ch + skip_ch, out_ch, time_emb_dim, dropout))
                self.up_attn_blocks.append(SelfAttention(out_ch, num_heads) if use_attn else nn.Identity())
                now_ch = out_ch
            if level != 0:
                self.upsamples.append(Upsample(now_ch))
                cur_res *= 2
            else:
                self.upsamples.append(None)

        self.out_norm = nn.GroupNorm(8, now_ch)
        self.out_conv = nn.Conv2d(now_ch, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, return_features: bool = False):
        t_emb = self.time_embedding(t)
        h = self.init_conv(x)
        skips = [h]

        idx = 0
        for level in range(len(self.channel_mults)):
            for _ in range(self.num_res_blocks):
                h = self.down_res_blocks[idx](h, t_emb)
                h = self.down_attn_blocks[idx](h)
                skips.append(h)
                idx += 1
            if self.downsamples[level] is not None:
                h = self.downsamples[level](h)
                skips.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        bottleneck_features = h  # used by the optional REPA loss

        idx = 0
        for up_pos, level in enumerate(reversed(range(len(self.channel_mults)))):
            for _ in range(self.num_res_blocks + 1):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.up_res_blocks[idx](h, t_emb)
                h = self.up_attn_blocks[idx](h)
                idx += 1
            if self.upsamples[up_pos] is not None:
                h = self.upsamples[up_pos](h)

        out = self.out_conv(F.silu(self.out_norm(h)))
        if return_features:
            return out, bottleneck_features
        return out
