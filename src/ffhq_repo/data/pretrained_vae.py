"""Loading a frozen, pretrained VAE for latent-space DDPM (LDM) experiments.


**Architecture note.** The actual VAE v2 checkpoint used for the LDM experiments (per the supplied
training config/results) was trained with the MSE-loss, no-Sigmoid
architecture in :mod:`ffhq_repo.models.vae` - see that module's docstring.
Because ``nn.Sigmoid`` carries no parameters, ``load_state_dict`` succeeds
either way and this discrepancy would not raise an error - it would only
silently squash the decoder's output through an extra sigmoid it was never
trained to expect. This module reconstructs the VAE using the correct
(no-Sigmoid) architecture, matching how VAE v2 was actually trained, and
clamps decoder output to [0, 1] explicitly instead (the same pattern the
VAE's own evaluation code uses). This is a corrected port, not a
re-design: same weights, same layer names, only the redundant/incorrect
Sigmoid activation is not applied.
"""
from __future__ import annotations

import os

import torch

from ffhq_repo.models.vae import VAE

__all__ = ["find_vae_checkpoint", "load_pretrained_vae", "factor_latent_shape", "decode_latent"]


def find_vae_checkpoint(search_root: str, filename: str = "checkpoint_best.pt") -> str:
    """Recursively search for the pretrained VAE checkpoint by exact filename."""
    for dirpath, _, filenames in os.walk(search_root):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    raise FileNotFoundError(
        f"Could not find {filename!r} anywhere under {search_root}. Point "
        "config.model.vae_checkpoint_path at the trained VAE checkpoint explicitly."
    )


def load_pretrained_vae(checkpoint_path: str, expected_img_size: int, device):
    """Load, freeze, and return (vae, img_size, latent_dim, base_channels).

    Inspects the checkpoint's structure rather than assuming it: a full
    training checkpoint (``dict(model=..., config=...)``, the format
    :mod:`scripts.train_vae` actually saves) or a bare state_dict.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
        saved_config = ckpt.get("config")
        print(f"Checkpoint is a full training checkpoint (epoch={ckpt.get('epoch')}, "
              f"val_loss={ckpt.get('val_loss')}).")
    elif isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in ckpt.values()):
        state_dict = ckpt
        saved_config = None
        print("Checkpoint is a bare state_dict (no epoch/config metadata).")
    else:
        raise ValueError(f"Unrecognized VAE checkpoint structure: top-level keys = {list(ckpt.keys())}")

    if saved_config is not None:
        img_size = saved_config["IMG_SIZE"]
        latent_dim = saved_config["LATENT_DIM"]
        base_channels = saved_config["BASE_CHANNELS"]
        print(f"VAE architecture read from checkpoint's saved config: "
              f"IMG_SIZE={img_size}, LATENT_DIM={latent_dim}, BASE_CHANNELS={base_channels}")
    else:
        # Infer entirely from the state_dict's own tensor shapes - never
        # from an unrelated config (e.g. the DDPM U-Net's own CONFIG keys).
        base_channels = state_dict["encoder.net.0.weight"].shape[0]
        latent_dim = state_dict["encoder.fc_mu.weight"].shape[0]
        flat_dim = state_dict["encoder.fc_mu.weight"].shape[1]
        reduced_size = round((flat_dim / (base_channels * 8)) ** 0.5)
        img_size = reduced_size * 16
        print("Checkpoint has no saved config - inferred entirely from state_dict tensor "
              f"shapes: BASE_CHANNELS={base_channels}, LATENT_DIM={latent_dim}, "
              f"IMG_SIZE={img_size} (encoder reduced_size={reduced_size}).")

    assert img_size == expected_img_size, (
        f"VAE checkpoint was trained with IMG_SIZE={img_size}, but this run's "
        f"config.data.img_size={expected_img_size}. These must match."
    )

    vae = VAE(img_size, latent_dim, base_channels).to(device)
    vae.load_state_dict(state_dict)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    print(f"VAE loaded and frozen: {sum(p.numel() for p in vae.parameters()):,} parameters, "
          f"all requires_grad=False, vae.training={vae.training}.")
    return vae, img_size, latent_dim, base_channels


def factor_latent_shape(latent_dim: int, min_spatial: int):
    """Smallest (channels, spatial) with spatial a positive multiple of
    min_spatial and spatial*spatial dividing latent_dim evenly (a pure
    reshape - no values dropped or padded)."""
    spatial = min_spatial
    while spatial * spatial <= latent_dim:
        if latent_dim % (spatial * spatial) == 0:
            return latent_dim // (spatial * spatial), spatial
        spatial += min_spatial
    raise ValueError(
        f"Could not factor latent_dim={latent_dim} into (channels, spatial, spatial) with "
        f"spatial a multiple of {min_spatial}."
    )


def decode_latent(vae: VAE, z_spatial: torch.Tensor, latent_dim: int) -> torch.Tensor:
    """Reshape a (B, C, H, W) DDPM output back into the VAE's flat latent
    vector and decode it into an RGB image via the frozen VAE decoder."""
    with torch.no_grad():
        z_flat = z_spatial.reshape(z_spatial.size(0), latent_dim)
        img = vae.decoder(z_flat)
    return img.clamp(0, 1)
