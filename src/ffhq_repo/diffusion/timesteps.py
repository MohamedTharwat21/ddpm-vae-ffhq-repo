"""Per-step training timestep sampling.

Audited (see ``reports/experiments/ddpm_pixel.md``) and found correct as
originally implemented in ``ddpm_ffhq_pipeline.ipynb`` - ported here
unchanged. mu/sigma are configurable per experiment.
"""
from __future__ import annotations

import torch

__all__ = ["sample_timesteps"]


def sample_timesteps(
    batch_size: int,
    timesteps: int,
    method: str,
    device,
    logit_normal_mean: float = 0.0,
    logit_normal_std: float = 1.0,
) -> torch.Tensor:
    """Sample a batch of integer timesteps in [0, timesteps).

    ``method="uniform"``: t ~ Uniform{0, ..., T-1} (the original DDPM recipe).

    ``method="logit_normal"``: draw z ~ Normal(logit_normal_mean,
    logit_normal_std), squash through sigmoid to get a fraction in (0, 1),
    scale by ``timesteps`` and truncate to an integer, then clamp to
    [0, timesteps - 1] as a safety net against the (extremely rare)
    floating-point edge case where the fraction rounds up to exactly 1.0.
    """
    if method == "uniform":
        return torch.randint(0, timesteps, (batch_size,), device=device).long()
    elif method == "logit_normal":
        u = torch.randn(batch_size, device=device) * logit_normal_std + logit_normal_mean
        frac = torch.sigmoid(u)  # (0, 1), concentrated around sigmoid(mean)
        t = (frac * timesteps).long().clamp(0, timesteps - 1)
        return t
    else:
        raise ValueError(f"Unknown timestep sampling method: {method!r} (expected 'uniform' or 'logit_normal')")
