"""Diffusion (beta/alpha) schedules and the DDPM/DDIM reverse process.

This module is the single, unified scheduler used by every diffusion
experiment in this project - pixel-space DDPM (linear, linear+REPA,
cosine+EMA, linear+EMA) and latent-space DDPM/LDM (linear, cosine) alike.
One implementation, driven entirely by config, instead of a separate
scheduler per experiment.

See ``reports/experiments/ddpm_pixel.md`` ("Root cause of the cosine
high-contrast artifact") for the full writeup of the ``clip_denoised`` fix
below - it is included here, defaulted to ``True``, and is schedule-agnostic
(safe, and recommended, for "linear" too). Experiments run before the fix
existed (LDM cosine, in particular) are documented in the README as
historical/failed runs; they did not have access to this option.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

__all__ = ["linear_beta_schedule", "cosine_beta_schedule", "DiffusionScheduler"]


def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """Original DDPM (Ho et al. 2020) linear beta schedule."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal (2021), eq. 17: alpha_bar(t) = cos^2(((t/T + s) / (1+s)) * pi/2).

    Betas are derived from consecutive alpha_bar ratios and clipped to a
    numerically safe range (matches the reference implementation - the
    upper clip in particular keeps alpha_bar strictly positive everywhere,
    so 1/alpha_bar never divides by zero in ``predict_x0_from_eps`` below).
    Audited: this formula itself was NOT the source of the historical
    high-contrast/saturated-image artifact - see the scheduler-diagnostic
    section referenced above.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-4, 0.9999)


class DiffusionScheduler:
    """Forward (``q_sample``) and reverse (``p_sample`` / ``ddim_sample``) DDPM process.

    Args:
        timesteps: number of discrete diffusion steps T.
        schedule: ``"linear"`` or ``"cosine"``.
        beta_start, beta_end: only used when ``schedule == "linear"``.
        device: torch device for the precomputed buffers.
        cosine_s: offset for the cosine schedule; only used when
            ``schedule == "cosine"``.
    """

    def __init__(
        self,
        timesteps: int,
        schedule: str,
        beta_start: float,
        beta_end: float,
        device,
        cosine_s: float = 0.008,
    ):
        if schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(timesteps, s=cosine_s)
        else:
            raise ValueError(f"Unknown schedule: {schedule!r} (expected 'linear' or 'cosine')")

        self.timesteps = timesteps
        self.schedule_name = schedule
        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

        # Explicit x0-reconstruction + posterior-mean coefficients (Ho et al.
        # 2020, eq. 6-7). Schedule-agnostic - correct for "linear" too. These
        # let p_sample optionally clip the reconstructed x0 back into
        # [-1, 1] before computing the next reverse step - see the module
        # docstring for why this matters.
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
        out = a.gather(0, t)
        return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward process: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise."""
        sqrt_ac = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_1mac = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ac * x0 + sqrt_1mac * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Invert q_sample for x0 given a predicted noise eps."""
        sqrt_recip_ac_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_ac_t = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip_ac_t * x_t - sqrt_recipm1_ac_t * eps

    def q_posterior_mean(self, x0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Mean of q(x_{t-1} | x_t, x0) (Ho et al. 2020, eq. 7)."""
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        return coef1 * x0 + coef2 * x_t

    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, t: torch.Tensor, t_index: int, clip_denoised: bool = True):
        """One ancestral (DDPM) reverse-process step: x_t -> x_{t-1}."""
        pred_noise = model(x_t, t)
        x0_pred = self.predict_x0_from_eps(x_t, t, pred_noise)
        if clip_denoised:
            x0_pred = x0_pred.clamp(-1.0, 1.0)
        model_mean = self.q_posterior_mean(x0_pred, x_t, t)

        if t_index == 0:
            return model_mean
        posterior_var_t = self._extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def sample(
        self,
        model,
        shape,
        device,
        start_noise: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
        trajectory_every: int = 50,
        clip_denoised: bool = True,
    ):
        """Full ancestral reverse diffusion (DDPM): x_T ~ N(0, I) -> ... -> x_0."""
        x = start_noise.clone() if start_noise is not None else torch.randn(shape, device=device)
        trajectory = [x.detach().cpu()] if return_trajectory else None
        for t_index in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), t_index, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, t_index, clip_denoised=clip_denoised)
            if return_trajectory and (t_index % trajectory_every == 0 or t_index == 0):
                trajectory.append(x.detach().cpu())
        if return_trajectory:
            return x, trajectory
        return x

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        shape,
        device,
        ddim_steps: int,
        eta: float = 0.0,
        start_noise: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
        return_trajectory: bool = False,
        trajectory_every: int = 5,
    ):
        """DDIM reverse sampling (Song, Meng & Ermon 2020).

        Strides across a subsequence of ``ddim_steps`` timesteps using the
        same trained model and the same ``alphas_cumprod`` this scheduler
        was built with (whichever schedule that is) - not a separate
        diffusion implementation. ``eta=0.0`` is deterministic; ``eta=1.0``
        recovers ancestral-DDPM-like stochasticity.
        """
        step_indices = torch.linspace(0, self.timesteps - 1, ddim_steps, device=device).long()
        step_indices = torch.unique(step_indices, sorted=True)
        step_indices = step_indices.flip(0)

        x = start_noise.clone() if start_noise is not None else torch.randn(shape, device=device)
        trajectory = [x.detach().cpu()] if return_trajectory else None

        for i, t_index in enumerate(step_indices.tolist()):
            t = torch.full((shape[0],), t_index, device=device, dtype=torch.long)
            pred_noise = model(x, t)
            x0_pred = self.predict_x0_from_eps(x, t, pred_noise)
            if clip_denoised:
                x0_pred = x0_pred.clamp(-1.0, 1.0)

            alpha_bar_t = self.alphas_cumprod[t_index]
            if i + 1 < len(step_indices):
                alpha_bar_prev = self.alphas_cumprod[step_indices[i + 1]]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device)

            sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
            dir_xt = torch.sqrt((1 - alpha_bar_prev - sigma_t ** 2).clamp(min=0)) * pred_noise
            noise = torch.randn_like(x) if (eta > 0 and i + 1 < len(step_indices)) else torch.zeros_like(x)
            x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma_t * noise

            if return_trajectory and (i % trajectory_every == 0 or i == len(step_indices) - 1):
                trajectory.append(x.detach().cpu())

        if return_trajectory:
            return x, trajectory
        return x

    def generate(
        self,
        model,
        shape,
        device,
        sampler: str = "ddpm",
        ddim_steps: int = 50,
        ddim_eta: float = 0.0,
        clip_denoised: bool = True,
        start_noise: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
        trajectory_every: Optional[int] = None,
    ):
        """Single entry point that dispatches to the DDPM or DDIM sampler."""
        if sampler == "ddpm":
            return self.sample(
                model, shape, device, start_noise=start_noise,
                return_trajectory=return_trajectory, trajectory_every=trajectory_every or 50,
                clip_denoised=clip_denoised,
            )
        elif sampler == "ddim":
            return self.ddim_sample(
                model, shape, device, ddim_steps, eta=ddim_eta, start_noise=start_noise,
                clip_denoised=clip_denoised, return_trajectory=return_trajectory,
                trajectory_every=trajectory_every or 5,
            )
        else:
            raise ValueError(f"Unknown sampler: {sampler!r} (expected 'ddpm' or 'ddim')")
