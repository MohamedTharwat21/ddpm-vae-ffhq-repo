"""Reusable inference-time estimation: VAE reconstruct/generate timing and
DDPM/DDIM sampler timing.

Generalized from the timing snippets supplied for the VAE ("INFERENCE TIME
ESTIMATION": reconstruction batch time, generation batch time, throughput,
estimated totals) and for the pixel/latent DDPM ("Benchmark DDPM and DDIM
sampling time for one batch": warmup runs, measured runs, CUDA
synchronization for accurate GPU timing, seconds/batch and seconds/image).
Not a one-off script tied to a single notebook - both entry points are
plain functions any training/evaluation script in this project can call.
"""
from __future__ import annotations

import statistics
import time
from typing import Callable

import torch

__all__ = ["benchmark_vae_inference", "benchmark_sampler"]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_vae_inference(
    model,
    input_batch: torch.Tensor,
    latent_batch: torch.Tensor,
    device: torch.device,
    repeats: int = 20,
):
    """Times VAE reconstruction (encoder -> mean z -> decoder) and unconditional
    generation (decoder only), matching the supplied VAE timing methodology.

    Returns a dict with per-batch seconds, images/sec, and the batch sizes
    used, for both paths - the caller multiplies by however many images it
    wants an estimate for (see the README's Inference Time tables for how
    this was used to report "estimated N images" figures).
    """
    model.eval()

    def reconstruct_batch(batch):
        mu, _ = model.encoder(batch)
        return model.decoder(mu)

    def time_inference(inference_fn, tensor):
        with torch.inference_mode():
            for _ in range(3):
                inference_fn(tensor)
            _synchronize(device)
            start = time.perf_counter()
            for _ in range(repeats):
                inference_fn(tensor)
            _synchronize(device)
        return (time.perf_counter() - start) / repeats

    recon_seconds = time_inference(reconstruct_batch, input_batch)
    gen_seconds = time_inference(model.decoder, latent_batch)

    return dict(
        reconstruction_batch_seconds=recon_seconds,
        reconstruction_images_per_second=input_batch.size(0) / recon_seconds,
        generation_batch_seconds=gen_seconds,
        generation_images_per_second=latent_batch.size(0) / gen_seconds,
        batch_size=input_batch.size(0),
    )


def benchmark_sampler(
    generate_fn: Callable[[str], torch.Tensor],
    sampler_names,
    batch_size: int,
    device: torch.device,
    warmup_runs: int = 1,
    measured_runs: int = 3,
):
    """Times one or more samplers (e.g. "ddpm", "ddim") with CUDA
    synchronization for accurate GPU timing, matching the supplied
    DDPM-vs-DDIM benchmark methodology.

    Args:
        generate_fn(sampler_name): runs one full generation for that
            sampler and returns the output tensor (only used for timing;
            its value is discarded).
        sampler_names: iterable of sampler name strings to benchmark.

    Returns ``{sampler_name: {"mean_seconds": ..., "std_seconds": ...,
    "seconds_per_image": ...}}``.
    """
    results = {}
    for sampler_name in sampler_names:
        times_seconds = []
        for run_index in range(warmup_runs + measured_runs):
            _synchronize(device)
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = generate_fn(sampler_name)
            _synchronize(device)
            elapsed = time.perf_counter() - start_time
            if run_index >= warmup_runs:
                times_seconds.append(elapsed)
        mean_seconds = statistics.mean(times_seconds)
        std_seconds = statistics.stdev(times_seconds) if len(times_seconds) > 1 else 0.0
        results[sampler_name] = dict(
            mean_seconds=mean_seconds,
            std_seconds=std_seconds,
            seconds_per_image=mean_seconds / batch_size,
        )
    return results
