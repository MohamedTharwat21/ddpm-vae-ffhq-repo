# Logs - Latent DDPM (LDM), Cosine Schedule

No per-epoch training history CSV was supplied for this experiment.

This run hit the x0-clipping / reconstruction bug documented in
`configs/ldm/cosine.yaml` and in README.md's "Failed / Abandoned
Experiments" section: with `clip_denoised` not applied correctly against
the cosine schedule's extreme early-timestep SNR, generated (and
intermediate reconstructed) latents were pushed outside their expected
range, decoding through the frozen VAE into high-contrast, saturated
images.

Training was stopped after approximately 15 epochs (~20 minutes wall
clock) rather than run to completion, once the saturation problem was
observed.

## Final loss snapshot (approximate epoch 15)

| Metric | Value |
|---|---|
| Epochs completed | ~15 |
| Wall-clock time | ~20 minutes |
| Train MSE | ≈ 0.33 |
| Val MSE | ≈ 0.32 |

These MSE values are on the latent (not pixel) reconstruction target and
are **not comparable** to the pixel-space DDPM loss values elsewhere in
this repository.

No FID/IS evaluation or inference-timing benchmark was run against this
checkpoint - it is documented as a failed/historical experiment only.
See the image placeholder at `reports/figures/ldm/cosine/high_contrast_examples.png`
for the qualitative failure mode once the user supplies generated samples.

`configs/ldm/cosine.yaml` sets `clip_denoised: true` as the recommended
fix for any future re-run of this experiment - it does **not** reproduce
the historical bug by default. The bug itself is documented here and in
README.md as a historical fact about the run that actually happened, not
reintroduced into the default configuration.
