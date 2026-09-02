# Pixel DDPM - Linear Schedule + EMA + Logit-Normal Timestep Sampling

**Status: PLACEHOLDER - no data supplied.**

This experiment directory exists because the project's experiment matrix
calls for a linear-schedule counterpart to
[`experiments/pixel_ddpm/cosine_ema_logit_v2_100`](../cosine_ema_logit_v2_100)
(same EMA + logit-normal timestep sampling setup, but with
`BETA_SCHEDULE="linear"` instead of `"cosine"`), so that the effect of the
beta schedule itself can eventually be isolated while holding EMA and
timestep sampling constant.

The corresponding upload for this experiment
(`linear__ema.txt`) was an **empty file** - no CONFIG dict, no training
history, no evaluation results, and no checkpoint reference were supplied
for this experiment anywhere in the source material.

## What exists in this repository for this experiment

- `config/` - [`configs/ddpm/linear_ema_logit.yaml`](../../../configs/ddpm/linear_ema_logit.yaml),
  which is **derived**, not transcribed: it inherits every setting from
  `configs/ddpm/cosine_ema_logit_v2_100.yaml` via `extends:` and overrides
  only `diffusion.schedule: linear`. This is the smallest reasonable
  engineering decision to make the experiment reproducible in principle,
  consistent with every other config in this project deriving from an
  actually-run configuration wherever possible.
- `checkpoints/`, `logs/`, `samples/`, `reconstructions/`, `evaluation/`,
  `timing/` - all empty except for `.gitkeep`, exactly as they are for
  every other experiment, pending the user manually adding real artifacts.

## What does NOT exist

- No training history CSV.
- No evaluation results (FID, IS, reconstruction metrics) - every mention
  of this experiment in `README.md` is marked **N/A**.
- No inference-timing benchmark.
- No checkpoint.
- No generated-image samples.

If real training/evaluation data for this experiment becomes available,
replace this README and populate `logs/`, `evaluation/`, and `samples/`
following the same structure used for
`experiments/pixel_ddpm/cosine_ema_logit_v2_100`.
