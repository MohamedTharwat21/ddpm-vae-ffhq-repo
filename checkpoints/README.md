# Checkpoints

This top-level directory is a convenience location for checkpoints you want available
across multiple experiments (for example, the VAE v2 checkpoint used as the frozen
pretrained encoder/decoder for both LDM experiments). **No checkpoints are included in
this repository** - they are large binary artifacts and are intentionally excluded
(see the repository root `.gitignore`).

## Where checkpoints actually belong

Per-experiment checkpoints belong under that experiment's own directory:

```text
experiments/<family>/<variant>/checkpoints/
```

for example `experiments/vae/v2/checkpoints/checkpoint_best.pt` or
`experiments/pixel_ddpm/cosine_ema_logit_v2_100/checkpoints/latest_model.pt`. Each of
those directories has its own `.gitkeep` and the corresponding config's
`evaluation.checkpoint_epoch` / this README's sibling files document which checkpoint
produced which reported number.

## Cross-experiment use (LDM's pretrained VAE)

`configs/ldm/linear.yaml` and `configs/ldm/cosine.yaml` reference
`model.vae_checkpoint_path` (default `null`, meaning auto-detect via
`ffhq_repo.data.pretrained_vae.find_vae_checkpoint` under `data.data_root_search`).
To reproduce the LDM experiments, place the trained VAE v2 checkpoint somewhere
reachable and either let auto-detection find it or set `vae_checkpoint_path`
explicitly - for example by copying it here as `checkpoints/vae_v2_checkpoint.pt` and
pointing the LDM configs at that path.