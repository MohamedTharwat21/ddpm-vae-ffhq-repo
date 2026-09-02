# Logs - Pixel DDPM, Cosine + EMA + Logit-Normal, 65-Epoch Checkpoint

No separate per-epoch training history CSV is stored in this directory.

The 65-epoch checkpoint evaluated here (`best_model.pt`, internal `Saved epoch: 65`)
comes from the **same training run** as
[`experiments/pixel_ddpm/cosine_ema_logit_v2_100`](../../cosine_ema_logit_v2_100/logs/training_history.csv) -
the same config, the same U-Net, the same optimizer state lineage, just evaluated
at an earlier checkpoint before training continued on to epoch 100.

To see this checkpoint's position in the training curve, refer to
`experiments/pixel_ddpm/cosine_ema_logit_v2_100/logs/training_history.csv` and
look at the row for `epoch == 65`. Duplicating the first 65 rows of that CSV
into this directory was avoided to prevent two copies of the same data from
silently drifting out of sync.

See `../evaluation/results.json` in this directory for the epoch-65-specific
evaluation results (FID/IS under both DDPM and DDIM sampling, plus timing).
