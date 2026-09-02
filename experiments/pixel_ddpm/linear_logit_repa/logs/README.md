# Logs - Pixel DDPM, Linear + Logit-Normal + REPA

No per-epoch training history CSV was supplied for this experiment.

Training was stopped after 16 epochs due to poor qualitative results
attributed to the REPA auxiliary loss (see `../evaluation/results.json`
for the final-epoch loss snapshot: train MSE 0.00978, val MSE 0.00973,
REPA loss 0.57).

Compare against `experiments/pixel_ddpm/linear_logit/logs/training_history.csv`,
which is the same base config (linear schedule, logit-normal timestep
sampling) trained to 70 epochs **without** REPA.
