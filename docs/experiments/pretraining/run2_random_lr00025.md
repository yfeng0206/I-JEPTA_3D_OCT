# Pretraining Run 2: Random Init, LR=0.00025

## Summary

Reduced LR worked well, but early stopping bug killed the run at ep9. Epoch 1 had artificially low val_loss=0.1197 (EMA target not diverged yet), making it unbeatable by any subsequent epoch.

## Config

| Parameter | Value |
|-----------|-------|
| Architecture | ViT-B/16 |
| Initialization | Random |
| Learning Rate | 0.00025 |
| Warmup Epochs | 5 |
| Early Stopping Patience | 8 |
| Batch Size | 64/GPU |
| Gradient Accumulation | 2 |
| Effective Batch Size | 512 |

## Training Log Excerpt

Run terminated at ep9 due to early stopping triggered against the ep1 baseline.

## Key Observations

- LR=0.00025 showed stable learning; the run was killed prematurely by a bug, not by divergence.
- Early stopping must ignore pre-warmup epochs. At ep1, the EMA target encoder has not diverged from the online encoder yet, producing an artificially low val_loss=0.1197 that no later epoch can beat.
- Fix applied for subsequent runs: only count patience after warmup completes.
