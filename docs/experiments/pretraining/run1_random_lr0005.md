# Pretraining Run 1: Random Init, LR=0.0005

## Summary

First attempt. LR too high for OCT data. Model learned during warmup (ep1-11) but destabilized at peak LR. Early stopped at ep26.

## Config

| Parameter | Value |
|-----------|-------|
| Architecture | ViT-B/16 |
| Initialization | Random |
| Learning Rate | 0.0005 |
| Warmup Epochs | 15 |
| Early Stopping Patience | 15 |
| Batch Size | 64/GPU |
| Gradient Accumulation | 2 |
| Effective Batch Size | 512 |
| Dataset | 600K OCT slices |

## Training Log Excerpt

| Epoch | train_loss | val_loss | cos_sim | rep_diversity |
|-------|-----------|---------|---------|---------------|
| 1 | 0.1298 | 0.2152 | 0.6971 | 0.81 |
| 11 | 0.2073 | 0.2081 | 0.72 | 0.69 |
| 15 | 0.2153 | 0.2187 | 0.68 | 0.68 |
| 25 | 0.3047 | 0.3059 | 0.58 | 0.50 |

## Key Observations

- LR=0.0005 too aggressive. OCT images less diverse than ImageNet, leading to correlated gradients and an effective LR that is too high.
- Best checkpoint at ep11 (during warmup phase, before peak LR destabilized training).
- Loss and cos_sim both degraded steadily after warmup ended at ep15.
