# Pretraining Run 3: Random Init, Resume from Ep9 (Converged)

## Summary

Resumed from Run 2's ep9 with the early stopping fix applied. Converged at ep11, plateaued through ep18 when an NCCL timeout crash occurred (blocking blob upload). Best checkpoint was saved before the crash. This is the best random-init encoder.

## Config

| Parameter | Value |
|-----------|-------|
| Architecture | ViT-B/16 |
| Initialization | Resume from Run 2 ep9 |
| Learning Rate | 0.00025 |
| Warmup Epochs | 5 |
| Early Stopping Patience | 8 |
| Batch Size | 64/GPU |
| Gradient Accumulation | 2 |
| Effective Batch Size | 512 |

## Training Log Excerpt

| Epoch | train_loss | val_loss | cos_sim | rep_diversity |
|-------|-----------|---------|---------|---------------|
| 10 | 0.1596 | 0.1586 | 0.81 | 0.80 |
| 11 | 0.1574 | 0.1586 | 0.82 | 0.73 |
| 15 | 0.1591 | 0.1598 | 0.83 | 0.72 |
| 18 | 0.1618 | 0.1616 | 0.79 | 0.70 |

## Key Observations

- Best checkpoint: ep11, val_loss=0.1586.
- Training plateaued from ep11 onward with minimal loss improvement and slowly declining rep_diversity.
- Crash at ep18 was caused by NCCL timeout due to a blocking blob upload operation, not a training issue.
- This encoder was used for downstream runs F1, F2, U1, U2.
