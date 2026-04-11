# Fine-tuning Experiments (Unfrozen Encoder)

## Summary

Fine-tuning the ViT-B/16 encoder end-to-end with a low learning rate (5e-6) while training the AttentiveProbe(d=3) and MLP head at higher LRs. All runs use the corrected evaluation pipeline with ImageNet normalization.

Training uses DDP on 4x NVIDIA T4 (16 GB each) with batch_size=1 per GPU and gradient accumulation of 4 steps, for an effective batch size of 16. We compare 32 vs 64 slices to measure the impact of doubling slice count (100 slices OOMs with encoder gradients).

## Current Runs (with normalization fix)

| Run | Encoder Init | Probe | Head | Slices | Val AUC | Test AUC | Status |
|-----|-------------|-------|------|--------|---------|----------|--------|
| U1 | Random→SSL ep11 | d=3 | MLP | 64 | -- | pending | queued |
| U2 | ImageNet→SSL ep32 | d=3 | MLP | 64 | -- | pending | queued |
| U3 | Random→SSL ep11 | d=3 | MLP | 32 | -- | pending | planned |
| U4 | ImageNet→SSL ep32 | d=3 | MLP | 32 | -- | pending | planned |

## Configuration

| Parameter | Value |
|-----------|-------|
| Encoder LR | 5e-6 |
| Probe LR | 1e-4 |
| Head LR | 1e-3 |
| Weight decay | 0.01 |
| Batch size / GPU | 1 |
| Gradient accumulation | 4 |
| Effective batch size | 16 (1 x 4 GPUs x 4 accum) |
| Epochs | 25 |
| Patience | 5 |
| Warmup | 3 epochs |
| LR schedule | Cosine with warmup |
| Probe depth | 3 blocks |
| Head type | MLP (hidden=256, dropout=0.1) |
| AMP | fp16 autocast |
| GPUs | 4x T4 16 GB (DDP) |

## Design Choices

- **d=3 only**: We tested d=2 vs d=3 in earlier experiments and found d=3 is marginally better. All current runs use d=3 to reduce experiment count.
- **ep32 checkpoint only**: We confirmed that ep32 is the best pretraining checkpoint. Downstream AUC degrades monotonically from ep32 → ep99 (0.774 → 0.685 in old frozen probe experiments).
- **32 vs 64 slices**: Key comparison — does doubling the number of OCT slices per volume improve classification? Memory limits: 64 slices is the max for unfrozen ViT-B/16 on T4 16GB.

## Previous Results (without normalization fix — for reference)

These used the old evaluation pipeline without ImageNet normalization. The normalization mismatch primarily affected the frozen probe (-10 AUC points), but may also impact unfrozen results since early encoder updates start from incorrectly-distributed inputs.

| Run | Encoder Init | Probe | Head | Slices | Val AUC | Test AUC |
|-----|-------------|-------|------|--------|---------|----------|
| U1 (old) | Random→SSL ep11 | d=2 | Linear | 32 | 0.819 | N/A* |
| U2 (old) | Random→SSL ep11 | d=3 | Linear | 64 | 0.815 | N/A* |
| U3 (old) | ImageNet→SSL ep32 | d=2 | MLP | 32 | 0.826 | 0.828 |
| U4 (old) | ImageNet→SSL ep32 | d=2 | MLP | 64 | 0.832 | 0.829 |
| U5 (old) | ImageNet→SSL ep32 | d=3 | MLP | 32 | 0.828 | 0.829 |
| U6 (old) | ImageNet→SSL ep32 | d=3 | MLP | 64 | 0.832 | 0.829 |

*\*N/A: DDP teardown crash during test evaluation (older code, since fixed).*

Old findings: All ImageNet-init unfrozen configs clustered at 0.828-0.829 test AUC — neither deeper probe nor more slices helped. The corrected runs will show whether this ceiling changes with proper normalization.
