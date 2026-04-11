# Frozen Probe Downstream Experiments

## Approach

These experiments evaluate frozen I-JEPA ViT-B/16 encoders on OCT-based glaucoma classification using the FairVision dataset. The pipeline is: **frozen ViT-B/16 encoder** (no gradient) produces per-patch embeddings from each OCT slice, which are **mean-pooled** across slices, then fed through a learnable **AttentiveProbe** (multi-head cross-attention, configurable depth), and finally a **classification head** (linear or MLP) trained with **BCEWithLogitsLoss**. Only the probe and head parameters are updated during training; the encoder remains frozen throughout.

All runs use 100 OCT slices per eye, 256x256 crop size, patch size 16, batch size 64, and cosine LR schedule with 3-epoch warmup.

## Comparison Table

Pretraining in progress — results will be updated once complete.

| Run ID | Encoder Init | Probe Depth | Head Type | Slices | Val AUC | Test AUC |
|--------|-------------|-------------|-----------|--------|---------|----------|
| frozen_random_d3 | Random→SSL | 3 | mlp | 100 | pending | pending |
| frozen_imagenet_d3 | ImageNet→SSL | 3 | mlp | 100 | pending | pending |

## Key Design Choices

- **Probe depth d=3 only**: We tested d=2 vs d=3 and found d=3 marginally better.
- **Best checkpoint only**: Downstream AUC degrades with continued pretraining past the best epoch. We evaluate the best (lowest val loss) checkpoint.
- **MLP head**: MLP head outperforms linear head by ~4 points on frozen probe.
- **100 slices**: Frozen probe uses all 100 slices per volume (no GPU memory constraint since encoder is not backpropagated).
