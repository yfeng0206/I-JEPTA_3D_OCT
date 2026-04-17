# Frozen Probe Downstream Experiments

## Approach

These experiments evaluate frozen I-JEPA ViT-B/16 encoders on OCT-based glaucoma classification using the FairVision dataset. The pipeline is: **frozen ViT-B/16 encoder** (no gradient) produces per-patch embeddings from each OCT slice, which are **mean-pooled** across slices, then fed through a learnable **AttentiveProbe** (multi-head cross-attention, configurable depth), and finally a **classification head** (linear or MLP) trained with **BCEWithLogitsLoss**. Only the probe and head parameters are updated during training; the encoder remains frozen throughout.

All runs use 100 OCT slices per eye, 256x256 crop size, patch size 16, batch size 64, and cosine LR schedule with 3-epoch warmup.

## Comparison Table

Pretraining in progress — results will be updated once complete.

### Linear probe (d=2, linear head — matches I-JEPA paper protocol)

| Run ID | Encoder Init | Checkpoint | Probe Depth | Head Type | Slices | Val AUC | Test AUC |
|--------|-------------|------------|-------------|-----------|--------|---------|----------|
| frozen_random_linear_ep25 | Random→SSL | ep25 | 2 | linear | 100 | pending | pending |
| frozen_random_linear_ep50 | Random→SSL | ep50 | 2 | linear | 100 | pending | pending |
| frozen_random_linear_ep75 | Random→SSL | ep75 | 2 | linear | 100 | pending | pending |
| frozen_random_linear_ep100 | Random→SSL | ep100 | 2 | linear | 100 | pending | pending |
| frozen_imagenet_linear_ep25 | ImageNet→SSL | ep25 | 2 | linear | 100 | pending | pending |
| frozen_imagenet_linear_ep50 | ImageNet→SSL | ep50 | 2 | linear | 100 | pending | pending |
| frozen_imagenet_linear_ep75 | ImageNet→SSL | ep75 | 2 | linear | 100 | pending | pending |
| frozen_imagenet_linear_ep100 | ImageNet→SSL | ep100 | 2 | linear | 100 | pending | pending |

### MLP probe (d=3, MLP head)

| Run ID | Encoder Init | Checkpoint | Probe Depth | Head Type | Slices | Val AUC | Test AUC |
|--------|-------------|------------|-------------|-----------|--------|---------|----------|
| frozen_random_mlp_ep25 | Random→SSL | ep25 | 3 | mlp | 100 | pending | pending |
| frozen_random_mlp_ep50 | Random→SSL | ep50 | 3 | mlp | 100 | pending | pending |
| frozen_random_mlp_ep75 | Random→SSL | ep75 | 3 | mlp | 100 | pending | pending |
| frozen_random_mlp_ep100 | Random→SSL | ep100 | 3 | mlp | 100 | pending | pending |
| frozen_imagenet_mlp_ep25 | ImageNet→SSL | ep25 | 3 | mlp | 100 | pending | pending |
| frozen_imagenet_mlp_ep50 | ImageNet→SSL | ep50 | 3 | mlp | 100 | pending | pending |
| frozen_imagenet_mlp_ep75 | ImageNet→SSL | ep75 | 3 | mlp | 100 | pending | pending |
| frozen_imagenet_mlp_ep100 | ImageNet→SSL | ep100 | 3 | mlp | 100 | pending | pending |

## Key Design Choices

- **Two head types**: We evaluate both a linear head (d=2 probe, 14.3M trainable params) and an MLP head (d=3 probe, 21.5M trainable params). The linear probe follows the original I-JEPA paper evaluation protocol for comparability with the literature.
- **Checkpoint sweep**: We evaluate at ep25, ep50, ep75, and ep100 to track how representation quality evolves during pretraining. I-JEPA loss is not a reliable signal for downstream quality.
- **100 slices**: Frozen probe uses all 100 slices per volume (no GPU memory constraint since encoder is not backpropagated).
