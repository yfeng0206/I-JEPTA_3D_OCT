# P1a/P1b: DINOv3 ViT-B/16 Encoder Probe

## Goal

Test whether a general-purpose vision foundation model (DINOv3, trained on 1.7B web images) produces better OCT features than our I-JEPA encoder (trained on 600K FairVision OCT slices), using the **exact same downstream pipeline**.

**Key question:** If DINOv3 frozen > I-JEPA frozen (0.774), our I-JEPA pretraining on FairVision isn't adding value. If DINOv3 unfrozen > 0.829, stronger pretrained features yield better fine-tuning.

## Architecture

All ablation experiments use a **shared downstream pipeline** — only the encoder changes. This ensures fair comparison.

### Shared pipeline (same for all experiments)

```
Per-slice encoding:
  OCT slice (256x256x3)
    → [ENCODER] → (N_patches, 768) patch tokens
    → Mean-pool across patches → (768,) per slice

Volume-level aggregation:
  Stack slices → (num_slices, 768)
    → [AttentiveProbe] d=3: learnable [CLS] + 3 transformer blocks (768-d, 12 heads)
    → CLS output → (768,)

Classification:
  (768,) → [MLP Head]: LayerNorm → Linear(768→256) → GELU → Dropout → Linear(256→1)
  → BCEWithLogitsLoss → P(glaucoma)
```

**Why mean-pool?** Each slice produces 256 patch tokens (16x16 grid from 256x256 image with patch_size=16). We mean-pool to get one 768-d vector per slice, then use the AttentiveProbe for **slice-level cross-attention** — learning which slices are most informative for glaucoma (e.g., slices near the optic nerve head vs peripheral slices). This is the same design as our I-JEPA experiments.

**Why AttentiveProbe for slice-level attention?** OCT volumes have 100-200 B-scans but not all slices are equally informative. The AttentiveProbe uses a learnable [CLS] token that attends to all slice embeddings, learning to weight diagnostically relevant slices more heavily. This is adapted from the I-JEPA evaluation protocol.

### DINOv3 encoder specifics

| Property | DINOv3 ViT-B/16 | Our I-JEPA ViT-B/16 |
|----------|-----------------|---------------------|
| Architecture | ViT-B/16 | ViT-B/16 |
| Parameters | 86M | 86M |
| Hidden dim | 768 | 768 |
| Layers | 12 | 12 |
| Attention heads | 12 | 12 |
| Patch size | 16 | 16 |
| Output per slice | (256, 768) → mean → (768,) | (256, 768) → mean → (768,) |
| Pretrain data | LVD-1.7B (web images) | FairVision 600K (OCT slices) |
| Pretrain method | Self-distillation + Gram anchoring | I-JEPA (masked representation prediction) |
| Paper | [arxiv 2508.10104](https://arxiv.org/abs/2508.10104) | [arxiv 2301.08243](https://arxiv.org/abs/2301.08243) |
| GitHub | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | This repo |

The architectures are **identical** — same ViT-B/16, same patch size, same output shape. The only difference is what data they were pretrained on and how. This makes it a clean comparison of pretraining strategy.

## Experiments

| Run | Mode | Slices | Head | Epochs | Multi-GPU | Status |
|-----|------|--------|------|--------|-----------|--------|
| P1a | Frozen probe | 100 | MLP | 100 | DataParallel (4 GPU encoding) | pending |
| P1b | Unfrozen fine-tune | 32 | MLP | 25 | DDP torchrun (4 GPU) | pending |

### P1a: Frozen probe
- Encoder frozen, features precomputed once and cached to disk
- Uses all 4 GPUs via `DataParallel` during feature encoding (~4x speedup)
- Probe training on cached tensors (fast, ~3 min/epoch)
- 100 epochs, WD=0, cosine LR with 3-epoch warmup (matching SSL eval literature)

### P1b: Unfrozen fine-tune
- Encoder unfrozen with low LR (5e-6), probe at 1e-4, head at 1e-3
- DDP with torchrun on 4 GPUs, batch=1/GPU, accum=4, effective batch=16
- 25 epochs, patience=5, WD=0.01

## Comparison targets

| Method | Encoder | Pretrain | Test AUC |
|--------|---------|----------|----------|
| I-JEPA frozen | ViT-B/16 | ImageNet→I-JEPA 600K OCT | 0.774 |
| I-JEPA unfrozen | ViT-B/16 | ImageNet→I-JEPA 600K OCT | 0.829 |
| **DINOv3 frozen (P1a)** | **ViT-B/16** | **LVD-1.7B web images** | **pending** |
| **DINOv3 unfrozen (P1b)** | **ViT-B/16** | **LVD-1.7B web images** | **pending** |

## Setup

1. Request access: [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
2. Set `HF_TOKEN` environment variable
3. Run: `bash ablation/dinov3_probe/run_dinov3.sh`

## Files

```
ablation/dinov3_probe/
  README.md              # This file
  eval_dinov3.py         # Main evaluation script (frozen + unfrozen)
  run_dinov3.sh          # AML/local entry point
  configs/
    frozen_d3_s100.yaml  # Frozen probe config
```
