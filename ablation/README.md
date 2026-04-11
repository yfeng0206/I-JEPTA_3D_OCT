# Ablation Studies & Encoder Benchmarking

Systematic experiments to find the best encoder and training strategy for OCT glaucoma classification. Each experiment answers a specific research question and uses the **same shared downstream pipeline** for fair comparison.

## Shared Downstream Pipeline

All experiments use this pipeline — only the encoder changes:

```
OCT Volume (200 B-scans)
  → Sample N slices (100 frozen / 32 unfrozen)
  → [ENCODER] per slice → (N_patches, 768) → mean-pool → (768,) per slice
  → Stack → (N, 768) slice embeddings
  → [AttentiveProbe] d=3: learnable [CLS] + 3 transformer blocks (slice-level attention)
  → [MLP Head] → P(glaucoma)
```

**Why this design:** The encoder produces per-slice features. Mean-pooling compresses each slice to a single vector. The AttentiveProbe then learns **which slices matter** for diagnosis via cross-attention with a learnable [CLS] token. This is consistent across all experiments so we isolate the effect of the encoder.

---

## Experiment Map

### Phase 1: Encoder Benchmarking
*Which pretrained encoder produces the best features for OCT?*

| ID | Experiment | Encoder | Pretrain Data | Details | Status |
|----|-----------|---------|---------------|---------|--------|
| [P1a](dinov3_probe/) | DINOv3 frozen probe | ViT-B/16 (86M) | LVD-1.7B web images | [details](dinov3_probe/) | **planned** |
| [P1b](dinov3_probe/) | DINOv3 unfrozen | ViT-B/16 (86M) | LVD-1.7B web images | [details](dinov3_probe/) | **planned** |
| P1c | RETFound frozen probe | ViT-L (300M) | 1.6M retinal images | — | planned |
| P1d | RETFound unfrozen | ViT-L (300M) | 1.6M retinal images | — | planned |

### Phase 2: Pretraining Data & Method
*Does the pretraining data or method matter more?*

| ID | Experiment | Question | Status |
|----|-----------|----------|--------|
| P2a | Kermany supervised → FairVision | Does supervised OCT pretraining beat self-supervised? | planned |
| P2b | Kermany I-JEPA → FairVision | Is I-JEPA on Kermany (84K labeled OCT) better than on FairVision? | planned |
| P2c | MAE pretrain on FairVision 600K | Is I-JEPA actually better than MAE on our data? (paper claims yes) | planned |

### Phase 3: Multi-view / 3D
*Does leveraging the 3D structure of OCT volumes help?*

| ID | Experiment | Question | Status |
|----|-----------|----------|--------|
| P3a | Multi-view inference (3 axes) | Does fusing B-scan + en-face + C-scan at test time help? | planned |
| P3b | Multi-view I-JEPA pretraining | Does pretraining on all 3 axes (1.8M images) improve the encoder? | planned |
| P3c | Multi-view pretrain + inference | Do pretraining and inference multi-view compound? | planned |

---

## Completed Results (Baselines)

| Method | Encoder (params) | Pretrain | Test AUC |
|--------|-----------------|----------|----------|
| **I-JEPA frozen** | **ViT-B/16 (86M)** | **Random → I-JEPA on 600K OCT** | **0.834** |
| I-JEPA unfrozen | ViT-B/16 (86M) | ImageNet → I-JEPA on 600K OCT | 0.829 |
| RETFound (literature) | MAE ViT-L (300M) | ImageNet → 1.6M retinal (MAE) | 0.91 |

## Priority Order

| # | Experiment | Time | Why first |
|---|-----------|------|-----------|
| 1 | [P1a/b: DINOv3](dinov3_probe/) | 1-2 days | Fastest signal — same arch, just swap weights |
| 2 | P1c/d: RETFound | 1-2 days | Ceiling check with OCT-specific model |
| 3 | P3a: Multi-view inference | 2-3 days | Free improvement, no retraining needed |
| 4 | P2a: Kermany supervised | 3-5 days | Test supervised vs self-supervised |
| 5 | P2c: MAE on FairVision | 1 week | Verify I-JEPA > MAE on our data |
| 6 | P3b/c: Multi-view pretraining | 2 weeks | Novel contribution for paper |
