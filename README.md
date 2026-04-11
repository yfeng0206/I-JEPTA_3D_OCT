# I-JEPA for FairVision OCT Glaucoma Classification

Self-supervised pretraining using [I-JEPA](https://github.com/facebookresearch/ijepa) (Assran et al., CVPR 2023) on [Harvard FairVision](https://github.com/Harvard-Ophthalmology-AI-Lab/FairVision) OCT data for binary glaucoma classification.

## Results Summary

Pretraining in progress — results will be updated once complete.

| Method | Encoder Init | Encoder | Slices | Probe | Head | Test AUC |
|--------|-------------|---------|--------|-------|------|----------|
| I-JEPA frozen d=3 | Random→SSL | ViT-B/16 frozen | 100 | 3 blocks | MLP | pending |
| I-JEPA frozen d=3 | ImageNet→SSL | ViT-B/16 frozen | 100 | 3 blocks | MLP | pending |
| I-JEPA unfrozen d=3 | Random→SSL | ViT-B/16 fine-tune | 32 | 3 blocks | MLP | pending |
| I-JEPA unfrozen d=3 | Random→SSL | ViT-B/16 fine-tune | 64 | 3 blocks | MLP | pending |
| I-JEPA unfrozen d=3 | ImageNet→SSL | ViT-B/16 fine-tune | 32 | 3 blocks | MLP | pending |
| I-JEPA unfrozen d=3 | ImageNet→SSL | ViT-B/16 fine-tune | 64 | 3 blocks | MLP | pending |

## Quick Links

| | |
|---|---|
| **Experiments** | [All experiments](docs/experiments) |
| **Pretraining** | [Pretraining runs](docs/experiments/pretraining) (Random-init, ImageNet-init, loss curves) |
| **Frozen Probe** | [Frozen probe eval](docs/experiments/downstream/frozen) (6 runs: Random vs ImageNet, multiple epochs) |
| **Fine-tuning** | [Unfrozen encoder eval](docs/experiments/downstream/unfrozen) (6 runs: d=2/3, 32/64 slices) |
| **Architecture** | [Model architecture details](docs/architecture.md) |
| **Lessons Learned** | [Mistakes & fixes log](docs/lessons_learned.md) |

## Motivation

I-JEPA learns representations directly from unlabeled OCT data through masked prediction in representation space. No hand-crafted augmentations are needed. We implement two approaches, with patch-level as the primary approach.

### Patch-level I-JEPA (primary)

Standard I-JEPA applied to individual 256x256 OCT slices (600K images from 6K volumes x 100 slices). The encoder learns within-slice spatial features by predicting masked patch representations from context patches. See [architecture details](docs/architecture.md).

### Slice-level I-JEPA (failed)

I-JEPA applied to sequences of 32 ConvNeXt slice features per volume. Collapsed within 1-2 epochs due to insufficient token diversity (adjacent OCT slices produce nearly identical features). See [lessons learned](docs/lessons_learned.md).

## Dataset

Harvard FairVision Glaucoma subset: 10,000 subjects (6K train / 1K val / 3K test), each with 200x200x200 OCT B-scan volume. Binary labels: glaucoma (1) or not (0). Available on [HuggingFace](https://huggingface.co/datasets/ming0100/Harvard_FairVision).

## Project Structure

```
src/
  models/vision_transformer.py    # ViT encoder, predictor, slice-level variants
  masks/multiblock.py             # 2D block masking (patch-level)
  masks/slice_mask.py             # 1D contiguous masking (slice-level)
  datasets/oct_slices.py          # Individual slice dataset (600K images)
  datasets/oct_volumes.py         # Volume dataset (6K volumes)
  utils/schedulers.py             # Warmup cosine LR, cosine WD
  train_patch.py                  # Patch-level I-JEPA pretraining
  train_slice.py                  # Slice-level I-JEPA pretraining
  eval_downstream.py              # Downstream classification
  helper.py                       # Model init, optimizer, checkpoint I/O

configs/                          # Training configs
scripts/                          # Entry point shell scripts
docs/
  experiments/                    # Detailed experiment logs & results
    pretraining/                  # Pretraining run details
    downstream/frozen/            # Frozen probe experiments
    downstream/unfrozen/          # Fine-tuning experiments
  architecture.md                 # Model architecture details
  lessons_learned.md              # Mistakes & fixes
results/                          # Training curves, plots, raw data
```

## References

- Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" ([paper](https://arxiv.org/abs/2301.08243), [code](https://github.com/facebookresearch/ijepa))
- Luo et al., "Harvard Ophthalmology AI-Lab FairVision Dataset" ([paper](https://arxiv.org/abs/2310.02492), [code](https://github.com/Harvard-Ophthalmology-AI-Lab/FairVision))
