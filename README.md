# I-JEPA for FairVision OCT Glaucoma Classification

Self-supervised pretraining using [I-JEPA](https://github.com/facebookresearch/ijepa) on FairVision OCT data, followed by fine-tuning for binary glaucoma classification. Builds on the SLIViT work in [SliViT_3D_OCT_Glaucoma](https://github.com/yfeng0206/SliViT_3D_OCT_Glaucoma).

Two approaches:
- **Patch-level**: standard I-JEPA on individual 256x256 OCT slices (192K images)
- **Slice-level**: I-JEPA on sequences of 32 slice features per volume (6K volumes), using frozen ConvNeXt from SLIViT as feature extractor

## Differences from Original I-JEPA

```
                          Original I-JEPA         Patch-level (ours)      Slice-level (ours)
------------------------  ----------------------  ----------------------  ----------------------
Dataset                   ImageNet 1.2M           192K OCT slices         6K OCT volumes
Image size                224x224                 256x256                 N/A (32 slice tokens)
Patch size                14 or 16                16                      N/A
Patch grid                14x14=196               16x16=256               32 tokens (1D)
Pos embedding             2D sinusoidal           2D sinusoidal           1D sinusoidal

Encoder model             ViT-H/14 (630M)         ViT-B/16 (86M)          SliceViT (85M)
Encoder dim               1280                    768                     768
Encoder depth             32                      12                      12
Encoder heads             16                      12                      12
Encoder head_dim          80                      64                      64

Predictor dim             384                     384                     384
Predictor depth           12                      6                       6
Predictor heads           16                      12                      12

Target blocks             4                       4                       4
Target scale              [0.15, 0.2]             [0.15, 0.2]             [0.1, 0.2]
Context scale             [0.85, 1.0]             [0.85, 1.0]             [0.75, 0.9]
Aspect ratio              [0.75, 1.5]             [0.75, 1.5]             N/A (1D)
Masking type              2D block                2D block                1D contiguous

Batch size (per GPU)      128                     32                      8
Effective batch           2048 (16 GPU)           128 (4 GPU)             32 (4 GPU)
Epochs                    300                     100-300                 100-200
LR                        0.001                   0.0005                  0.0003
Start LR                  0.0002                  0.0001                  0.00005
Final LR                  1e-6                    1e-6                    1e-6
Warmup                    40 epochs               15 epochs               10 epochs
Weight decay              0.04 -> 0.4             0.04 -> 0.4             0.04 -> 0.4
EMA momentum              [0.996, 1.0]            [0.996, 1.0]            [0.996, 1.0]
Mixed precision           bfloat16                fp16 (T4)               fp16 (T4)

Feature extractor         N/A                     N/A                     Frozen ConvNeXt (SLIViT)
Hardware                  16x A100 80GB           4x T4 16GB              4x T4 16GB
Training time             ~72 hours               ~8-25 hours             ~6-12 hours
```

### Why these changes

**Smaller encoder (ViT-B vs ViT-H)**: Original trains on 1.2M images. We have 192K (patch) or 6K (slice). ViT-H would massively overfit. ViT-B has 86M params, a reasonable ratio for our data.

**Shallower predictor (6 vs 12 layers)**: Predictor should be weaker than encoder to force the encoder to learn rich representations. With a 12-layer encoder, 6-layer predictor maintains the same 2:1 ratio as the original (32-layer encoder, 12-layer predictor).

**Lower learning rate**: Smaller batch size (128 vs 2048) means less gradient averaging, so we use a lower peak LR to compensate. Follows linear scaling: 0.001 * (128/2048) * ~8 = ~0.0005.

**Fewer warmup epochs**: Original warms up for 40/300 = 13% of training. We match: 15/100 = 15%.

**Slice-level masking is 1D**: With only 32 tokens, 2D block masking doesn't apply. We mask contiguous runs of slices (e.g., slices 12-16) since adjacent slices contain similar structures. Target scale [0.1, 0.2] = 3-6 slices masked, context scale [0.75, 0.9] = 24-29 slices visible.

**fp16 instead of bfloat16**: T4 GPUs don't support bfloat16. We use standard fp16 AMP.

## Pipeline

### Patch-level

```
Phase 1: I-JEPA pretraining (self-supervised, 192K slices, ~8-25 hrs)
    Individual OCT slices (256x256) -> ViT-B/16 encoder learns within-slice features

Phase 2: Downstream classification (supervised, 6K volumes, ~2-3 hrs)
    32 slices -> pretrained encoder -> 32 x 768-d -> ViT integrator -> MLP -> glaucoma logit
```

### Slice-level

```
Phase 1: I-JEPA pretraining (self-supervised, 6K volumes, ~6-12 hrs)
    Load frozen ConvNeXt from SLIViT -> encode 32 slices -> 32 x 768-d
    Slice-level ViT encoder learns cross-slice relationships via masked prediction

Phase 2: Downstream classification (supervised, 6K volumes, ~1-2 hrs)
    32 slices -> frozen ConvNeXt -> frozen/fine-tune slice encoder -> avg pool -> MLP -> logit
```

## Project structure

```
src/
  models/
    vision_transformer.py    Encoder + predictor (adapted from I-JEPA)
    feature_extractor.py     Frozen ConvNeXt loader for slice-level
  masks/
    multiblock.py            2D block masking for patch-level
    slice_mask.py            1D contiguous masking for slice-level
    utils.py                 apply_masks helper
  datasets/
    oct_slices.py            Individual slice loader (patch-level)
    oct_volumes.py           Volume loader (slice-level)
  utils/
    tensors.py               trunc_normal_, repeat_interleave_batch
    distributed.py           DDP helpers
    schedulers.py            WarmupCosine LR, CosineWD
    logging.py               CSVLogger, AverageMeter
  train_patch.py             Patch-level I-JEPA training
  train_slice.py             Slice-level I-JEPA training
  eval_downstream.py         Classification fine-tuning + evaluation
  helper.py                  Model init, optimizer init, checkpoint I/O
  transforms.py              Data augmentation (minimal)

configs/
  patch_vitb16_ep100.yaml    Patch-level, 100 epochs
  slice_ep100.yaml           Slice-level, 100 epochs
  downstream_patch.yaml      Downstream for patch approach
  downstream_slice.yaml      Downstream for slice approach

scripts/
  run_patch.sh               AML entry point for patch pretraining
  run_slice.sh               AML entry point for slice pretraining
  run_downstream.sh          AML entry point for downstream

docs/
  training_plan.md           Detailed training plan with memory/time estimates
```

## Data

Uses the same FairVision Glaucoma dataset as SLIViT:
- 6,000 train / 1,000 val / 3,000 test volumes
- Each volume: 200x200x200 OCT, stored as .npz
- Data stored in Azure Blob Storage (configure storage account/container in shell scripts)
- ConvNeXt checkpoint: SLIViT's Kermany-pretrained `feature_extractor.pth`

## References

- Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" ([paper](https://arxiv.org/abs/2301.08243), [code](https://github.com/facebookresearch/ijepa))
- Avram et al., "SLIViT" ([paper](https://pubmed.ncbi.nlm.nih.gov/38045283/), [code](https://github.com/cozygene/SLIViT))
