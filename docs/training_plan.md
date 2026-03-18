# I-JEPA Training Plan for FairVision OCT Glaucoma Classification

## Overview

Two I-JEPA self-supervised pretraining approaches on FairVision OCT data, followed by fine-tuning for binary glaucoma classification.

- **Slice-level**: I-JEPA predicts masked slices from context slices within a volume. Learns cross-slice relationships. Downstream: add MLP head.
- **Patch-level**: I-JEPA predicts masked patches within individual 2D slices. Learns within-slice spatial features. Downstream: add ViT integrator + MLP head.

---

## Slice-level I-JEPA

### Pretraining (self-supervised, 6K volumes, ~6-12 hours)

```
Input: OCT volume (32 slices, each 256x256)

Step 1: Extract per-slice features (frozen feature extractor)

    slice_1  --> [Frozen ConvNeXt or ViT] --> 768-d --+
    slice_2  --> [Frozen ConvNeXt or ViT] --> 768-d   |
    ...                                                +--> 32 x 768-d
    slice_32 --> [Frozen ConvNeXt or ViT] --> 768-d --+

Step 2: Slice-level masking

    32 slice tokens --> mask ~5 slices (targets)
                    --> keep ~27 slices (context)

Step 3: Encode + Predict

    Context Encoder (slice-level ViT)        Target Encoder (EMA copy)
    In:  ~27 tokens (768-d each)             In:  32 tokens (768-d each)
    Out: ~27 x 768-d                         Out: 32 x 768-d
         |                                        |
         v                                        v
    Predictor (narrow ViT)                   Target features at masked slices
    In:  27 context + 5 mask tokens          Out: ~5 x 768-d
    Out: ~5 x 768-d (predicted)                   |
         |                                        |
         +-----> Smooth L1 Loss <-----------------+

Output: Pretrained slice-level encoder (target encoder weights)
        Has cross-slice attention baked in
```

### Downstream classification (supervised, 6K labeled, ~1-2 hours)

```
Input: OCT volume (32 slices)

    slice_1  --> [Frozen feature extractor] --> 768-d --+
    ...                                                  +--> 32 x 768-d
    slice_32 --> [Frozen feature extractor] --> 768-d --+

    32 x 768-d --> [Frozen/fine-tune slice encoder] --> 32 x 768-d

               --> Average Pool --> 768-d

               --> MLP Head: LayerNorm(768) + Linear(768, 1) --> logit

Loss: BCEWithLogitsLoss
Trained: MLP only (~1.5K params)
```

---

## Patch-level I-JEPA

### Pretraining (self-supervised, 192K slices, ~8-25 hours)

```
Input: Single OCT slice (256x256, 3-channel)
Dataset: 6K volumes x 32 slices = 192,000 images

    Patchify: 256x256 --> 256 patches (16x16 each)

    Multi-block masking:
      4 target blocks (~38 patches total)
      1 context block (~218 patches, targets removed)

    Context Encoder (ViT-B/16)               Target Encoder (ViT-B/16, EMA)
    In:  ~218 patches (768-d each)           In:  256 patches (768-d each)
    Out: 218 x 768-d                         Out: 256 x 768-d
         |                                        |
         v                                        v
    Predictor (narrow ViT)                   Target features at masked patches
    In:  218 context + 38 mask tokens        Out: ~38 x 768-d
    Out: ~38 x 768-d (predicted)                  |
         |                                        |
         +-----> Smooth L1 Loss <-----------------+

Output: Pretrained ViT-B/16 encoder (target encoder weights)
        Understands within-slice spatial features
```

### Downstream classification (supervised, 6K labeled, ~2-3 hours)

```
Input: OCT volume (32 slices)

    slice_1  --> [Pretrained ViT-B/16] --> avg pool --> 768-d --+
    slice_2  --> [Pretrained ViT-B/16] --> avg pool --> 768-d   |
    ...                                                          +--> 32 x 768-d
    slice_32 --> [Pretrained ViT-B/16] --> avg pool --> 768-d --+

    32 x 768-d
         |
         v
    ViT Integrator (random init)
    In:  33 tokens (32 slices + 1 CLS)
    Out: CLS 768-d
         |
         v
    MLP Head
    LayerNorm(768) + Linear(768, 1) --> logit

Loss: BCEWithLogitsLoss
Trained: ViT integrator + MLP (~23M params)
Encoder: frozen or light fine-tune (lr=1e-6)
```

---

## Component Details

### ViT-B/16 Encoder (patch-level, used in both approaches)

```
Property                Value
----------------------  ------------------------------
Input                   256x256 image (3-channel)
Patch size              16x16
Num patches             256 (16x16 grid)
Embed dim               768
Depth                   12 layers
Num heads               12
Head dim                64
MLP dim                 3072 (768 x 4)
Positional embedding    2D sinusoidal (fixed)
Output                  256 x 768-d tokens (no CLS)
Parameters              86M
Weights (fp32)          344 MB
```

### Target Encoder (EMA copy of context encoder)

```
Property                Value
----------------------  ------------------------------
Architecture            Identical to context encoder
Parameters              Same (no gradients, no optimizer)
Weights                 344 MB (fp32 only)
Update rule             w_t = m * w_t + (1-m) * w_context
Momentum schedule       0.996 -> 1.0 (linear ramp)
```

### Patch-level Predictor

```
Property                Value
----------------------  ------------------------------
Input                   ~218 context tokens (proj to 384-d)
                        + ~38 mask tokens (384-d)
Input projection        Linear(768, 384)
Embed dim               384
Depth                   6 layers
Num heads               12
Head dim                32
MLP dim                 1536 (384 x 4)
Output projection       Linear(384, 768)
Output                  ~38 x 768-d predicted
Parameters              11M
Weights (fp32)          44 MB
```

### Slice-level Encoder (slice approach only)

```
Property                Value
----------------------  ------------------------------
Input                   32 slice tokens (each 768-d)
Embed dim               768
Depth                   12 layers
Num heads               12
Head dim                64
MLP dim                 3072
Positional embedding    1D sinusoidal (32 positions)
Output                  32 x 768-d (cross-slice attn)
Parameters              85M
Weights (fp32)          340 MB
```

### Slice-level Predictor (slice approach only)

```
Property                Value
----------------------  ------------------------------
Input                   ~27 context + ~5 mask tokens
                        projected to 384-d
Input projection        Linear(768, 384)
Embed dim               384
Depth                   6 layers
Num heads               12
Output projection       Linear(384, 768)
Output                  ~5 x 768-d predicted
Parameters              11M
Weights (fp32)          44 MB
```

### ViT Integrator (patch approach downstream only)

```
Property                Value
----------------------  ------------------------------
Input                   32 slice tokens (768-d) + 1 CLS
Embed dim               768
Depth                   5 layers
Num heads               12
Head dim                64
MLP dim                 3072
Output                  CLS token 768-d
Parameters              23M
Weights (fp32)          92 MB
```

### MLP Classification Head (both approaches)

```
Property                Value
----------------------  ------------------------------
Input                   768-d (CLS token or avg pool)
Architecture            LayerNorm(768) + Linear(768, 1)
Output                  1 logit
Parameters              1,537
```

---

## Memory Budgets

All numbers per GPU (T4 16GB), fp16 mixed precision.

### Slice-level I-JEPA Pretraining

Model weights:

```
Component                                    Params    Memory
-------------------------------------------  --------  --------
Frozen feature extractor (ConvNeXt)          28M       112 MB
Slice-level context encoder                  85M       510 MB
Slice-level target encoder (EMA, no grad)    85M       340 MB
Slice-level predictor                        11M        66 MB
-------------------------------------------  --------  --------
Total                                        209M     1028 MB
```

Optimizer (AdamW, encoder + predictor): 96M params = 768 MB

Total by batch size:

```
Batch/GPU  Weights   Optimizer  Activations  Gradients  Total     Fits T4?
---------  --------  ---------  -----------  ---------  --------  --------
2          1028 MB   768 MB      30 MB        384 MB    ~2.2 GB   Yes
4          1028 MB   768 MB      60 MB        384 MB    ~2.2 GB   Yes
8          1028 MB   768 MB     120 MB        384 MB    ~2.3 GB   Yes
16         1028 MB   768 MB     240 MB        384 MB    ~2.4 GB   Yes
```

Note: Bottleneck is compute (encoding 32 slices per volume), not memory.

### Patch-level I-JEPA Pretraining

Model weights:

```
Component                          Params    Memory
---------------------------------  --------  --------
Context encoder (ViT-B/16)         86M       516 MB
Target encoder (EMA, no grad)      86M       344 MB
Predictor                          11M        66 MB
---------------------------------  --------  --------
Total                              183M      926 MB
```

Optimizer (AdamW, encoder + predictor): 97M params = 776 MB

Total by batch size:

```
Batch/GPU  Weights  Optimizer  Activations  Gradients  Total      Fits T4?
---------  -------  ---------  -----------  ---------  ---------  --------
2          926 MB   776 MB      132 MB       400 MB    ~2.2 GB    Yes
4          926 MB   776 MB      264 MB       400 MB    ~2.4 GB    Yes
8          926 MB   776 MB      528 MB       400 MB    ~2.6 GB    Yes
16         926 MB   776 MB     1056 MB       400 MB    ~3.2 GB    Yes
32         926 MB   776 MB     2112 MB       400 MB    ~4.2 GB    Yes
64         926 MB   776 MB     4224 MB       400 MB    ~6.3 GB    Yes
128        926 MB   776 MB     8448 MB       400 MB    ~10.6 GB   Tight
```

Recommended: batch_size=32-64 per GPU.

### Downstream: Slice approach (MLP only)

```
Batch/GPU  Frozen FE  Frozen slice enc  MLP     Optimizer  Total     Fits T4?
---------  ---------  ----------------  ------  ---------  --------  --------
2          112 MB     340 MB            <1 MB   <1 MB      ~0.5 GB   Yes
4          112 MB     340 MB            <1 MB   <1 MB      ~0.5 GB   Yes
16         112 MB     340 MB            <1 MB   <1 MB      ~0.5 GB   Yes
```

### Downstream: Patch approach (ViT integrator + MLP)

```
Batch/GPU  Frozen enc  ViT integrator  MLP     Optimizer  Total     Fits T4?
---------  ----------  --------------  ------  ---------  --------  --------
2          344 MB       92 MB          <1 MB   184 MB     ~1.0 GB   Yes
4          344 MB       92 MB          <1 MB   184 MB     ~1.1 GB   Yes
8          344 MB       92 MB          <1 MB   184 MB     ~1.2 GB   Yes
```

---

## Training Time Estimates (4x T4 GPUs)

### Slice-level I-JEPA Pretraining

```
Setting                Value
---------------------  ----------------------------------
Dataset                6,000 volumes
Batch size             8 per GPU, effective 32
Steps per epoch        6,000 / 32 = 188
Time per step          ~1.2s (32 encoder passes per vol)
Time per epoch         ~3.5 min
100 epochs             ~6 hours
200 epochs             ~12 hours
```

### Patch-level I-JEPA Pretraining

```
Setting                Value
---------------------  ----------------------------------
Dataset                192,000 slices
Batch size             32 per GPU, effective 128
Steps per epoch        192,000 / 128 = 1,500
Time per step          ~0.2s
Time per epoch         ~5 min
100 epochs             ~8 hours
300 epochs             ~25 hours
```

### Downstream Classification

```
Approach                What's trained      Time/epoch  Epochs   Total
----------------------  ------------------  ----------  -------  ----------
Slice (MLP only)        ~1.5K params        ~1 min      20-100   ~0.5-2 hrs
Patch (ViT int + MLP)   ~23M params         ~2 min      20-50    ~1-2 hrs
```

### Total End-to-End

```
Approach                Pretraining   Downstream    Total
----------------------  -----------   -----------   ----------
Slice (100 ep)          ~6 hours      ~0.5-2 hrs    ~8 hours
Slice (200 ep)          ~12 hours     ~0.5-2 hrs    ~14 hours
Patch (100 ep)          ~8 hours      ~1-2 hrs      ~10 hours
Patch (300 ep)          ~25 hours     ~1-2 hrs      ~27 hours
Current SLIViT (ref)    N/A           ~3 hrs        ~3 hours
```

---

## Hyperparameters

### Slice-level I-JEPA

```yaml
# Feature extractor (frozen, encodes individual slices)
feature_extractor:
  model: convnext_tiny_kermany   # or pretrained ViT
  output_dim: 768
  frozen: true

# Slice-level encoder
encoder:
  embed_dim: 768
  depth: 6
  num_heads: 12
  mlp_dim: 3072

# Slice-level predictor
predictor:
  embed_dim: 384
  depth: 6
  num_heads: 12

# Masking (over 32 slices)
masking:
  num_target_blocks: 4
  target_scale: [0.1, 0.2]       # ~3-6 slices masked
  context_scale: [0.75, 0.9]     # ~24-29 slices kept
  min_keep: 10

# Training
training:
  optimizer: AdamW
  lr: 0.0005
  warmup_epochs: 10
  weight_decay: 0.04
  ema_momentum: [0.996, 1.0]
  batch_size_per_gpu: 8
  epochs: 100-200
  mixed_precision: fp16
```

### Patch-level I-JEPA

```yaml
# Encoder
encoder:
  model: vit_base
  patch_size: 16
  crop_size: 256
  embed_dim: 768
  depth: 12
  num_heads: 12

# Predictor
predictor:
  embed_dim: 384
  depth: 6
  num_heads: 12

# Masking (over 256 patches per slice)
masking:
  num_target_blocks: 4
  target_scale: [0.15, 0.2]      # ~38 patches masked
  context_scale: [0.85, 1.0]     # ~218 patches kept
  aspect_ratio: [0.75, 1.5]
  allow_overlap: false
  min_keep: 10

# Training
training:
  optimizer: AdamW
  lr: 0.001
  start_lr: 0.0002
  final_lr: 0.000001
  warmup_epochs: 15
  weight_decay: 0.04
  final_weight_decay: 0.4
  ema_momentum: [0.996, 1.0]
  batch_size_per_gpu: 32
  epochs: 100-300
  mixed_precision: fp16
```

### Downstream: Slice approach

```yaml
training:
  encoder: frozen
  head: MLP (LayerNorm + Linear)
  optimizer: AdamW
  lr: 1e-3
  weight_decay: 0.01
  patience: 20
  batch_size_per_gpu: 16
```

### Downstream: Patch approach

```yaml
integrator:
  depth: 5
  heads: 12
  dim: 768
  mlp_dim: 3072

training:
  encoder: frozen or fine-tune (lr=1e-6)
  integrator: trained from scratch (lr=1e-4)
  head: MLP (lr=1e-3)
  optimizer: AdamW
  weight_decay: 0.01
  warmup_epochs: 3
  patience: 10
  batch_size_per_gpu: 8
```

---

## Key Differences Summary

```
                         Slice-level                 Patch-level
-----------------------  -----------------------     -----------------------
What is a "token"        Entire slice (768-d)        16x16 pixel patch (768-d)
Sequence length          32                          256
Pretraining data         6,000 volumes               192,000 slices
What it learns           Cross-slice structure        Within-slice features
Downstream classifier    MLP only (~1.5K params)     ViT integrator + MLP (~23M)
Downstream overfit risk  Very low                    Moderate (23M on 6K)
Pretraining (100ep)      ~6 hours                    ~8 hours
Downstream time          ~0.5-2 hours                ~2-3 hours
Total (100ep)            ~8 hours                    ~10 hours
Needs separate FE        Yes (frozen ConvNeXt/ViT)   No (encoder IS the FE)
```
