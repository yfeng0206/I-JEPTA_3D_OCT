# I-JEPA for FairVision OCT Glaucoma Classification

Self-supervised pretraining using [I-JEPA](https://github.com/facebookresearch/ijepa) (Assran et al., CVPR 2023) on [Harvard FairVision](https://github.com/Harvard-Ophthalmology-AI-Lab/FairVision) OCT data for binary glaucoma classification. Builds on our SLIViT reproduction in [SliViT_3D_OCT_Glaucoma](https://github.com/yfeng0206/SliViT_3D_OCT_Glaucoma).

## Motivation

Our SLIViT experiments reached 0.869 test AUC using a ConvNeXt feature extractor pretrained on Kermany OCT (CNV, DME, drusen, normal) and a ViT integrator trained on 6K labeled FairVision volumes. Two bottlenecks limited further improvement: the ConvNeXt features were pretrained on a different task (not glaucoma), and the ViT integrator was trained from scratch on a small labeled dataset.

I-JEPA addresses both by learning representations directly from unlabeled OCT data through masked prediction in representation space. No hand-crafted augmentations are needed. We implement two approaches that target each bottleneck separately, with patch-level being the primary approach.

## Approaches

### Patch-level I-JEPA

Standard I-JEPA applied to individual 256x256 OCT slices. Each slice is patchified into a 16x16 grid of 256 patches. The encoder learns within-slice spatial features (retinal layer boundaries, RNFL thickness patterns, optic nerve structures) by predicting masked patch representations from context patches.

Training data: 6,000 volumes x 100 slices = 600,000 slice images (self-supervised, no labels needed). Using 100 uniformly sampled slices from the 200 available per volume provides dense coverage while reducing redundancy from near-identical adjacent slices. The pretrained encoder replaces ConvNeXt as the feature extractor for downstream classification.

For downstream, each of the 100 slices is encoded independently by the frozen ViT, mean-pooled over patches to produce one 768-d token per slice, giving 100 slice tokens per volume. An attentive probe (adapted from the I-JEPA evaluation protocol) with 2 self-attention layers aggregates these into a volume-level representation via a learnable [CLS] token, followed by a linear classifier. Two layers are needed (vs the paper's 1) because our slice tokens are independently encoded with no inter-slice context — unlike the paper's patch tokens which already carry global context from 12 encoder layers. Features are pre-computed once with the frozen encoder and cached to disk, making probe training very fast (~30s/epoch on cached tensors).

### Slice-level I-JEPA (experimental)

I-JEPA applied to sequences of slice features within each volume. A ConvNeXt feature extractor encodes each of the 32 slices to a 768-d vector. The slice-level encoder then learns cross-slice relationships by predicting masked slice representations from context slices using 1D contiguous masking.

Training data: 6,000 volumes (self-supervised). In practice, we found that 32 slice tokens from a pretrained ConvNeXt lack sufficient diversity for I-JEPA pretraining: representations collapse to near-uniform vectors (pairwise cosine similarity >0.999) and the prediction loss reaches near-zero within 1-2 epochs. This is because adjacent OCT slices produce highly correlated features, making the masked prediction task trivially solvable by interpolation regardless of masking strategy. The patch-level approach avoids this by operating on 256 diverse spatial patches per image.

## Architecture

### Patch-level

| Component | Architecture | Params | Input | Output |
|-----------|-------------|--------|-------|--------|
| Context encoder | ViT-B/16 (12 layers, 768-d, 12 heads) | 86M | ~218 context patches (768-d) | 218 x 768-d |
| Target encoder | Same as context (EMA, no grad) | 86M | 256 patches (768-d) | 256 x 768-d |
| Predictor | Narrow ViT (6 layers, 384-d, 12 heads) | 11M | 218 context + 38 mask tokens (384-d) | 38 x 768-d |
| Patch embedding | Conv2d(3, 768, k=16, s=16) | 0.6M | (B, 3, 256, 256) | (B, 256, 768) |
| Positional embedding | 2D sinusoidal (fixed) | 0 | N/A | (1, 256, 768) |

Encoder:predictor depth ratio is 2:1 (12:6), matching the original I-JEPA design where the predictor must be weaker than the encoder.

### Slice-level

| Component | Architecture | Params | Input | Output |
|-----------|-------------|--------|-------|--------|
| Feature extractor | ConvNeXt-Tiny (Kermany pretrained, low LR fine-tune) | 28M | (B, 3, 256, 256) per slice | (B, 768) per slice |
| Context encoder | SliceViT (12 layers, 768-d, 12 heads) | 85M | ~27 context slice tokens (768-d) | 27 x 768-d |
| Target encoder | Same as context (EMA, no grad) | 85M | 32 slice tokens (768-d) | 32 x 768-d |
| Predictor | Narrow SliceViT (6 layers, 384-d, 12 heads) | 11M | 27 context + 5 mask tokens (384-d) | 5 x 768-d |
| Positional embedding | 1D sinusoidal (fixed) | 0 | N/A | (1, 32, 768) |

The feature extractor is fine-tuned with a very low learning rate (1e-6) rather than frozen. This allows gradual adaptation from Kermany OCT features to glaucoma-specific patterns, which was the single biggest improvement in our SLIViT experiments.

### Downstream classifier (patch-level)

Follows the I-JEPA attentive probe protocol, adapted for 3D volumes:

| Component | Architecture | Params | Trained? |
|-----------|-------------|--------|----------|
| Frozen encoder | ViT-B/16 (I-JEPA target encoder, epoch 11) | 86M | No |
| Slice pooling | Mean-pool 256 patch tokens → 1 per slice | 0 | No |
| Attentive probe | 2 transformer blocks (768-d, 12 heads) + [CLS] + 1D pos embed | ~14.3M | Yes |
| Linear head | LayerNorm(768) → Linear(768→1) | 769 | Yes |
| **Total trainable** | | **~14.3M** | |

Training protocol (matched to SLIViT for fair comparison):
- Optimizer: AdamW, weight_decay=0.01
- LR: probe=1e-4, head=1e-3, cosine schedule with 3-epoch warmup
- Batch size: 64 (on cached features)
- Early stopping: patience=5 on val AUC
- Loss: BCEWithLogitsLoss
- AUC: sklearn.metrics.roc_auc_score
- No data augmentation (same as SLIViT)

## Comparison with Original I-JEPA

| | Original I-JEPA | Patch-level (ours) | Slice-level (ours) |
|--|----------------|-------------------|-------------------|
| Dataset | ImageNet 1.2M | 600K OCT slices | 6K OCT volumes |
| Image size | 224x224 | 256x256 | N/A (32 tokens) |
| Encoder | ViT-H/14 (630M, 32 layers) | ViT-B/16 (86M, 12 layers) | SliceViT (85M, 12 layers) |
| Predictor | 384-d, 12 layers | 384-d, 6 layers | 384-d, 6 layers |
| Masking | 2D block, 4 targets | 2D block, 4 targets | 1D contiguous, 4 targets |
| Target scale | [0.15, 0.2] | [0.15, 0.2] | [0.1, 0.2] |
| Context scale | [0.85, 1.0] | [0.85, 1.0] | [0.75, 0.9] |
| Batch (per GPU) | 128 | 64 | 8 |
| LR | 0.001 | 0.00025 | 0.0003 |
| Warmup | 40 / 300 epochs | 5 / 50 epochs | 10 / 100 epochs |
| EMA momentum | [0.996, 1.0] | [0.996, 1.0] | [0.996, 1.0] |
| Hardware | 16x A100 80GB | 4x T4 16GB | 4x T4 16GB |

We use ViT-B instead of ViT-H because our dataset is 6-160x smaller than ImageNet. The predictor is 6 layers (not 12) to maintain the 2:1 encoder:predictor ratio. Learning rate was initially scaled via sqrt(batch_ratio) from the paper's 0.001 at batch=2048 to 0.0005 at effective batch=512, but empirically this was too high for OCT data (see Run 1 below): OCT images are less diverse than ImageNet so gradients are more correlated, requiring a lower peak LR. fp16 is used instead of bfloat16 (T4 limitation). Slice-level masking is 1D contiguous since adjacent OCT slices contain similar structures.

## Adapting I-JEPA for 1D Slice Sequences

The original I-JEPA operates on 2D patch grids from natural images. Adapting it to 1D slice sequences involved three key design choices.

### Joint predictor processing

The predictor concatenates mask tokens from all 4 target blocks together with context tokens and processes them in a single transformer pass. This allows mask tokens from different target blocks to attend to each other through self-attention, making the prediction problem a joint reasoning task across the full volume rather than 4 independent interpolations.

### Sampled context block

The context mask is a contiguous block of slices (75-90% of the sequence) rather than all non-target positions. Target slices that fall within this block are removed, creating gaps. Slices outside the context block are invisible to the encoder. This prevents trivial interpolation from immediate neighbors and forces the encoder to learn broader spatial relationships across the volume.

### Block-wise batch repetition

When pairing batch samples with multiple masks, samples are repeated in block-wise order (all samples for group 0, then all for group 1) to maintain correct alignment between predicted and target representations across the multi-mask I-JEPA training loop.

## Memory Budget

Per GPU (T4 16GB), fp16 mixed precision.

### Patch-level pretraining

| Batch/GPU | Model + Optimizer | Activations | Total | Fits T4? |
|-----------|------------------|-------------|-------|----------|
| 8 | 1.7 GB | 0.9 GB | ~2.6 GB | Yes |
| 32 | 1.7 GB | 2.5 GB | ~4.2 GB | Yes |
| 64 | 1.7 GB | 4.6 GB | ~6.3 GB | Yes |
| 128 | 1.7 GB | 8.8 GB | ~10.6 GB | Tight |

### Slice-level pretraining

| Batch/GPU | Model + Optimizer | Activations | Total | Fits T4? |
|-----------|------------------|-------------|-------|----------|
| 4 | 1.8 GB | 0.1 GB | ~2.2 GB | Yes |
| 8 | 1.8 GB | 0.1 GB | ~2.3 GB | Yes |
| 16 | 1.8 GB | 0.2 GB | ~2.4 GB | Yes |

Slice-level memory is dominated by the model weights, not activations (only 32 tokens). The bottleneck is compute: encoding 32 slices through ConvNeXt per volume.

## Training Time Estimates (4x T4 GPUs)

| Phase | Dataset | Batch | Time/epoch | Epochs | Total |
|-------|---------|-------|-----------|--------|-------|
| Patch pretraining | 600K slices | 512 eff | ~71 min | 50 | ~59 hrs |
| Slice pretraining | 6K volumes | 32 eff | ~3.5 min | 100 | ~6 hrs |
| Downstream feature extraction | 10K volumes × 100 slices | 1 vol | N/A | 1 pass | ~50 min |
| Downstream probe training | cached features | 64 | ~30 sec | 50 max | ~25 min |
| **Downstream total** | | | | | **~1.5 hrs** |
| SLIViT baseline | 6K labeled | 16 eff | ~5 min | 10-25 | ~3 hrs |

Both pretraining approaches include validation loss tracking and early stopping (configurable patience, default=8) to prevent overfitting.

## Training Details

### Loss function

Smooth L1 (Huber) loss between predicted and target representations, computed only at masked positions. Prediction happens in representation space, not pixel space. This forces the encoder to learn semantic features rather than low-level pixel statistics.

### EMA target encoder

The target encoder is never trained by gradient descent. Its weights are an exponential moving average of the context encoder, updated each iteration: `w_target = m * w_target + (1-m) * w_context`. The momentum `m` ramps linearly from 0.996 to 1.0 over training, providing increasingly stable targets.

### Optimizer

AdamW with cosine learning rate schedule (warmup then decay) and cosine weight decay schedule (0.04 to 0.4). Bias and LayerNorm parameters are excluded from weight decay.

### Masking strategy

Patch-level uses 2D block masking: 4 rectangular target blocks (15-20% of patches each, aspect ratio 0.75-1.5) and 1 large context block (85-100% of patches, targets removed). Block sizes are sampled once per batch for consistency; positions are random per image.

Slice-level uses 1D contiguous masking: 4 contiguous segments of 3-6 slices as targets. The context is a sampled contiguous block (75-90% of slices) with target positions removed, matching how the original I-JEPA samples context. Slices outside both the context block and target segments are invisible to the encoder, preventing trivial interpolation between adjacent slices.

### Early stopping

Both pretraining scripts evaluate reconstruction loss on the validation set each epoch. Training stops if validation loss does not improve for `patience` epochs (configurable, default=8). Best model (by val loss) is saved separately from the latest checkpoint.

## Preliminary Results

### Slice-level: representation collapse

Slice-level I-JEPA training on 6K volumes (32 slice tokens each) collapsed within 1-2 epochs across multiple configurations. Diagnostics showed:

| Epoch | train_loss | val_loss | cos_sim (pred vs actual) | rep_diversity (lower=better) |
|-------|-----------|---------|--------------------------|------------------------------|
| 1 | 0.0375 | 0.0006 | 0.9998 | 0.9983 |
| 2 | 0.0006 | 0.0003 | 1.0000 | 0.9993 |
| 3 | 0.0000 | 0.0000 | 1.0000 | 0.9992 |

The encoder converged to producing near-identical representations for all 32 slice positions (pairwise cosine similarity 0.999), making prediction trivial. This occurred regardless of masking strategy (full complement vs sampled block context, per-block vs joint predictor). The root cause is insufficient token diversity: adjacent OCT slices produce highly correlated ConvNeXt features, and 32 tokens is too few for I-JEPA's prediction task to remain challenging.

### Patch-level Run 1: LR=0.0005 (too high)

Full training run on 600K slices (100 per volume), ViT-B/16, batch=64/GPU, accum=2, effective batch=512, LR=0.0005, 15 warmup epochs, early stopping patience=15. Stopped at epoch 26.

| Epoch | train_loss | val_loss | cos_sim | rep_diversity | LR |
|-------|-----------|---------|---------|---------------|-----|
| 1 | 0.1298 | 0.2152 | 0.6971 | 0.81 | ~0.0001 (warmup) |
| 7 | 0.2159 | 0.2136 | 0.6972 | 0.68 | ~0.0003 |
| 8 | 0.2117 | 0.2113 | 0.7193 | 0.66 | ~0.0003 |
| 10 | 0.2086 | 0.2084 | 0.7424 | 0.73 | ~0.0004 |
| **11** | **0.2073** | **0.2081** | **0.72** | **0.69** | **~0.0004** |
| 15 | 0.2153 | 0.2187 | 0.68 | 0.68 | 0.0005 (peak) |
| 20 | 0.2646 | 0.2720 | 0.66 | 0.51 | ~0.0005 |
| 25 | 0.3047 | 0.3059 | 0.58 | 0.50 | ~0.0004 |

The model learned well during warmup (LR 0.0001 to 0.0004, epochs 1-11), but destabilized once LR reached the peak of 0.0005. Both train and val loss increased monotonically from epoch 12 onward, cos_sim dropped, and rep_diversity degraded toward 0.5. Early stopping triggered at epoch 26 (15 epochs without improvement from best at epoch 11).

Diagnosis: LR=0.0005 is too aggressive for OCT data. OCT images are less diverse than ImageNet (all retinal scans), producing more correlated gradients, so the effective learning rate is higher than the sqrt scaling formula predicted.

Best checkpoint (epoch 11, val_loss=0.2081) was saved. GPU memory: 12 GB at batch=64 on T4 16GB.

### Patch-level Run 2: LR=0.00025, warmup=5 (early stopping bug)

Reduced peak LR to 0.00025, shortened warmup to 5 epochs, patience=8. Resumed from fresh init. Training showed steady improvement but was cut short by an early stopping bug: epoch 1 (pre-warmup) recorded artificially low val_loss=0.1197 because the EMA target hadn't diverged yet, making it impossible to beat. The model improved from 0.2114 to 0.1636 (epochs 3-9) but early stopping triggered at epoch 9 since it could never beat epoch 1. Fix applied: early stopping now only counts after warmup.

### Patch-level Run 3: LR=0.00025, resume from epoch 9 (converged)

Resumed from Run 2's epoch 9 checkpoint with the early stopping fix. Training converged at epoch 11 (val_loss=0.1586) and plateaued through epoch 18 when the job crashed due to an NCCL timeout from blocking blob uploads. The best checkpoint was already saved.

| Epoch | train_loss | val_loss | cos_sim | rep_diversity | Notes |
|-------|-----------|---------|---------|---------------|-------|
| 10 | 0.1596 | 0.1586* | 0.81 | 0.80 | new best |
| 11 | 0.1574 | 0.1586* | 0.82 | 0.73 | tied best |
| 12 | 0.1589 | 0.1590 | 0.81 | 0.73 | patience 1 |
| 13 | 0.1589 | 0.1593 | 0.82 | 0.80 | patience 2 |
| 14 | 0.1595 | 0.1595 | 0.81 | 0.74 | patience 3 |
| 15 | 0.1591 | 0.1598 | 0.83 | 0.72 | patience 4 |
| 16 | 0.1595 | 0.1599 | 0.80 | 0.72 | patience 5 |
| 17 | 0.1603 | 0.1606 | 0.83 | 0.73 | patience 6 |
| 18 | 0.1618 | 0.1616 | 0.79 | 0.70 | patience 7 (crashed) |

Diagnostics remained healthy throughout: cos_sim ~0.80 (good prediction quality), rep_diversity 0.70-0.80 (no collapse). The val_loss plateau at ~0.159 is expected for I-JEPA — loss is not strongly correlated with downstream representation quality (confirmed by community reports and [Rethinking JEPA](https://openreview.net/forum?id=2r3GUcMIFe)).

Best checkpoint: `checkpoints/jepa_patch-run3-ep11.pth.tar` (epoch 11, val_loss=0.1586, ViT-B/16 encoder).

**Key lessons learned:**
- LR=0.0005 too aggressive for OCT → 0.00025 works well
- Early stopping must ignore pre-warmup epochs (EMA target hasn't diverged)
- Blob uploads must be non-blocking to avoid DDP NCCL timeouts
- I-JEPA loss plateaus are normal; use downstream probes or RankMe to evaluate representation quality

### Downstream: Frozen Encoder + Attentive Probe (test AUC: 0.733)

Using the best pretrained encoder (Run 3, epoch 11, val_loss=0.1586) for downstream glaucoma classification with frozen encoder + attentive probe (2 blocks) + linear head. Training protocol matched to SLIViT for fair comparison.

**Architecture:**
```
OCT Volume (200 B-scans)
  → Sample 100 slices (uniform)
  → Frozen ViT-B/16 per slice → mean-pool patches → (B, 100, 768)
  → AttentiveProbe: [CLS] + pos embed + 2 transformer blocks → (B, 768)
  → LinearHead: LayerNorm → Linear(768→1)
  → BCEWithLogitsLoss → P(glaucoma)
```

**Configuration:**
- 100 slices per volume, features pre-computed and cached to disk (~2.9 GB)
- Probe: 2 blocks, 12 heads, 768-d (~14.3M trainable params)
- Batch size: 64, patience: 5, LR probe=1e-4, LR head=1e-3
- AdamW, weight_decay=0.01, cosine schedule with 3-epoch warmup
- No data augmentation (same as SLIViT)

**Results:**

| Epoch | Train Loss | Val Loss | Val AUC | LR |
|-------|-----------|---------|---------|-----|
| 1 | 0.7033 | 0.6847 | 0.5854 | 3.3e-5 (warmup) |
| 10 | 0.6420 | 0.6439 | 0.7026 | 9.5e-5 |
| 20 | 0.6040 | 0.6122 | 0.7319 | 7.1e-5 |
| **28** | **0.5743** | **0.6025** | **0.7435** | **4.5e-5** |
| 33 | 0.5631 | 0.6018 | 0.7406 | 2.9e-5 (early stop) |

Best epoch 28, early stopped at 33. **Test AUC: 0.7327** (vs SLIViT baseline: 0.869).

Feature extraction: 6,000 train + 1,000 val + 3,000 test volumes in ~78 min (2.1 vol/s on T4). Training on cached features: ~5 min for 33 epochs.

**Comparison with SLIViT baseline:**

| | I-JEPA Frozen Probe | SLIViT (best) |
|--|-------------------|---------------|
| Test AUC | 0.733 | **0.869** |
| Encoder training | Self-supervised (no labels) | Fine-tuned **with glaucoma labels** |
| Encoder pretrained on | OCT patches (from scratch) | Kermany OCT (medical classification) |
| Encoder adaptation | Frozen (no task adaptation) | Full fine-tune with low LR |
| Pretraining epochs | 18 (crashed) | N/A (used pretrained ConvNeXt) |
| Trainable params | 14.3M (probe only) | 50-77M (everything) |

**Why the gap is large (0.73 vs 0.87):**

1. **Undertrained encoder.** The I-JEPA paper trains ViT-H for 300-600 epochs on 1.2M ImageNet images. We trained ViT-B for only 18 epochs on 600K OCT slices before the job crashed. The representations haven't fully converged.

2. **Frozen encoder = no task adaptation.** Glaucoma diagnosis requires detecting subtle structural changes (RNFL thinning, optic cup enlargement). SLIViT fine-tunes its encoder to amplify these glaucoma-specific signals. Our frozen encoder only learned generic "predict masked patches" features — it doesn't know what glaucoma looks like. Even on ImageNet, the I-JEPA paper shows a ~4% gap between frozen probe and fine-tuned; for medical imaging where diagnostic features are subtle, the gap is expected to be much larger.

3. **Mean-pooling discards spatial info.** We collapse 256 patch tokens → 1 vector per slice, throwing away *where* in the slice the features are. SLIViT preserves the full spatial feature map (768×8×8 = 49K-d per slice). Glaucoma is about *where* the RNFL is thin — spatial information matters.

4. **No medical pretraining.** SLIViT's ConvNeXt was pretrained on Kermany OCT (a medical classification task), giving it domain-relevant features before fine-tuning. Our ViT was trained entirely from scratch on OCT patches with no prior medical knowledge.

The 0.73 AUC for a frozen probe is consistent with the medical SSL literature, where frozen probe AUC typically ranges 0.70-0.80 and fine-tuning reaches 0.85-0.90. The next step is to unfreeze the encoder with a low learning rate (Phase 2 fine-tuning) to close the gap.

## Dataset

Harvard FairVision Glaucoma subset:
- 10,000 subjects total (6,000 train / 1,000 validation / 3,000 test)
- Each subject has a 200x200x200 OCT B-scan volume stored as `.npz`
- Binary labels: glaucoma (1) or not (0)
- 100 slices uniformly sampled from each volume for patch-level (600K images), resized to 256x256
- ~63GB compressed, available on [HuggingFace](https://huggingface.co/datasets/ming0100/Harvard_FairVision)

## Project Structure

```
src/
  models/
    vision_transformer.py    ViT encoder, predictor, slice-level variants
    feature_extractor.py     ConvNeXt loader (supports frozen or fine-tuned)
  masks/
    multiblock.py            2D block masking (patch-level)
    slice_mask.py            1D contiguous masking (slice-level)
    utils.py                 apply_masks helper
  datasets/
    oct_slices.py            Individual slice dataset (600K images at 100 slices)
    oct_volumes.py           Volume dataset (6K volumes)
  utils/
    tensors.py               trunc_normal_, repeat_interleave_batch
    distributed.py           DDP init, AllReduce
    schedulers.py            Warmup cosine LR, cosine WD
    logging.py               CSVLogger, AverageMeter, gpu_timer
  train_patch.py             Patch-level I-JEPA pretraining
  train_slice.py             Slice-level I-JEPA pretraining
  eval_downstream.py         Downstream classification (both approaches)
  helper.py                  Model init, optimizer, checkpoint I/O
  transforms.py              Data augmentation (minimal)

configs/
  patch_vitb16_ep100.yaml    Patch pretraining config
  slice_ep100.yaml           Slice pretraining config
  downstream_patch.yaml      Downstream probe config (100 slices, attentive probe)
  downstream_slice.yaml      Downstream with slice encoder
  aml_downstream.yml         AzureML job config for downstream probe

scripts/
  run_patch.sh               Entry point for patch pretraining jobs
  run_slice.sh               Entry point for slice pretraining jobs
  run_downstream.sh          Entry point for downstream evaluation
```

## References

- Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" ([paper](https://arxiv.org/abs/2301.08243), [code](https://github.com/facebookresearch/ijepa))
- Avram et al., "SLIViT: a general AI framework for clinical-feature diagnosis from limited 3D biomedical-imaging data" ([paper](https://pubmed.ncbi.nlm.nih.gov/38045283/), [code](https://github.com/cozygene/SLIViT))
- Luo et al., "Harvard Ophthalmology AI-Lab FairVision Dataset" ([paper](https://arxiv.org/abs/2310.02492), [code](https://github.com/Harvard-Ophthalmology-AI-Lab/FairVision))
