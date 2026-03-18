# I-JEPA for FairVision OCT Glaucoma Classification

Self-supervised pretraining using [I-JEPA](https://github.com/facebookresearch/ijepa) (Assran et al., CVPR 2023) on [Harvard FairVision](https://github.com/Harvard-Ophthalmology-AI-Lab/FairVision) OCT data for binary glaucoma classification. Builds on our SLIViT reproduction in [SliViT_3D_OCT_Glaucoma](https://github.com/yfeng0206/SliViT_3D_OCT_Glaucoma).

## Motivation

Our SLIViT experiments reached 0.869 test AUC using a ConvNeXt feature extractor pretrained on Kermany OCT (CNV, DME, drusen, normal) and a ViT integrator trained on 6K labeled FairVision volumes. Two bottlenecks limited further improvement: the ConvNeXt features were pretrained on a different task (not glaucoma), and the ViT integrator was trained from scratch on a small labeled dataset.

I-JEPA addresses both by learning representations directly from unlabeled OCT data through masked prediction in representation space. No hand-crafted augmentations are needed. We implement two complementary approaches that target each bottleneck separately.

## Approaches

### Patch-level I-JEPA

Standard I-JEPA applied to individual 256x256 OCT slices. Each slice is patchified into a 16x16 grid of 256 patches. The encoder learns within-slice spatial features (retinal layer boundaries, RNFL thickness patterns, optic nerve structures) by predicting masked patch representations from context patches.

Training data: 6,000 volumes x 32 slices = 192,000 slice images (self-supervised, no labels needed). The pretrained encoder replaces ConvNeXt as the feature extractor for downstream classification.

For downstream, each of the 32 slices is encoded independently, producing 32 feature vectors. A ViT integrator (trained from scratch on 6K labeled volumes) learns cross-slice relationships and feeds a classification head.

### Slice-level I-JEPA

I-JEPA applied to sequences of slice features within each volume. A ConvNeXt feature extractor (fine-tuned with low LR) encodes each of the 32 slices to a 768-d vector. The slice-level encoder then learns cross-slice relationships by predicting masked slice representations from context slices using 1D contiguous masking.

Training data: 6,000 volumes (self-supervised). The pretrained slice encoder captures which spatial patterns across slices are predictive. For downstream, the encoder output is average-pooled and classified with an MLP head (~1.5K params), minimizing overfitting risk.

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

### Downstream classifiers

| Approach | What is trained | Params trained | Method |
|----------|----------------|---------------|--------|
| Patch-level | ViT integrator (5 layers, 768-d) + MLP | ~23M | 32 slice features from frozen encoder, CLS token, BCEWithLogitsLoss |
| Slice-level | MLP head only | ~1.5K | Average pool over 32 encoded slice features, BCEWithLogitsLoss |

## Comparison with Original I-JEPA

| | Original I-JEPA | Patch-level (ours) | Slice-level (ours) |
|--|----------------|-------------------|-------------------|
| Dataset | ImageNet 1.2M | 192K OCT slices | 6K OCT volumes |
| Image size | 224x224 | 256x256 | N/A (32 tokens) |
| Encoder | ViT-H/14 (630M, 32 layers) | ViT-B/16 (86M, 12 layers) | SliceViT (85M, 12 layers) |
| Predictor | 384-d, 12 layers | 384-d, 6 layers | 384-d, 6 layers |
| Masking | 2D block, 4 targets | 2D block, 4 targets | 1D contiguous, 4 targets |
| Target scale | [0.15, 0.2] | [0.15, 0.2] | [0.1, 0.2] |
| Context scale | [0.85, 1.0] | [0.85, 1.0] | [0.75, 0.9] |
| Batch (per GPU) | 128 | 32 | 8 |
| LR | 0.001 | 0.0005 | 0.0003 |
| Warmup | 40 / 300 epochs | 15 / 100 epochs | 10 / 100 epochs |
| EMA momentum | [0.996, 1.0] | [0.996, 1.0] | [0.996, 1.0] |
| Hardware | 16x A100 80GB | 4x T4 16GB | 4x T4 16GB |

We use ViT-B instead of ViT-H because our dataset is 6-160x smaller than ImageNet. The predictor is 6 layers (not 12) to maintain the 2:1 encoder:predictor ratio. Learning rate is scaled down proportionally to the smaller effective batch size. fp16 is used instead of bfloat16 (T4 limitation). Slice-level masking is 1D contiguous since adjacent OCT slices contain similar structures.

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
| Patch pretraining | 192K slices | 128 eff | ~5 min | 100 | ~8 hrs |
| Slice pretraining | 6K volumes | 32 eff | ~3.5 min | 100 | ~6 hrs |
| Downstream (patch) | 6K labeled | 32 eff | ~2 min | 20-50 | ~1-2 hrs |
| Downstream (slice) | 6K labeled | 64 eff | ~1 min | 20-100 | ~0.5-2 hrs |
| SLIViT baseline | 6K labeled | 16 eff | ~5 min | 10-25 | ~3 hrs |

Both pretraining approaches include validation loss tracking and early stopping (patience=15) to prevent overfitting.

## Training Details

### Loss function

Smooth L1 (Huber) loss between predicted and target representations, computed only at masked positions. Prediction happens in representation space, not pixel space. This forces the encoder to learn semantic features rather than low-level pixel statistics.

### EMA target encoder

The target encoder is never trained by gradient descent. Its weights are an exponential moving average of the context encoder, updated each iteration: `w_target = m * w_target + (1-m) * w_context`. The momentum `m` ramps linearly from 0.996 to 1.0 over training, providing increasingly stable targets.

### Optimizer

AdamW with cosine learning rate schedule (warmup then decay) and cosine weight decay schedule (0.04 to 0.4). Bias and LayerNorm parameters are excluded from weight decay.

### Masking strategy

Patch-level uses 2D block masking: 4 rectangular target blocks (15-20% of patches each, aspect ratio 0.75-1.5) and 1 large context block (85-100% of patches, targets removed). Block sizes are sampled once per batch for consistency; positions are random per image.

Slice-level uses 1D contiguous masking: 4 contiguous segments of 3-6 slices as targets, with the remaining 24-29 slices as context. Contiguous masking is chosen because adjacent OCT slices share structural features, making the prediction task semantically meaningful.

### Early stopping

Both pretraining scripts evaluate reconstruction loss on the validation set each epoch. Training stops if validation loss does not improve for 15 epochs. Best model (by val loss) is saved separately from the latest checkpoint.

## Dataset

Harvard FairVision Glaucoma subset:
- 10,000 subjects total (6,000 train / 1,000 validation / 3,000 test)
- Each subject has a 200x200x200 OCT B-scan volume stored as `.npz`
- Binary labels: glaucoma (1) or not (0)
- 32 slices uniformly sampled from each volume, resized to 256x256
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
    oct_slices.py            Individual slice dataset (192K images)
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
  downstream_patch.yaml      Downstream with patch encoder
  downstream_slice.yaml      Downstream with slice encoder

scripts/
  run_patch.sh               Entry point for patch pretraining jobs
  run_slice.sh               Entry point for slice pretraining jobs
  run_downstream.sh          Entry point for downstream evaluation
```

## References

- Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" ([paper](https://arxiv.org/abs/2301.08243), [code](https://github.com/facebookresearch/ijepa))
- Avram et al., "SLIViT: a general AI framework for clinical-feature diagnosis from limited 3D biomedical-imaging data" ([paper](https://pubmed.ncbi.nlm.nih.gov/38045283/), [code](https://github.com/cozygene/SLIViT))
- Luo et al., "Harvard Ophthalmology AI-Lab FairVision Dataset" ([paper](https://arxiv.org/abs/2310.02492), [code](https://github.com/Harvard-Ophthalmology-AI-Lab/FairVision))
