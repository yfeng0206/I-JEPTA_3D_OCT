# Frozen Probe: Random-init I-JEPA, d=3, MLP (normalization fixed)

Run ID: `frozen_random_d3_s100_normfix`

## Configuration

| Parameter | Value |
|-----------|-------|
| Mode | patch |
| Encoder | vit_base (ViT-B/16) |
| Encoder Checkpoint | jepa_patch-best.pth.tar (Random-init, ep11) |
| Freeze Encoder | true |
| Probe Depth | 3 |
| Probe Num Heads | 12 |
| Head Type | mlp |
| Num Slices | 100 |
| Slice Size | 256 |
| Crop Size | 256 |
| Patch Size | 16 |
| Batch Size | 64 |
| Accum Steps | 4 |
| LR (probe) | 1e-4 |
| LR (head) | 1e-3 |
| Weight Decay | 0 |
| Dropout | 0.1 |
| Epochs | 100 |
| Patience | 20 |
| Warmup Epochs | 3 |
| Seed | 42 |
| ImageNet Normalization | **Yes (fixed)** |

## Results

| Metric | Value |
|--------|-------|
| **Test AUC** | **0.8339** |
| Val AUC (best) | 0.8279 |
| Test Loss | 0.5409 |
| Sensitivity | 0.8274 |
| Specificity | 0.6402 |
| Best Epoch | 7 |
| Probe Params | 21,343,488 |
| Head Params | 198,657 |

### Comparison with old (pre-normfix) result

| Metric | Old (no norm) | New (normfix) | Change |
|--------|--------------|---------------|--------|
| Test AUC | 0.734 | **0.834** | **+10.0 points** |
| Val AUC | 0.752 | 0.828 | +7.6 points |
| Sensitivity | N/A | 0.827 | -- |
| Best Epoch | 27 | 7 | Converges faster |

## Training Log

| Epoch | Train Loss | Train AUC | Val Loss | Val AUC | LR |
|-------|-----------|-----------|----------|---------|-----|
| 1 | 0.6106 | 0.7239 | 0.5675 | 0.8033 | 3.33e-5 |
| 2 | 0.5617 | 0.7787 | 0.5363 | 0.8067 | 6.67e-5 |
| 3 | 0.5251 | 0.8080 | 0.5364 | 0.8155 | 1.00e-4 |
| 4 | 0.5160 | 0.8129 | 0.5961 | 0.8203 | 9.99e-5 |
| 5 | 0.5054 | 0.8251 | 0.5593 | 0.8207 | 9.99e-5 |
| 6 | 0.4872 | 0.8357 | 0.5152 | 0.8210 | 9.98e-5 |
| **7** | **0.4900** | **0.8348** | **0.5422** | **0.8279** | **9.96e-5** |
| 8 | 0.4789 | 0.8445 | 0.5342 | 0.8224 | 9.94e-5 |
| 9 | 0.4584 | 0.8558 | 0.5200 | 0.8220 | 9.91e-5 |
| 10 | 0.4554 | 0.8581 | 0.5698 | 0.8181 | 9.87e-5 |
| 11 | 0.4367 | 0.8697 | 0.5414 | 0.8150 | 9.83e-5 |
| 12 | 0.4213 | 0.8775 | 0.5687 | 0.7971 | 9.79e-5 |
| 13 | 0.4109 | 0.8851 | 0.5830 | 0.8139 | 9.74e-5 |
| 14 | 0.3909 | 0.8981 | 0.5911 | 0.7990 | 9.69e-5 |
| 15 | 0.3628 | 0.9127 | 0.6238 | 0.8107 | 9.63e-5 |
| 16 | 0.3446 | 0.9228 | 0.6567 | 0.7886 | 9.56e-5 |
| 17 | 0.3175 | 0.9358 | 0.8018 | 0.7830 | 9.49e-5 |
| 18 | 0.2909 | 0.9461 | 0.8042 | 0.7864 | 9.42e-5 |
| 19 | 0.2915 | 0.9471 | 0.7971 | 0.7811 | 9.34e-5 |
| 20 | 0.2524 | 0.9596 | 0.8880 | 0.7770 | 9.26e-5 |
| 21 | 0.2263 | 0.9671 | 0.8523 | 0.7776 | 9.17e-5 |
| 22 | 0.1985 | 0.9738 | 0.8017 | 0.7816 | 9.08e-5 |
| 23 | 0.1668 | 0.9819 | 0.8636 | 0.7667 | 8.99e-5 |
| 24 | 0.1257 | 0.9893 | 1.0847 | 0.7644 | 8.89e-5 |
| 25 | 0.1215 | 0.9900 | 1.1542 | 0.7510 | 8.78e-5 |
| 26 | 0.0930 | 0.9936 | 1.2391 | 0.7554 | 8.68e-5 |
| 27 | 0.0940 | 0.9939 | 1.0792 | 0.7498 | 8.56e-5 |

*Early stopping at epoch 27 (patience=20). Best val AUC at epoch 7.*

## Analysis

1. **Fast convergence**: Model reaches best val AUC (0.828) by epoch 7 — much faster than the old (pre-normfix) run which peaked at epoch 27. With correct normalization, the encoder features are immediately useful and the probe converges quickly.

2. **Strong overfitting after epoch 7**: Train AUC continues to 0.99+ while val AUC declines. The val loss increases from 0.54 to 1.08 by epoch 27. This suggests the probe has enough capacity to memorize training data but the generalization gap widens rapidly.

3. **High sensitivity**: 82.7% sensitivity (glaucoma detection rate) with 64.0% specificity at threshold=0.5. The old run didn't report sensitivity, but based on the AUC difference, the normalization fix dramatically improved the model's ability to detect glaucoma cases.

[<-- Back to frozen probe overview](README.md)
