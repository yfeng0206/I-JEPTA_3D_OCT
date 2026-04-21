# Frozen Probe: MeanPool + Linear on ep100 (ablation floor)

Mean-pool across slices (no attention, no parameters in the probe itself) + LinearHead. Serves as the ablation floor to quantify how much slice-level attention and positional embeddings contribute.

Completed 2026-04-18.

## Architecture

Defined in [`src/models/attentive_pool_minimal.py`](../../../src/models/attentive_pool_minimal.py) (`MeanPool`).

```
x: (B, 100 slices, 768)  →  x.mean(dim=1)  →  (B, 768)  →  LinearHead  →  logit
```

Zero trainable parameters in the probe. Only the LinearHead (LayerNorm + Linear(768, 1) = 2,305 params) trains.

## Why it's the floor

- No attention → cannot re-weight slices by relevance
- No pos_embed → slice ordering fully discarded (two volumes with the same set of slice features in different axial orders pool to identical vectors)
- For glaucoma, which localizes to the optic-cup region (~slices 60-90), mean-pool dilutes informative slices with the rest

The number we get here answers: "with everything position-aware removed, how much test AUC do we retain?"

## Result

| Metric | Value |
|---|---|
| Best epoch | 48 |
| Best Val AUC | 0.8559 |
| **Test AUC** | **0.8746** |
| Test loss | 0.4373 |
| Sensitivity | 0.761 |
| Specificity | 0.838 |

## Three-way frozen-probe comparison on ep100

| Probe | Params | Val AUC | Test AUC |
|---|---|---|---|
| **MeanPool + Linear (this run)** | **2.3K** (no probe params) | **0.8559** | **0.8746** |
| d=1 AttentiveProbe + Linear | 7.17M | 0.8597 | 0.8706 |
| CrossAttnPool + Linear | 277K | 0.8650 | 0.8791 |

**Surprising finding**: mean_pool beats the d=1 AttentiveProbe on Test AUC (+0.004), despite discarding slice ordering entirely. Strong evidence that the d=1 probe's 7M params cause overfitting rather than capturing real signal. Attention-based pooling IS genuinely helping on this task — but only in the minimal CrossAttnPool form (+0.005 Test AUC over mean_pool), not the over-parameterized d=1 form.

Interpretation: for FairVision glaucoma (6K volumes, ~balanced prevalence), a tiny linear probe on mean-pooled features is already a very strong baseline. Slice-level attention + position embeddings add ~0.5% AUC on top. Anything bigger overfits.

## Config (matches other frozen ablations for clean comparison)

| Parameter | Value |
|---|---|
| Num slices | 100 |
| Batch size | 256 |
| Epochs / patience | 50 / 15 |
| Warmup | 5 epochs |
| LR head | 4e-4 |
| Weight decay | 0.05 |
| Optimizer | AdamW + cosine |
