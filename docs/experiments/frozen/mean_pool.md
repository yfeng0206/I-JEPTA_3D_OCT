# Frozen Probe: MeanPool + Linear on ep100 (ablation floor)

Mean-pool across slices (no attention, no parameters in the probe itself) + LinearHead. Serves as the ablation floor to quantify how much slice-level attention and positional embeddings contribute.

AML job: `quirky_branch_vkcy47sptn` — running as of 2026-04-18.

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
| Best epoch | TBD |
| Best Val AUC | TBD |
| Test AUC | TBD |

Will be filled in when `quirky_branch_vkcy47sptn` terminates (~1h). Expected to underperform CrossAttnPool and d=1-attn, but the gap quantifies how much slice-aware attention is earning.

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
