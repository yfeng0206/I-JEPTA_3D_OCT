# Frozen Probe: CrossAttnPool on ep100

Minimal cross-attention pool (277K params) on the best pretraining checkpoint (ep100). Ablation against the d=1 AttentiveProbe (7.17M) to test whether a 26× smaller probe is sufficient.

AML job: `good_eye_l97n8vn10l` — completed 2026-04-18.

## Architecture

Defined in [`src/models/attentive_pool_minimal.py`](../../../src/models/attentive_pool_minimal.py).

- 1 learnable query token (768-d)
- Slice-axis positional embeddings (100 × 768 = 76,800 params)
- Single-head cross-attention, head_dim=64 (compressed)
  - Q, K, V, O projections: 4 × (768 × 64) = ~200K params
- LayerNorm(768)
- No FFN, no self-attention between slices

Forward: `x + pos_embed → q=query, k=v=x → single-head attn → o_proj → LayerNorm → pooled 768`.

Parameter breakdown:

| Component | Params |
|---|---|
| query_token | 768 |
| pos_embed (100 × 768) | 76,800 |
| q_proj (768 → 64) | 49,216 |
| k_proj (768 → 64) | 49,216 |
| v_proj (768 → 64) | 49,216 |
| o_proj (64 → 768) | 49,920 |
| LayerNorm | 1,536 |
| **Total (probe)** | **276,672** |
| LinearHead | 2,305 |
| **Total trainable** | **278,977** |

## Result

| Metric | Value |
|---|---|
| Best epoch | 11 |
| Best Val AUC | 0.8650 |
| **Test AUC** | **0.8791** |
| Test loss | 0.4340 |
| Sensitivity | 0.7913 |
| Specificity | 0.8083 |

Compared to d=1 baseline on same checkpoint:

| Probe | Params | Val AUC | Test AUC |
|---|---|---|---|
| d=1 AttentiveProbe (baseline) | 7.17M | 0.8597 | 0.8706 |
| **CrossAttnPool (this run)** | **277K** | **0.8650 (+0.005)** | **0.8791 (+0.009)** |

CrossAttnPool wins by +0.009 Test AUC at 26× fewer parameters. Val AUC delta (+0.005) is within the ~0.008 noise band on 1000 val samples; Test AUC delta (+0.009) on 3000 test samples is ~1.6σ — borderline-significant favorable.

## Interpretation

- The full self-attention across slices and the per-slice FFN in the I-JEPA-style AttentiveProbe are **not earning their 6.9M extra parameters** on this task.
- Single cross-attention pooling with position embeddings captures the slice-weighting signal that matters for glaucoma (localized optic-cup region).
- Supports the "Attention, Please!" (ICLR 2026) observation that attentive probes are typically over-parameterized.

## Config

| Parameter | Value |
|---|---|
| Num slices | 100 |
| Batch size | 256 |
| Epochs / patience | 50 / 15 |
| Warmup | 5 epochs |
| LR probe = LR head | 4e-4 |
| Weight decay | 0.05 |
| Dropout (probe N/A — no FFN or dropout in CrossAttnPool) | — |
| Optimizer | AdamW + cosine |
