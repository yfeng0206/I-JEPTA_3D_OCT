# Frozen Probe Downstream Experiments

## Approach

Evaluate **frozen** I-JEPA ViT-B/16 encoders on OCT glaucoma classification using the FairVision dataset:

1. Frozen ViT-B/16 produces per-patch embeddings for each OCT slice.
2. Patches are **mean-pooled within each slice** → one 768-dim token per slice.
3. Slice tokens → learnable **AttentiveProbe d=1** (7M trainable, one self-attention block with FFN + slice pos_embed) → pooled 768-dim volume vector.
4. **LinearHead** → logit. Trained with BCEWithLogitsLoss. Encoder never updated.

All runs: 100 slices per eye, 256×256 crop, patch 16.

## Current results

### random_posfix_d1_sweep — d=1 literature-tuned sweep

Full writeup: [random_posfix_d1_sweep.md](random_posfix_d1_sweep.md).

| Checkpoint | Val AUC | Test AUC | Best Epoch | Sensitivity | Specificity |
|---|---|---|---|---|---|
| ep25  | 0.8460 | 0.8558 | 8 | 0.609 | 0.910 |
| ep50  | 0.8452 | 0.8611 | 4 | 0.808 | 0.731 |
| ep75  | 0.8580 | 0.8691 | 4 | 0.772 | 0.791 |
| **ep100** | **0.8597** | **0.8706** | 4 | 0.821 | 0.716 |

ep100 wins. Test AUC improves monotonically with pretraining length (+0.015 from ep25 to ep100).

| Config | Value |
|---|---|
| Probe | AttentiveProbe d=1 (7.17M trainable) + LinearHead (2305) |
| Batch size | 256 |
| Epochs / patience | 50 / 15 |
| Warmup | 5 epochs |
| LR probe = LR head | 4e-4 (linear-scaled from 1e-3 @ bs=1024) |
| Weight decay | 0.05 |
| Dropout (probe) | 0.2 |
| Optimizer | AdamW + cosine |

Plots: [val AUC per epoch](../../../../results/downstream/linear_sweep_random_posfix_d1/val_auc_per_epoch.png) · [loss grid](../../../../results/downstream/linear_sweep_random_posfix_d1/train_val_loss_grid.png) · [AUC per checkpoint](../../../../results/downstream/linear_sweep_random_posfix_d1/auc_per_checkpoint.png).

## Planned ablations (probe architecture)

All run on cached features from ep100. No encoder re-eval needed.

| Ablation | Probe | Params | Purpose |
|---|---|---|---|
| Mean-pool + linear | `(B, 100, 768) → mean(dim=1) → Linear(768, 1)` | ~800 | Does attention earn its keep at all? |
| **Cross-attn pool** (minimal) | single learnable query, single-head cross-attn (head_dim=64), no FFN, slice pos_embed | ~280K | Does a 26× smaller cross-attn probe match d=1 self-attn? |
| V-JEPA-style cross-attn + FFN | cross-attn + FFN, no self-attn between slices | ~6.5M | Direct port of V-JEPA evaluation probe |

## Key lessons

1. **Probe capacity isn't the bottleneck.** d=3 (21M) vs d=1 (7M) → ~0 AUC delta. Encoder features are the ceiling. See [lessons_learned.md #7](../../../lessons_learned.md) + [research_log.md #5](../../../research_log.md).
2. **Overfit dynamics are intrinsic** to attentive probes on small medical data. Even with literature-standard regularization, train AUC hits 0.99+ by epoch 10-15. This is an open problem ([Attention, Please! ICLR 2026](https://arxiv.org/abs/2506.10178)) — not a bug in our setup.
3. **Frozen-probe ceiling is ~0.85-0.87 on this dataset.** Fine-tuning is the lever for the next 3-5% (see [unfrozen README](../unfrozen/README.md)).
