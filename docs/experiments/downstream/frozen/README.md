# Frozen Probe Downstream Experiments

## Approach

These experiments evaluate **frozen** I-JEPA ViT-B/16 encoders on OCT glaucoma classification using the FairVision dataset. The pipeline:

1. **Frozen ViT-B/16** produces per-patch embeddings for each OCT slice.
2. Patches are **mean-pooled within each slice** → one 768-dim token per slice.
3. Slice tokens → learnable **AttentiveProbe** (cross-attention over slice set) → pooled 768-dim volume vector.
4. **LinearHead** → logit. Trained with BCEWithLogitsLoss. Encoder never updated.

All runs use 100 OCT slices per eye, 256×256 crop, patch size 16.

## Current results

Probe depth default is **d=1** (matches I-JEPA paper), literature-tuned regularization.

### random_posfix_d1_sweep — d=1 literature-tuned sweep

Full details: [random_posfix_d1_sweep.md](random_posfix_d1_sweep.md).

| Checkpoint | Val AUC | Test AUC | Best Epoch | Sensitivity | Specificity |
|---|---|---|---|---|---|
| ep25  | 0.8460 | 0.8558 | 8 | 0.609 | 0.910 |
| ep50  | 0.8452 | 0.8611 | 4 | 0.808 | 0.731 |
| ep75  | 0.8580 | 0.8691 | 4 | 0.772 | 0.791 |
| **ep100** | **0.8597** | **0.8706** | 4 | 0.821 | 0.716 |

Winner: ep100. Monotonic improvement with pretraining length.

| Parameter | Value |
|---|---|
| Probe | AttentiveProbe d=1 (7.17M trainable) + LinearHead (2305) |
| Batch size | 256 |
| Epochs / patience | 50 / 15 |
| Warmup | 5 epochs |
| LR probe = LR head | 4e-4 (linear-scaled from 1e-3 @ bs=1024) |
| Weight decay | 0.05 |
| Dropout (probe) | 0.2 |
| Optimizer | AdamW + cosine |

Plots: [val AUC per epoch](../../../../results/downstream/linear_sweep_random_posfix_d1/val_auc_per_epoch.png), [loss grid](../../../../results/downstream/linear_sweep_random_posfix_d1/train_val_loss_grid.png), [AUC per checkpoint](../../../../results/downstream/linear_sweep_random_posfix_d1/auc_per_checkpoint.png).

## Planned ablations

| Ablation | Probe | Purpose |
|---|---|---|
| Mean-pool + linear | ~800 params | Does attention earn its keep at all? |
| **Cross-attn pool** (minimal) | ~280K params | Does a 26× smaller cross-attn probe match d=1 self-attn? |
| V-JEPA-style cross-attn + FFN | ~6.5M | Direct port of V-JEPA evaluation architecture |

All run on cached features from ep100, no encoder re-eval needed.

## Historical runs (kept for reference)

Pre-posfix (broken position encoding in SSL) and d=3 (overparameterized) runs. Not directly comparable to current results.

| File | Encoder | Probe | Note |
|---|---|---|---|
| random_d3_normfix.md | Random-init posfix | d=3 | d=3 overfits catastrophically (train AUC 1.000 by ep10-15) |
| random_d3_linear.md | Random-init pre-posfix | d=3 linear | Broken pos_embed era |
| random_d2_linear.md | Random-init pre-posfix | d=2 linear | Broken pos_embed era |
| imagenet_ep*_d3_mlp.md | ImageNet-init pre-posfix | d=3 MLP | ImageNet-init deprecated; switching to DINO-init |

## Key lessons from this branch

1. **Probe capacity isn't the bottleneck.** d=3 (21M) gave ep25 Val AUC 0.8437; d=1 (7M) gave 0.8460. 3× the params ≈ 0 AUC. Encoder features are the limit. See [research_log.md #5](../../research_log.md).
2. **Overfit dynamics are intrinsic.** Even with wd=0.05, dropout=0.2, d=1 → train AUC still hits 0.99+ by epoch 10-15. Known field-wide issue ([Attention, Please! ICLR 2026](https://arxiv.org/abs/2506.10178)).
3. **Frozen probe has a ceiling.** ~0.85-0.87 on this data. Fine-tuning is the lever for the next 3-5% (see unfrozen README).
