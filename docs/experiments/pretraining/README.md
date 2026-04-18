# Pretraining Experiments

Self-supervised I-JEPA pretraining on 600K OCT slices (FairVision Training split, 6K volumes × 100 slices).

## Runs

| Run | Init | LR | Epochs | Warmup | EMA | Status |
|---|---|---|---|---|---|---|
| [Random-init 100ep](random_100ep.md) | Random | 0.00025 | 100 | 5 | 0.996→1.0 | **completed** — current baseline |
| DINO-init continuation | DINOv2 or DINOv3 ViT-B/16 | TBD | 100 | TBD | 0.996→1.0 | planned (Phase 3) |

Shared config: ViT-B/16, batch 64×4 GPUs × 2 accum = 512 effective, weight_decay 0.04→0.4 cosine, no early stopping.

**ImageNet-init is off the roadmap.** Zhou 2025 ([arxiv 2509.03421](https://arxiv.org/abs/2509.03421v1)) shows DINO-family continuation beats both MAE-family and ImageNet-supervised init on retinal tasks. Phase 3 switches to DINO-init. Rationale in `research_log.md` #9, #12.

## Diagnostic plots

Per-epoch plots under [`results/pretraining/pretrain_random_posfix/`](../../../results/pretraining/pretrain_random_posfix/):

- `train_val_loss.png` — train & val loss. **I-JEPA loss increases as EMA target learns harder representations — not a quality signal.**
- `rep_diversity.png` — representation diversity (lower = better; 1.0 = collapsed).
- `cos_sim.png` — predictor-target cosine similarity (stable 0.78-0.87 = healthy).
- `diagnostics_all.png` — all four metrics in a 2×2 grid.

Regenerate: `python scripts/plot_pretraining.py --csv <log.csv> --stdout <stdout.log> --output <dir> --title <name>`.

## Key takeaways (from random_100ep + research_log.md)

1. **Peak LR 0.00025 for OCT + effective batch 512.** Correlated gradients from less diverse OCT data make the effective LR higher than nominal; I-JEPA's default 0.0005 for ImageNet is too hot here.
2. **Warmup gate for early-stopping AND best-checkpoint save.** Pre-warmup epochs have artificially low val_loss because EMA target hasn't diverged from online encoder. See `lessons_learned.md` #2 + research_log.md #4.
3. **I-JEPA loss goes up with training.** Monitor `rep_diversity` and `cos_sim` instead.
4. **No early stopping.** Literature standard (RETFound, V-JEPA) is fixed-epoch. We run full 100 epochs and save checkpoints every 25.
5. **Downstream AUC is the quality signal.** The d=1 linear probe sweep across ep25/50/75/100 picks the best encoder. See [frozen/d1_sweep.md](../frozen/d1_sweep.md).
