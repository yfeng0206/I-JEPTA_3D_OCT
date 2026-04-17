# Pretraining Experiments

Summary of I-JEPA pretraining runs on 600K OCT slices for glaucoma detection.

## Current Runs (posfix branch)

| Run | Init | LR | Epochs | Warmup | EMA | Diagnostics | Status |
|-----|------|----|--------|--------|-----|-------------|--------|
| [Random-init posfix](run6_random_posfix.md) | Random | 0.00025 | 100 | 5 | 0.996→1.0 | [plots](../../../results/pretraining/pretrain_random_posfix/) | **completed** |
| DINO-init (planned) | DINOv2 or DINOv3 ViT-B/16 | TBD | 100 | TBD | 0.996→1.0 | — | Phase 3 |

ViT-B/16, batch 64×4 GPUs×2 accum = 512 effective, weight decay 0.04→0.4, no early stopping. Fixed position encoding (posfix).

**ImageNet-init is deprecated** as the next continuation source. Based on [Zhou 2025](https://arxiv.org/abs/2509.03421v1), DINO-family beats MAE-family and ImageNet-supervised init for retinal tasks. Next pretraining experiment is DINO-init continuation. See `research_log.md` #9, #12 for the reasoning.

### Diagnostic Plots

Each run has 4 per-epoch diagnostic plots:
- **train_val_loss.png** — Train & val loss (I-JEPA loss increases as EMA target learns; NOT a quality signal)
- **rep_diversity.png** — Representation diversity (lower = better; 1.0 = collapsed)
- **cos_sim.png** — Predictor-target cosine similarity (stable 0.78-0.87 = healthy)
- **diagnostics_all.png** — All 4 metrics in a 2×2 grid

Generate plots: `python scripts/plot_pretraining.py --csv <log.csv> --stdout <stdout.log> --output <dir> --title <name>`

## Previous Runs (pre-posfix, broken position encoding)

All runs below had the position encoding bug (all 256 patches received identical positional embeddings). Results are not directly comparable to posfix runs.

| Run | Init | LR | Outcome | Lesson | Details |
|-----|------|----|---------|--------|---------|
| Run 1 | Random | 0.0005 | Diverged after warmup | LR too high for OCT data | [details](run1_random_lr0005.md) |
| Run 2 | Random | 0.00025 | Stopped ep9 | Early stopping bug (counted pre-warmup) | [details](run2_random_lr00025.md) |
| Run 3 | Random | 0.00025 | Converged ep11 | Best random-init with this LR | [details](run3_random_resume.md) |
| Run 4 | ImageNet | 0.0001 | Collapsed | Gentle LR preserves ImageNet features → collapse on OCT | [details](run4_imagenet_gentle.md) |
| Run 5 | ImageNet | 0.00025 | 100 epochs | Aggressive LR works — forces domain adaptation | [details](run5_imagenet_100ep.md) |

## Key Takeaways

1. **OCT requires lower LR than ImageNet**: Correlated gradients from less diverse data make the effective LR higher than the nominal value. Peak LR=0.00025 with effective batch 512.
2. **Early stopping must ignore pre-warmup epochs**: EMA target has not diverged yet at ep1, producing artificially low val_loss. Best checkpoint saving must also be gated on past_warmup (bug found in current run).
3. **ImageNet init collapse may have been caused by broken pos_embed**: All pre-posfix runs had zero positional embeddings. ImageNet-init collapse (rep_diversity=0.98) could be a pos_embed bug, not a fundamental incompatibility.
4. **I-JEPA loss increases as training progresses**: This is EXPECTED — the EMA target learns harder representations. Loss is NOT a quality signal. Monitor rep_diversity and cos_sim instead.
5. **No early stopping for final runs**: Literature standard (RETFound: 800 epochs, US-JEPA: 100 epochs) is fixed-epoch training. We run full 100 epochs and save checkpoints every 25.
6. **Downstream AUC is the only reliable metric**: The checkpoint sweep (ep25/50/75/100) determines which epoch has the best representations.
