# Experiment Log

All experiments for I-JEPA on FairVision OCT glaucoma classification. Each page has config, training curves, results, and analysis.

**Current default probe**: AttentiveProbe d=1 + LinearHead (literature-aligned, I-JEPA paper single-block convention).

## Pretraining

Self-supervised I-JEPA pretraining on 600K OCT slices (100 slices × 6K Training volumes from FairVision).

| Run | Init | LR | Epochs | Status | Details |
|---|---|---|---|---|---|
| Random-init | Random | 0.00025 | 100 | **completed** | [run6_random_posfix](pretraining/run6_random_posfix.md) |
| DINO-init continuation | DINOv2 or DINOv3 ViT-B/16 | TBD | 100 | planned (Phase 3) | — |

See [pretraining README](pretraining/README.md) for hyperparameter choices.

## Downstream: Frozen Probe

Frozen ViT-B/16 encoder + AttentiveProbe(d=1) + LinearHead. Features pre-computed once and cached. All runs use 100 slices.

| Sweep | Encoder Init | Probe | Val AUC | Test AUC | Status | Details |
|---|---|---|---|---|---|---|
| random_posfix_d1_sweep | Random-init ep25/50/75/100 | d=1 attn + linear (~7M) | **0.860** (ep100) | **0.871** (ep100) | completed | [details](downstream/frozen/random_posfix_d1_sweep.md) |
| cross_attn_pool ablation | Random-init ep100 | minimal cross-attn pool (~280K) | — | — | planned | — |
| DINO-init probe | DINOv2/v3 → SSL ep100 | d=1 attn + linear | — | — | Phase 3 | — |

## Downstream: Fine-tuning (Unfrozen Encoder)

Encoder unfrozen with MAE-style LLRD (γ=0.65). DDP 4× T4, effective batch=16, 64 slices.

| Run | Encoder Init | Probe | Val AUC | Test AUC | Status |
|---|---|---|---|---|---|
| finetune_ep100_d1_llrd | Random-init SSL ep100 | d=1 attn + linear | — | — | running (`willing_yogurt_6t1cvqhy7w`) |

See [unfrozen README](downstream/unfrozen/README.md) for config detail.

## Reference

- [research_log.md](../research_log.md) — chronological problem/solution log, paper bibliography, backlog
- [lessons_learned.md](../lessons_learned.md) — mistakes, fixes, invariants
- [architecture.md](../architecture.md) — model architecture spec
