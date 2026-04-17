# Frozen Linear Probe Sweep — Random-init posfix (d=1, literature-tuned)

Linear probe sweep across 4 pretraining checkpoints from the random-init posfix run (`ijepa-results/patch_vit_base_ps16_ep100_bs64_lr0.00025_20260411_063607`). Probe depth=1 following I-JEPA paper convention, with small-data regularization (weight_decay=0.05, dropout=0.2) and literature-scaled LR.

**AML job**: `busy_roof_xjmvcyb7pm` — completed 2026-04-17.

## Config

| Parameter | Value |
|---|---|
| Probe | AttentiveProbe d=1 + LinearHead |
| Probe params | 7.17M (trainable) |
| Head params | 2305 (trainable) |
| Encoder | Frozen ViT-B/16 (random-init I-JEPA) |
| Num slices | 100 (linspace 0..199) |
| Batch size | 256 |
| Epochs | 50, patience 15 |
| Warmup | 5 epochs (10%) |
| LR_probe = LR_head | 4e-4 (linear-scaled from 1e-3 @ bs=1024) |
| Weight decay | 0.05 |
| Dropout (probe) | 0.2 |
| Optimizer | AdamW + cosine schedule |
| Mode | Sequential on GPU 0 (parallel 4-GPU version hung; see research_log.md #1) |

## Results

| Checkpoint | Best Val AUC | Test AUC | Best Epoch | Sensitivity | Specificity |
|---|---|---|---|---|---|
| ep25  | 0.8460 | 0.8558 | 8 | 0.609 | 0.910 |
| ep50  | 0.8452 | 0.8611 | 4 | 0.808 | 0.731 |
| ep75  | 0.8580 | 0.8691 | 4 | 0.772 | 0.791 |
| **ep100** | **0.8597** | **0.8706** | **4** | **0.821** | **0.716** |

**Winner: ep100** (monotonic improvement with pretraining length; +0.015 Test AUC from ep25 to ep100).

## Plots

![Val AUC per epoch](../../../../results/downstream/linear_sweep_random_posfix_d1/val_auc_per_epoch.png)

![Train vs Val loss grid](../../../../results/downstream/linear_sweep_random_posfix_d1/train_val_loss_grid.png)

![AUC per checkpoint](../../../../results/downstream/linear_sweep_random_posfix_d1/auc_per_checkpoint.png)

## Observations

1. **Encoder is the bottleneck, not probe.** d=3 (21M params, earlier run) gave Val AUC 0.8437 on ep25. d=1 (7M params, this run) gave 0.8460. 3× the capacity moved the needle by 0.002. Consistent with `lessons_learned.md` item #6.

2. **Overfit dynamics persist even with literature-standard regularization.** Train AUC hits 0.99+ by epoch 10-15, val loss rises 2-3× from best, val AUC peaks early (epoch 4-8) then drifts. This is [a known field-wide issue with attentive probes](https://arxiv.org/abs/2506.10178) on small datasets.

3. **Pretraining produces monotonically better features.** ep25 → ep100 on Test AUC: 0.8558 → 0.8706 (+0.015). Small but consistent. Shows pretraining hasn't plateaued within 100 epochs.

4. **Val AUC is literature-competitive for frozen probe.** RETFound on PAPILA (fundus) reports 0.86 frozen; Zhou 2025 reports 0.76-0.79 avg across 10 retinal tasks. Our 0.86 on FairVision OCT glaucoma is in the upper range.

5. **Sensitivity / Specificity tradeoff varies across checkpoints.** ep25 is conservative (0.61/0.91); later checkpoints more balanced (~0.80/0.75). Same probe hyperparameters, different feature regimes.

## Next Steps

- Fine-tune on **ep100** checkpoint (submitted as `shy_star_12ywn5x5g8` with 50 epochs, 10 warmup, 15 patience, LR_encoder=1e-5, past_warmup gate on patience). Expected uplift to ~0.89-0.91 Test AUC based on Zhou 2025's fine-tune-vs-LP gap.
- (Optional ablation) Mean-pool + linear head — tests whether the d=1 attentive probe is earning its 7M params.
- Cross-dataset evaluation (Kermany) pending.

## Comparison to earlier runs

| Run | Probe | WD | Dropout | LR | Best ep25 Val AUC |
|---|---|---|---|---|---|
| `random_d3_normfix.md` (d=3, wd=0) | d=3 AttentiveProbe | 0 | 0.1 | 1e-4/1e-3 | 0.8437 |
| **This run (d=1, literature-tuned)** | d=1 AttentiveProbe | 0.05 | 0.2 | 4e-4 single | **0.8460** |

d=1 with regularization is marginally better and has less overfit-collapse. Main value is establishing a literature-aligned baseline rather than gaining AUC.
