# Experiment Log

All experiments for I-JEPA on FairVision OCT glaucoma classification. Each page contains config, training curves, results, and analysis.

> **Note:** All downstream results use the corrected evaluation pipeline with ImageNet normalization. Earlier results without normalization were significantly underestimated (see [lessons learned](../lessons_learned.md#10-imagenet-normalization-mismatch-caused--10-auc-in-frozen-probe)). We use probe depth d=3 and the ep32 (best SSL) checkpoint throughout, having confirmed that d=3 outperforms d=2 and ep32 is the best checkpoint (downstream AUC degrades monotonically with continued pretraining past ep32).

## Pretraining Runs

Self-supervised I-JEPA pretraining on 600K OCT slices.

| Run | Init | Epochs | Best Val Loss | Status | Details |
|-----|------|--------|--------------|--------|---------|
| Run 1 | Random | 26 (stopped) | 0.2081 (ep11) | LR too high | [details](pretraining/run1_random_lr0005.md) |
| Run 2 | Random | 9 (stopped) | 0.1197 (ep1, bug) | Early stop bug | [details](pretraining/run2_random_lr00025.md) |
| Run 3 | Random | 18 (converged) | 0.1586 (ep11) | Best random-init | [details](pretraining/run3_random_resume.md) |
| Run 4 | ImageNet | 30 (ep30, crashed) | ~0.008 (collapsed) | Collapsed, gentle LR | [details](pretraining/run4_imagenet_gentle.md) |
| Run 5 | ImageNet | 100 (completed) | plateau ~0.25 | Completed, all ckpts | [details](pretraining/run5_imagenet_100ep.md) |

## Downstream: Frozen Probe (with normalization fix)

Frozen ViT-B/16 encoder + AttentiveProbe(d=3) + MLP head. Features pre-computed once and cached. All runs use 100 slices, batch size 64, 100 epochs, WD=0, patience=20.

| Run | Encoder Init | Val AUC | Test AUC | Status | Details |
|-----|-------------|---------|----------|--------|---------|
| **F1** | **Random→SSL ep11** | **0.828** | **0.834** | **done** | [details](downstream/frozen/README.md) |
| F2 | ImageNet→SSL ep32 | -- | pending | running | [details](downstream/frozen/README.md) |

*Previous experiments (d=2, ep50/75/99, without normalization) are archived in the [frozen probe page](downstream/frozen/README.md) for reference. ep32 was confirmed as the best checkpoint: AUC degrades monotonically from ep32 (0.774) → ep50 (0.706) → ep75 (0.695) → ep99 (0.685).*

## Downstream: Fine-tuning (Unfrozen Encoder, with normalization fix)

Encoder unfrozen with low LR (5e-6), DDP on 4x T4 GPUs, effective batch=16. All use probe d=3, MLP head.

| Run | Encoder Init | Slices | Val AUC | Test AUC | Status |
|-----|-------------|--------|---------|----------|--------|
| U1 | Random→SSL ep11 | 64 | -- | pending | queued |
| U2 | ImageNet→SSL ep32 | 64 | -- | pending | queued |
| U3 | Random→SSL ep11 | 32 | -- | pending | planned |
| U4 | ImageNet→SSL ep32 | 32 | -- | pending | planned |

*Previous unfrozen experiments (d=2, without normalization) reached 0.819-0.829 test AUC. These reruns with corrected normalization will establish the true baseline. We compare 32 vs 64 slices to measure the impact of doubling slice count.*
