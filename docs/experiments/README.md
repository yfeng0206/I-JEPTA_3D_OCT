# Experiment Log

All experiments for I-JEPA on FairVision OCT glaucoma classification. Each page contains config, training curves, results, and analysis.

We use probe depth d=3 and the best SSL checkpoint throughout, having confirmed that d=3 outperforms d=2 and best checkpoint outperforms later epochs.

## Pretraining Runs

Self-supervised I-JEPA pretraining on 600K OCT slices (100 slices × 6K training volumes).

| Run | Init | LR | Epochs | Status | Details |
|-----|------|----|--------|--------|---------|
| Random-init | Random | 0.00025 | 100 | pretraining | [details](pretraining/README.md) |
| ImageNet-init | ImageNet ViT-B/16 | 0.00025 | 100 | queued | [details](pretraining/README.md) |

See [pretraining page](pretraining/README.md) for hyperparameter choices and lessons from exploratory runs.

## Downstream: Frozen Probe

Frozen ViT-B/16 encoder + AttentiveProbe(d=3) + MLP head. Features pre-computed once and cached. All runs use 100 slices, batch size 64, 100 epochs, WD=0, patience=20.

| Run | Encoder Init | Val AUC | Test AUC | Status | Details |
|-----|-------------|---------|----------|--------|---------|
| F1 | Random→SSL | -- | pending | waiting for pretraining | [details](downstream/frozen/README.md) |
| F2 | ImageNet→SSL | -- | pending | waiting for pretraining | [details](downstream/frozen/README.md) |

## Downstream: Fine-tuning (Unfrozen Encoder)

Encoder unfrozen with low LR (5e-6), DDP on 4x T4 GPUs, effective batch=16. All use probe d=3, MLP head.

| Run | Encoder Init | Slices | Val AUC | Test AUC | Status |
|-----|-------------|--------|---------|----------|--------|
| U1 | Random→SSL | 64 | -- | pending | waiting for pretraining |
| U2 | ImageNet→SSL | 64 | -- | pending | waiting for pretraining |
| U3 | Random→SSL | 32 | -- | pending | waiting for pretraining |
| U4 | ImageNet→SSL | 32 | -- | pending | waiting for pretraining |

We compare 32 vs 64 slices to measure the impact of doubling slice count on classification performance.
