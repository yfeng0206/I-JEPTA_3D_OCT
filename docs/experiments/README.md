# Experiment Log

All experiments for I-JEPA on FairVision OCT glaucoma classification. Each page contains config, training curves, results, and analysis.

## Pretraining Runs

Self-supervised I-JEPA pretraining on 600K OCT slices.

| Run | Init | Epochs | Best Val Loss | Status | Details |
|-----|------|--------|--------------|--------|---------|
| Run 1 | Random | 26 (stopped) | 0.2081 (ep11) | LR too high | [details](pretraining/run1_random_lr0005.md) |
| Run 2 | Random | 9 (stopped) | 0.1197 (ep1, bug) | Early stop bug | [details](pretraining/run2_random_lr00025.md) |
| Run 3 | Random | 18 (converged) | 0.1586 (ep11) | Best random-init | [details](pretraining/run3_random_resume.md) |
| Run 4 | ImageNet | 30 (ep30, crashed) | ~0.008 (collapsed) | Collapsed, gentle LR | [details](pretraining/run4_imagenet_gentle.md) |
| Run 5 | ImageNet | 100 (completed) | plateau ~0.25 | Completed, all ckpts | [details](pretraining/run5_imagenet_100ep.md) |

## Downstream: Frozen Probe

Frozen ViT-B/16 encoder + AttentiveProbe + classification head. Features pre-computed once and cached.

| Run | Encoder Checkpoint | Probe | Head | Slices | Val AUC | Test AUC | Details |
|-----|-------------------|-------|------|--------|---------|----------|---------|
| F1 | Random→SSL ep11 | d=2 | Linear | 100 | 0.744 | 0.733 | [details](downstream/frozen/random_d2_linear.md) |
| F2 | Random→SSL ep11 | d=3 | Linear | 100 | 0.752 | 0.734 | [details](downstream/frozen/random_d3_linear.md) |
| **F3** | **ImageNet→SSL ep32** | **d=3** | **MLP** | **100** | **0.799** | **0.774** | [details](downstream/frozen/imagenet_ep32_d3_mlp.md) |
| F4 | ImageNet→SSL ep50 | d=3 | MLP | 100 | 0.679 | 0.706 | [details](downstream/frozen/imagenet_ep50_d3_mlp.md) |
| F5 | ImageNet→SSL ep75 | d=3 | MLP | 100 | 0.664 | 0.695 | [details](downstream/frozen/imagenet_ep75_d3_mlp.md) |
| F6 | ImageNet→SSL ep99 | d=3 | MLP | 100 | 0.659 | 0.685 | [details](downstream/frozen/imagenet_ep99_d3_mlp.md) |

## Downstream: Fine-tuning (Unfrozen Encoder)

Encoder unfrozen with low LR (5e-6), DDP on 4x T4 GPUs, effective batch=16.

| Run | Encoder Checkpoint | Probe | Head | Slices | Val AUC | Test AUC | Details |
|-----|-------------------|-------|------|--------|---------|----------|---------|
| U1 | Random→SSL ep11 | d=2 | Linear | 32 | 0.819 | N/A* | [details](downstream/unfrozen/random_d2_s32.md) |
| U2 | Random→SSL ep11 | d=3 | Linear | 64 | 0.815 | N/A* | [details](downstream/unfrozen/random_d3_s64.md) |
| U3 | ImageNet→SSL ep32 | d=2 | MLP | 32 | 0.826 | 0.828 | [details](downstream/unfrozen/README.md) |
| U4 | ImageNet→SSL ep32 | d=2 | MLP | 64 | 0.832 | 0.829 | [details](downstream/unfrozen/README.md) |
| **U5** | **ImageNet→SSL ep32** | **d=3** | **MLP** | **32** | **0.828** | **0.829** | [details](downstream/unfrozen/README.md) |
| U6 | ImageNet→SSL ep32 | d=3 | MLP | 64 | 0.832 | 0.829 | [details](downstream/unfrozen/README.md) |
