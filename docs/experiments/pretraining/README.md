# Pretraining Experiments

Summary of I-JEPA pretraining runs on 600K OCT slices for glaucoma detection.

## Runs Overview

| Run | Init | LR | Outcome | Best Val Loss | Notes |
|-----|------|----|---------|---------------|-------|
| [Run 1](run1_random_lr0005.md) | Random | 0.0005 | Early stopped ep26 | 0.2081 (ep11) | LR too high for OCT |
| [Run 2](run2_random_lr00025.md) | Random | 0.00025 | Bug stopped ep9 | 0.1197 (ep1, artificial) | Early stopping bug |
| [Run 3](run3_random_resume.md) | Random (resume) | 0.00025 | Converged ep11 | 0.1586 (ep11) | Best random-init encoder |
| [Run 4](run4_imagenet_gentle.md) | ImageNet | 0.0001 | Collapsed | 0.008 (collapsed) | Gentle LR caused collapse |
| [Run 5](run5_imagenet_100ep.md) | ImageNet | 0.00025 | Completed 100ep | ~0.25 plateau | Best ImageNet-init encoder |

## Key Takeaways

1. **OCT requires lower LR than ImageNet**: Correlated gradients from less diverse data make the effective LR higher than the nominal value.
2. **Early stopping must ignore pre-warmup epochs**: EMA target has not diverged yet at ep1, producing artificially low val_loss.
3. **ImageNet init needs aggressive, not gentle, tuning**: Gentle LR preserves the ImageNet representation, which collapses on OCT data. Higher LR forces the model to restructure.
4. **I-JEPA loss plateau is normal**: Pretraining loss does not correlate with downstream quality. Always evaluate with downstream probes.
5. **Random init (Run 3) outperformed ImageNet init (Run 5) on downstream tasks**: Run 3 encoder used in F1/F2/U1/U2; Run 5 ep32 achieved 0.774 AUC frozen but ep99 degraded to 0.685.
