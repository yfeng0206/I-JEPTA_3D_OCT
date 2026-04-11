# Pretraining Experiments

Summary of I-JEPA pretraining runs on 600K OCT slices for glaucoma detection.

## Current Runs (in progress)

| Run | Init | LR | Epochs | Warmup | EMA | Status |
|-----|------|----|--------|--------|-----|--------|
| Random-init | Random | 0.00025 | 100 | 5 | 0.996→1.0 | pending |
| ImageNet-init | ImageNet ViT-B/16 (timm) | 0.00025 | 100 | 5 | 0.996→1.0 | pending |

Both use: ViT-B/16, batch 64×4 GPUs×2 accum = 512 effective, weight decay 0.04→0.4, no early stopping.

## Previous Runs (exploratory)

These early runs informed our hyperparameter choices. Key lessons:

| Run | Init | LR | Outcome | Lesson |
|-----|------|----|---------|--------|
| Run 1 | Random | 0.0005 | Diverged after warmup | LR too high for OCT data |
| Run 2 | Random | 0.00025 | Stopped ep9 | Early stopping bug (counted pre-warmup) |
| Run 3 | Random | 0.00025 | Converged ep11 | Best random-init with this LR |
| Run 4 | ImageNet | 0.0001 | Collapsed | Gentle LR preserves ImageNet features → collapse on OCT |
| Run 5 | ImageNet | 0.00025 | 100 epochs | Aggressive LR works — forces domain adaptation |

## Key Takeaways

1. **OCT requires lower LR than ImageNet**: Correlated gradients from less diverse data make the effective LR higher than the nominal value. Peak LR=0.00025 with effective batch 512.
2. **Early stopping must ignore pre-warmup epochs**: EMA target has not diverged yet at ep1, producing artificially low val_loss.
3. **ImageNet init needs aggressive, not gentle, tuning**: Gentle LR preserves the ImageNet representation, which collapses on OCT data. Higher LR forces the model to restructure.
4. **I-JEPA loss plateau is normal**: Pretraining loss does not correlate with downstream quality. Always evaluate with downstream probes.
5. **No early stopping for final runs**: Literature standard (RETFound: 800 epochs, US-JEPA: 100 epochs) is fixed-epoch training. We run full 100 epochs and save checkpoints every 25.
