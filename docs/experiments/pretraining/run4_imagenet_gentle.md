# Pretraining Run 4: ImageNet Init, Gentle LR (Collapsed)

## Summary

First attempt with ImageNet pretrained ViT-B/16. Used EMA=0.999 and LR=0.0001 (gentle). COLLAPSED -- rep_diversity=0.98 at ep1, loss dropped to 0.008. ImageNet features produced near-identical representations for all OCT patches. Gentle tuning was the wrong approach; the model needed aggressive updates to escape the collapsed ImageNet representation.

## Config

| Parameter | Value |
|-----------|-------|
| Architecture | ViT-B/16 |
| Initialization | ImageNet ViT-B/16 pretrained |
| Learning Rate | 0.0001 |
| EMA Schedule | [0.999, 1.0] |
| Warmup Epochs | 5 |

## Training Log Excerpt

Collapsed at ep1. rep_diversity=0.98, loss=0.008. Run abandoned.

## Key Observations

- With pretrained init, the encoder needs AGGRESSIVE updates to specialize for OCT, not gentle ones. This is counter-intuitive but confirmed by this experiment.
- rep_diversity near 1.0 indicates representation collapse: the model maps all OCT patches to nearly the same embedding.
- The ImageNet representation is a poor starting point for OCT unless the model is pushed hard enough to restructure its feature space.
