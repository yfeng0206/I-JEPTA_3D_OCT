# Fine-tune: LLRD γ=0.5 on ep100 — three probes × one encoder

Three parallel fine-tune runs with identical LLRD + optimizer settings, differing only in the slice-aggregation probe. All three initialize from the random-init I-JEPA ViT-B/16 ep100 checkpoint, unfreeze the encoder, and use MAE-style Layer-wise LR Decay (γ=0.5).

## Results — headline

| Run | Probe | Probe params | Best epoch | Val AUC | Test AUC |
|---|---|---|---|---|---|
| d=1 attentive fine-tune | AttentiveProbe d=1 + Linear | 7.17M | 4 | 0.8751 | **0.8878** |
| CrossAttnPool fine-tune | CrossAttnPool + Linear | 277K | 5 | 0.8729 | **0.8872** |
| MeanPool fine-tune | MeanPool + Linear | **0** (just 2.3K head) | 5 | 0.8717 | **0.8868** |

All three runs land within 0.001 Test AUC of each other. Paired bootstrap (B=2000) confirms they are statistically indistinguishable:

| Pairwise | Δ Test AUC | 95% CI | two-sided p | Verdict |
|---|---|---|---|---|
| d=1 − MeanPool | +0.0009 | [−0.004, +0.005] | 0.69 | tied |
| CrossAttnPool − MeanPool | +0.0004 | [−0.001, +0.002] | 0.63 | tied |
| d=1 − CrossAttnPool | +0.0005 | [−0.004, +0.005] | 0.81 | tied |

**Under fine-tune, probe architecture is irrelevant on this task.** The encoder's top-block + encoder.norm adaptation absorbs whatever slice-weighting the attention probes provide in the frozen regime.

## Fine-tune uplift vs frozen — scales inversely with probe capacity

| Probe | Frozen Test AUC | Fine-tune Test AUC | Uplift | p (two-sided) |
|---|---|---|---|---|
| d=1 AttentiveProbe (7.17M) | 0.8706 | 0.8878 | **+0.0172** | <0.001 *** |
| MeanPool (0 probe params) | 0.8746 | 0.8868 | **+0.0122** | <0.001 *** |
| CrossAttnPool (277K) | 0.8791 | 0.8872 | **+0.0080** | 0.009 ** |

The probe that had the most to recover (d=1, over-parameterized in frozen) gains the most from fine-tuning. The probe that was already near ceiling frozen (CrossAttnPool) gains the least. MeanPool's +0.012 uplift is pure encoder-side adaptation — the probe has zero trainable parameters.

Even fine-tune + MeanPool (0.8868) beats the best frozen probe (CrossAttnPool, 0.8791) by +0.008 Test AUC (p=0.013, *). Fine-tune without any attentive probing still wins.

## Sensitivity / Specificity (threshold 0.5)

| Run | Sensitivity | Specificity |
|---|---|---|
| d=1 attentive fine-tune | 0.741 | 0.877 |
| CrossAttnPool fine-tune | 0.822 | 0.779 |
| MeanPool fine-tune | 0.827 | 0.769 |

Different operating points despite tied AUC. CrossAttnPool and MeanPool land in a similar balanced region; d=1 is conservative at threshold 0.5. All three AUCs are near-identical, so a downstream threshold sweep would reach the same working point from any starting probe.

## LLRD setup (shared by all runs)

For ViT-B with 12 transformer blocks, γ=0.5, base LR 2e-4:

```
Layer                       Effective LR        Role
------------------------------------------------------------------
patch_embed + pos_embed     1.48e-09            essentially frozen
encoder.blocks[0]  deepest  2.28e-09            essentially frozen
encoder.blocks[5]  middle   1.96e-07            slow update
encoder.blocks[11] top      1.00e-04            moderate update
encoder.norm                2.00e-04            base LR
probe + head                2.00e-04            base LR
```

Implemented in `build_finetune_param_groups` in `src/eval_downstream.py`. Groups tagged by name (`embed`, `block_0..11`, `encoder_norm`, `probe`, `head`) so the optimizer can filter empty groups (MeanPool has 0 probe params → its group is dropped from the optimizer entirely).

## Why the peaks are during warmup

All three runs peaked at ep4-5 during the 10-epoch warmup. At ep5, the top encoder block's effective LR is only ~5e-5 (half of its post-warmup value) — the encoder has barely moved. What IS training at full speed during warmup is:
- **encoder.norm** at base LR 2e-4 (LayerNorm scale/bias on the final representation)
- **head** at base LR 2e-4 (Linear(768→1) + LayerNorm)
- For d=1 and CrossAttnPool: **probe** at base LR 2e-4

Post-warmup, the top encoder block reaches LR ~1e-4, starts moving, and the train loss keeps falling (overfit) while val loss climbs. Patience=15 triggers early stopping around ep19-20.

The fact that all three probe architectures converge to the same Test AUC at the same early epoch suggests **the lift is dominated by encoder.norm + head adaptation** — which are identical across the three runs — rather than by probe-specific learning.

## Training dynamics

| Epoch | d=1 Val | CrossAttnPool Val | MeanPool Val |
|---|---|---|---|
| 1 | 0.8463 | 0.8416 | 0.8439 |
| 2 | 0.8655 | 0.8623 | 0.8636 |
| 3 | 0.8677 | 0.8667 | 0.8664 |
| 4 | **0.8751** | 0.8709 | 0.8715 |
| 5 | 0.8693 | **0.8729** | **0.8717** |
| 10 (warmup ends) | 0.8615 | 0.8630 | 0.8639 |
| 15 | 0.8427 | 0.8313 | 0.8321 |
| 19/20 (early stop) | 0.8410 | 0.8386 | 0.8455 |

All three peak at ep4-5, plateau through ep10, then decline as the encoder over-adapts. The gate fix (commit `9f96c6b`) is what preserves the warmup peak — without it, best_model.pt would snapshot post-warmup at ~1 pp lower AUC.

## Shared config

| Parameter | Value |
|---|---|
| Base LR | 2e-4 |
| LLRD γ | 0.5 |
| Weight decay | 0.05 |
| Dropout (probe) | 0.2 |
| Batch size / GPU | 1 |
| Grad accumulation | 4 |
| Effective batch | 16 |
| Epochs / patience | 50 / 15 (gated post-warmup) |
| Warmup | 10 epochs |
| Num slices | 64 (max fitting with encoder grads on T4 16GB) |
| AMP | fp16 autocast |
| GPUs | 4× T4 (DDP) |

## Interpretation — the two-regime story

**Frozen eval**: probe architecture matters. CrossAttnPool beats d=1 (p=0.002, +0.009), beats MeanPool (p=0.04, +0.005). The probe has to do all the work of slice-weighting on its own; d=1 at 7M params overfits the 6K-volume dataset while CrossAttnPool at 277K finds the sweet spot.

**Fine-tune eval**: probe architecture is noise. d=1 / CrossAttnPool / MeanPool all tie at ~0.887 AUC. Encoder top-block + LayerNorm adaptation provides enough additional capacity that the probe doesn't need to do anything sophisticated.

Paper claim:
> *"Probe architecture matters for frozen-probe evaluation only. Under fine-tune, a trivial mean-pool with zero probe parameters matches a 7M AttentiveProbe on multi-slice OCT classification. The common practice of bolting attentive probes onto fine-tuned encoders is unjustified on this task — encoder adaptation absorbs the probe's role."*

## Note: the gate fix was essential

All three runs peaked during warmup (ep4-5). Without the gate fix (commit `9f96c6b` — remove past_warmup gate from best-model save), the saved checkpoint would be from epoch 11+ (first post-warmup), already past the peak. We'd have reported ~1 pp lower Test AUC for each run.

## Completeness

This run (fine-tune + MeanPool) closes the 2×3 matrix: 3 probes × {frozen, fine-tune}. No more cells pending.
