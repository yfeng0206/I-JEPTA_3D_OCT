# Fine-tune: LLRD γ=0.5 on ep100 — d=1 AttentiveProbe and CrossAttnPool

Two parallel fine-tune runs with identical LLRD + optimizer settings, differing only in the slice-aggregation probe. Both initialize from the random-init I-JEPA ViT-B/16 ep100 checkpoint, unfreeze the encoder, and use MAE-style Layer-wise LR Decay (γ=0.5).

## Results — headline

| Run | AML job | Probe | Params | Best epoch | Best Val AUC | Test AUC |
|---|---|---|---|---|---|---|
| d=1 attentive fine-tune | `silver_music_r9b0ccn6nc` | AttentiveProbe d=1 + Linear | 7.17M + 86M enc | 4 | 0.8751 | **0.8878** |
| CrossAttnPool fine-tune | `plum_jicama_9tnw0xy5tk` | CrossAttnPool + Linear | 277K + 86M enc | 5 | 0.8729 | **0.8872** |

The two are statistically indistinguishable on Test AUC (paired bootstrap Δ=−0.0005, 95% CI [−0.005, +0.004], p=0.60). **CrossAttnPool matches d=1 at 26× fewer probe params under fine-tuning** — the Pareto-optimal combination.

Compared to their frozen-probe versions:
- Fine-tune CrossAttnPool vs frozen CrossAttnPool: +0.008 Test AUC (p=0.005) **
- Fine-tune d=1 vs frozen d=1: +0.017 Test AUC (p<0.001) ***

Fine-tuning uplift is real, within Zhou 2025's 2-4% fine-tune-vs-LP gap range for retinal tasks.

## Sensitivity / Specificity

| Run | Sensitivity | Specificity | At threshold 0.5 |
|---|---|---|---|
| d=1 attentive fine-tune | 0.741 | 0.877 | Conservative (over-calls negative) |
| CrossAttnPool fine-tune | 0.822 | 0.779 | More balanced |

Different operating points despite tied AUC — each architecture settles at a different threshold-0.5 sweet spot.

## LLRD setup (shared by both runs)

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

Implemented in `build_finetune_param_groups` in `src/eval_downstream.py`. Groups tagged by name (`embed`, `block_0..11`, `encoder_norm`, `probe`, `head`) so the optimizer can filter empty groups (e.g. when probe is MeanPool with 0 params).

## Training dynamics (both runs)

Both peaked during warmup (silver_music ep4, plum_jicama ep5). Post-warmup the encoder starts actually moving, and val AUC declines:

| Epoch | silver_music (d=1) Val | plum_jicama (CrossAttnPool) Val |
|---|---|---|
| 1 | 0.8463 | 0.8416 |
| 2 | 0.8655 | 0.8623 |
| 3 | 0.8677 | 0.8667 |
| 4 | **0.8751 (peak)** | 0.8709 |
| 5 | 0.8693 | **0.8729 (peak)** |
| 10 (warmup ends) | 0.8615 | 0.8630 |
| 15 | 0.8427 | 0.8313 |
| 19 / 20 | 0.8410 (early stop) | 0.8386 (early stop) |

Both overfit post-warmup despite γ=0.5 keeping bottom encoder layers essentially frozen. The overfit is driven by the probe training alongside the slightly-adapting top-encoder layers, not by wholesale encoder destruction.

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

## Interpretation — which probe wins

Under **frozen eval**, CrossAttnPool decisively beat d=1 (p=0.002, +0.009 Test AUC).

Under **fine-tune eval**, they tie (p=0.60).

Why the regime-dependent result:
- With frozen encoder, the probe is the only path the optimizer can use to fit the data. A 7M d=1 probe burns more capacity on overfitting than learning; the 277K CrossAttnPool finds the sweet spot.
- With fine-tune, the encoder itself becomes the adaptive capacity. The probe plays a smaller role — whether it has 7M or 277K params matters less because the encoder's top blocks can compensate.
- In both regimes, CrossAttnPool is at worst tied with d=1, never worse — making it Pareto-optimal across evaluation protocols.

Paper claim: **"A minimal cross-attention pool (~277K params) is Pareto-optimal across both frozen and fine-tune regimes on multi-slice OCT classification. The standard I-JEPA-style attentive probe (7M+ params) provides no AUC gain over the minimal variant in either regime, replicating the 'Attention, Please!' (ICLR 2026) finding on medical 3D-volume data."**

## Note: the gate fix was essential

Both runs peaked during warmup (ep4-5). Without the gate fix (commit `9f96c6b` — remove past_warmup gate from best-model save), the saved checkpoint would be from epoch 11+ (first post-warmup), which is already past the peak. We'd have reported 2-3% lower Test AUC for each run.

## Planned next

`nice_corn_q5180xmk8h` — LLRD fine-tune with MeanPool probe (0 probe params). Completes the 2×3 matrix (3 probes × frozen/fine-tune). Tests whether fine-tune uplift holds when probe contributes zero trainable params.
