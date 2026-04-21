# Fine-tune Experiments (Unfrozen Encoder)

Encoder unfrozen. Each of its 12 transformer blocks gets its own learning rate via Layer-wise LR Decay (LLRD), MAE-style. Probe and head train at the full base LR.

DDP on 4× T4 (16 GB each), batch_size=1/GPU, grad accumulation=4 → effective batch=16.

## Current run

| Run | Encoder init | Probe | Val AUC (peak ep) | Test AUC | Detail |
|---|---|---|---|---|---|
| LLRD γ=0.5 on ep100, d=1 attentive | Random-init SSL ep100 | AttentiveProbe d=1 + Linear | 0.8751 (ep4) | **0.8878** | [llrd.md](llrd.md) |
| LLRD γ=0.5 on ep100, CrossAttnPool | Random-init SSL ep100 | CrossAttnPool + Linear (277K) | 0.8729 (ep5) | **0.8872** | [llrd.md](llrd.md) |
| LLRD γ=0.5 on ep100, MeanPool | Random-init SSL ep100 | MeanPool + Linear (0 probe params) | 0.8717 (ep5) | **0.8868** | [llrd.md](llrd.md) |

**Headline**: all three fine-tune runs are **statistically tied** on Test AUC (paired bootstrap, pairwise two-sided p > 0.6). The probe architecture — from 7.17M params (AttentiveProbe d=1) down to 0 probe params (MeanPool) — makes no measurable difference under fine-tuning on this task.

| Pairwise | Δ Test AUC | 95% CI | two-sided p | Verdict |
|---|---|---|---|---|
| FT d=1 − FT MeanPool | +0.0009 | [−0.004, +0.005] | 0.69 | tied |
| FT CrossAttnPool − FT MeanPool | +0.0004 | [−0.001, +0.002] | 0.63 | tied |
| FT d=1 − FT CrossAttnPool | +0.0005 | [−0.004, +0.005] | 0.81 | tied |

All three fine-tune results significantly exceed the best frozen probe. Even FT + MeanPool (0.8868) beats frozen CrossAttnPool (0.8791) by +0.008 (p=0.013, *).

Fine-tune uplift scales inversely with frozen-probe strength:
- **d=1**: +0.017 uplift (p<0.001) — biggest, because frozen d=1 was the worst (over-parameterized)
- **MeanPool**: +0.012 uplift (p<0.001) — encoder adaptation alone, no probe contribution
- **CrossAttnPool**: +0.008 uplift (p=0.009) — smallest, frozen CrossAttnPool was already best

Interpretation: whatever slice-weighting the attention probes learn in the frozen regime, the encoder's top-block + encoder.norm adaptation absorbs under fine-tune. The three probes converge to a common ceiling of ~0.887 Test AUC on FairVision glaucoma.

## Key lesson

Fine-tune v1 (LLRD γ=0.65, lr=4e-4) overfit post-warmup: peaked during warmup at Val 0.878 but post-warmup best was only 0.867. Root cause: 6K volumes is too small for full fine-tune of 86M params, even with γ=0.65.

Fix in v2 (this run): stronger LLRD (γ=0.5 → bottom encoder layers effectively frozen), lower base LR (4e-4 → 2e-4), and a gate fix so `best_model.pt` tracks improvement across warmup epochs (not just post-warmup). Result: v2 picked ep4 as best, and ep4's test AUC is the real uplift.

Full writeup: [llrd.md](llrd.md).
