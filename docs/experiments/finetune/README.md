# Fine-tune Experiments (Unfrozen Encoder)

Encoder unfrozen. Each of its 12 transformer blocks gets its own learning rate via Layer-wise LR Decay (LLRD), MAE-style. Probe and head train at the full base LR.

DDP on 4× T4 (16 GB each), batch_size=1/GPU, grad accumulation=4 → effective batch=16.

## Current run

| Run | AML job | Encoder init | Probe | Val AUC | Test AUC | Detail |
|---|---|---|---|---|---|---|
| LLRD γ=0.5 on ep100, d=1 attentive | `silver_music_r9b0ccn6nc` | Random-init SSL ep100 | AttentiveProbe d=1 + Linear | **0.8751** (ep4) | **0.8878** | [llrd.md](llrd.md) |
| LLRD γ=0.5 on ep100, CrossAttnPool | `plum_jicama_9tnw0xy5tk` | Random-init SSL ep100 | CrossAttnPool + Linear (277K) | running (ep5 peak 0.8729) | running | in-flight |
| LLRD γ=0.5 on ep100, MeanPool | queued next | Random-init SSL ep100 | MeanPool + Linear (0 probe params) | planned | planned | sequential after plum_jicama |

The last row completes a 2×3 matrix (frozen × fine-tune, 3 probes). Tests whether fine-tune uplift is probe-invariant or probe-dependent.

Beats the frozen d=1 baseline (Test 0.8706) by **+0.017 Test AUC**, within Zhou 2025's 2-4% fine-tune-vs-LP gap for retinal tasks.

## Key lesson

Fine-tune v1 (LLRD γ=0.65, lr=4e-4) overfit post-warmup: peaked during warmup at Val 0.878 but post-warmup best was only 0.867. Root cause: 6K volumes is too small for full fine-tune of 86M params, even with γ=0.65.

Fix in v2 (this run): stronger LLRD (γ=0.5 → bottom encoder layers effectively frozen), lower base LR (4e-4 → 2e-4), and a gate fix so `best_model.pt` tracks improvement across warmup epochs (not just post-warmup). Result: v2 picked ep4 as best, and ep4's test AUC is the real uplift.

Full writeup: [llrd.md](llrd.md).
