# Fine-tune: LLRD γ=0.5, d=1 AttentiveProbe on ep100

Fine-tune the random-init I-JEPA ViT-B/16 (ep100) with MAE-style Layer-wise LR Decay. d=1 AttentiveProbe + LinearHead on top. This is the current best result: Test AUC 0.8878.

AML job: `silver_music_r9b0ccn6nc` — completed 2026-04-18.

## Result

| Metric | Value |
|---|---|
| Best epoch | 4 |
| Best Val AUC | 0.8751 |
| **Test AUC** | **0.8878** |
| Test loss | 0.4057 |
| Sensitivity | 0.741 |
| Specificity | 0.877 |
| Training epochs completed | 19 (early-stopped from patience=15 after no improvement since ep4) |

Comparison to frozen probe on the same encoder:

| Setup | Val AUC | Test AUC | Gain |
|---|---|---|---|
| Frozen d=1 probe, ep100 | 0.8597 | 0.8706 | — |
| **Fine-tune LLRD γ=0.5, ep100** | **0.8751** | **0.8878** | **+0.0172 Test AUC** |

## LLRD setup (MAE-style)

For ViT-B with 12 transformer blocks and γ=0.5, base LR 2e-4:

```
Layer                       Effective LR          Role
------------------------------------------------------------------
patch_embed + pos_embed     1.48e-09             essentially frozen
encoder.blocks[0]  deepest  2.28e-09             essentially frozen
encoder.blocks[5]  middle   1.96e-07             slow update
encoder.blocks[11] top      1.00e-04             moderate update
encoder.norm                2.00e-04             base LR
probe + head                2.00e-04             base LR
```

Implemented in `build_finetune_param_groups` in `src/eval_downstream.py`. The convention: groups[0] = deepest encoder layer, groups[-2] = probe, groups[-1] = head. Logging reads those positions without branching on mode.

## Training dynamics

Best epoch is ep4, during warmup. Post-warmup encoder movement hurt val AUC (classic small-data fine-tune overfit). The `past_warmup` gate on best-model save was removed in this run (fix commit `9f96c6b`) so the real peak at ep4 gets saved instead of being ignored because "training was still warming up."

| Epoch | Train loss | Val loss | Val AUC |
|---|---|---|---|
| 1 | 0.5342 | 0.4724 | 0.8463 |
| 2 | 0.4467 | 0.4416 | 0.8655 |
| 3 | 0.4183 | 0.4541 | 0.8677 |
| **4** | **0.3979** | **0.4291** | **0.8751 (peak)** |
| 5 | 0.3778 | 0.4324 | 0.8693 |
| 10 | 0.3017 | 0.5579 | 0.8615 (end of warmup) |
| 15 | 0.2074 | 0.7045 | 0.8427 (overfit settling in) |
| 19 | 0.1568 | 0.6998 | 0.8410 (early-stop) |

Train loss keeps falling; val loss climbs after ep4. Encoder is memorizing the training distribution. Patience=15 correctly detects "no improvement since ep4" and stops at ep19.

## Config

| Parameter | Value | Reasoning |
|---|---|---|
| Base LR | 2e-4 | Halved from v1's 4e-4 |
| LLRD γ | 0.5 | Stronger decay than MAE's γ=0.65 default, appropriate for 6K-volume fine-tune |
| Weight decay | 0.05 | MAE / RETFound standard |
| Dropout (probe) | 0.2 | Small-data regularization |
| Batch size / GPU | 1 | OOM budget with unfrozen encoder + 64 slices |
| Grad accumulation | 4 | Effective batch = 16 |
| Epochs | 50 | Zhou 2025 fine-tune convention |
| Patience | 15 | Gated on `past_warmup` for early-stop trigger only |
| Warmup | 10 epochs | MAE 20%-of-epochs convention |
| Num slices | 64 | Max fitting with encoder grads on T4 16GB |
| Probe depth | 1 | AttentiveProbe d=1 (7.17M) |
| Head | Linear | I-JEPA paper protocol |

## What v1 got wrong (preserved for the record)

v1 (`willing_yogurt_6t1cvqhy7w`): LLRD γ=0.65, lr=4e-4, past_warmup gate on best-save.

- Hit Val 0.8781 at ep4 during warmup — genuine peak, but the gate blocked it from being saved.
- Post-warmup the encoder moved enough to overfit; best post-warmup was ep13 at Val 0.8665.
- Patience triggered, test AUC for ep13 was ~0.88 (estimated; job was cancelled + restarted before test completed).

Fix: relax LLRD (γ=0.65 → 0.5), halve LR (4e-4 → 2e-4), remove best-save gate. v2 (this run) captures the ep4 peak cleanly.

See research_log.md #15 for full v1 → v2 diagnosis.
