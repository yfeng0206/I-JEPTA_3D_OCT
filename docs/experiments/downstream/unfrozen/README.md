# Fine-tuning (Unfrozen Encoder)

## Summary

Fine-tune ViT-B/16 end-to-end with **MAE-style Layer-wise LR Decay (LLRD, γ=0.65)** + d=1 AttentiveProbe + LinearHead. Encoder is unfrozen: each of its 12 transformer blocks gets its own LR (deepest ~1.5e-6, topmost ~2.6e-4); probe and head train at the full base LR of 4e-4.

DDP on 4× T4 (16 GB each), batch_size=1/GPU, grad accum=4 → effective batch=16.

## Current run

| Run | AML job | Init | Probe | Slices | Val AUC | Test AUC | Status |
|---|---|---|---|---|---|---|---|
| finetune_ep100_d1_llrd | `willing_yogurt_6t1cvqhy7w` | Random-init SSL ep100 | d=1 attn + linear | 64 | — | — | running |

Picks up the winner of `random_posfix_d1_sweep` (ep100, best frozen Val/Test AUC).

## Config

| Parameter | Value | Reason |
|---|---|---|
| Base LR | 4e-4 | Linear-scaled from 1e-3 @ bs=1024 |
| Layer decay γ | 0.65 | MAE standard for ViT-B |
| Weight decay | 0.05 | MAE / RETFound standard |
| Dropout (probe) | 0.2 | Same as frozen sweep |
| Batch size / GPU | 1 | OOM budget with encoder grads |
| Grad accumulation | 4 | Effective batch = 16 |
| Epochs | 50 | Zhou 2025 / RETFound standard |
| Patience | 15 | Gated on `past_warmup` (`epoch > warmup_epochs`) |
| Warmup | 10 epochs (20%) | MAE convention |
| Num slices | 64 | Max that fits with encoder grads on T4 16GB |
| Probe depth | 1 | AttentiveProbe d=1 (7.17M) |
| Head | Linear | I-JEPA paper protocol |
| AMP | fp16 autocast | Memory + speed |

LLRD LR distribution (base=4e-4, γ=0.65, ViT-B with 12 blocks):

```
embed / pos_embed   1.48e-6   (base × 0.65^13)
block[0]  deepest   2.28e-6
block[5]  middle    1.96e-5
block[11] top       2.60e-4   (base × 0.65)
encoder.norm        4.00e-4   (base)
probe + head        4.00e-4   (base)
```

## Expected

Literature (Zhou 2025, RETFound) shows fine-tune beats frozen probe by 3-5% AUC on retinal tasks. Baseline to beat: **0.8706 Test AUC** from the frozen d=1 sweep on ep100. Target: 0.89-0.91.

## Planned follow-ups

- Flat-encoder-LR ablation (disable LLRD, set `layer_decay=1.0`) to quantify the LLRD contribution.
- Phase 3: fine-tune from DINO-init continuation checkpoint once available.
