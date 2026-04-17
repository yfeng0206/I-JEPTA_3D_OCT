# Fine-tuning Experiments (Unfrozen Encoder)

## Summary

Fine-tune ViT-B/16 end-to-end with **MAE-style Layer-wise LR Decay (LLRD, γ=0.65)** and a d=1 AttentiveProbe + LinearHead. Encoder is unfrozen and each of its 12 transformer blocks gets a different LR (deepest ~1.5e-6, topmost ~2.6e-4); probe and head train at the full base LR of 4e-4.

DDP on 4× T4 (16 GB each), batch_size=1 per GPU, grad accum=4 → effective batch=16.

## Current Run

| Run | AML Job | Init | Probe | Slices | Val AUC | Test AUC | Status |
|---|---|---|---|---|---|---|---|
| finetune_ep100_d1_llrd | `willing_yogurt_6t1cvqhy7w` | Random-init SSL ep100 | d=1 attn + linear | 64 | pending | pending | running |

Picks up from the winner of `random_posfix_d1_sweep` (ep100 checkpoint, best frozen Val/Test AUC).

## Configuration

| Parameter | Value | Reasoning |
|---|---|---|
| Base LR | 4e-4 | Linear-scaled from 1e-3 @ bs=1024 (matches frozen sweep) |
| Layer decay γ | 0.65 | MAE standard for ViT-B |
| Weight decay | 0.05 | MAE/RETFound standard |
| Dropout (probe) | 0.2 | Same as frozen sweep |
| Batch size / GPU | 1 | OOM budget with encoder grads |
| Grad accumulation | 4 | Effective batch = 16 |
| Epochs | 50 | Matches Zhou 2025 fine-tune recipe |
| Patience | 15 | Gated on `past_warmup` (epoch > warmup_epochs) |
| Warmup | 10 epochs (20%) | MAE convention |
| Num slices | 64 | Max that fits with encoder grads on T4 16GB |
| Probe depth | 1 | d=1 AttentiveProbe (7.17M) |
| Head | Linear | I-JEPA paper protocol |
| AMP | fp16 autocast | Mem + speed |

LLRD LR range at base=4e-4, γ=0.65 for ViT-B (13 layers + head):
```
embed / pos_embed     1.48e-6   (base * 0.65^13)
block[0]  (deepest)   2.28e-6
block[5]  (middle)    1.96e-5
block[11] (top)       2.60e-4
encoder.norm          4e-4
probe + head          4e-4
```

## Expected Outcome

Literature (Zhou 2025, RETFound) shows fine-tune beats frozen probe by 3-5% AUC on retinal tasks. Baseline to beat: **0.8706 Test AUC** from the frozen d=1 sweep on ep100. Target: 0.89-0.91.

## Historical Runs (retained for reference)

Pre-normfix (old eval pipeline without ImageNet normalization), pre-LLRD, d=3 probe. Not comparable to current setup.

| Run | Encoder Init | Probe | Slices | Val AUC | Test AUC |
|---|---|---|---|---|---|
| U1 (old) | Random→SSL ep11 | d=2 | 32 | 0.819 | N/A* |
| U2 (old) | Random→SSL ep11 | d=3 | 64 | 0.815 | N/A* |
| U3 (old) | ImageNet→SSL ep32 | d=2 MLP | 32 | 0.826 | 0.828 |
| U4 (old) | ImageNet→SSL ep32 | d=2 MLP | 64 | 0.832 | 0.829 |
| U5 (old) | ImageNet→SSL ep32 | d=3 MLP | 32 | 0.828 | 0.829 |
| U6 (old) | ImageNet→SSL ep32 | d=3 MLP | 64 | 0.832 | 0.829 |

*N/A: DDP teardown crash during test eval (fixed in current code).*

ImageNet-init runs above are no longer on the forward roadmap — switching to DINO-init continuation for Phase 3.

## Next Steps

1. Complete current `willing_yogurt_6t1cvqhy7w` fine-tune → report Val/Test AUC
2. (Optional ablation) Run fine-tune without LLRD (flat encoder LR) to quantify LLRD gain
3. Phase 3: fine-tune from DINO-init I-JEPA continuation checkpoint (once pretrained)
