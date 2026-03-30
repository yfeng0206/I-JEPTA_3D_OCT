# Lessons Learned

Record of mistakes, failed experiments, and fixes so we don't repeat them.

---

## Pretraining Issues

### 1. LR=0.0005 too high for OCT data (Run 1)
- **What happened:** Model learned well during warmup (epochs 1-11, LR 0.0001-0.0004) but destabilized once LR hit peak 0.0005. Val loss increased monotonically from epoch 12 onward.
- **Root cause:** OCT images are less diverse than ImageNet (all retinal scans), producing more correlated gradients. sqrt scaling formula overestimated the right LR.
- **Fix:** Reduced peak LR to 0.00025. Worked well.
- **Rule:** For OCT data with ViT-B/16 and effective batch=512, peak LR=0.00025 is the sweet spot.

### 2. Early stopping counted pre-warmup epochs (Run 2)
- **What happened:** Epoch 1 had artificially low val_loss=0.1197 because EMA target hadn't diverged yet. No subsequent epoch could beat it. Early stopping triggered at epoch 9 while the model was still improving.
- **Root cause:** Before warmup, both encoders are nearly identical, making prediction trivially easy (low loss). This isn't real performance.
- **Fix:** Only start counting patience after warmup epochs. Added `past_warmup = (epoch + 1) > warmup_epochs`.
- **Rule:** Never count pre-warmup epochs for early stopping in I-JEPA.

### 3. Blocking blob uploads caused NCCL timeout (Run 3, multiple downstream runs)
- **What happened:** Uploading 1.5 GB checkpoint to blob blocked rank 0 for >30 min. Ranks 1-3 waited at next DDP collective, hit NCCL 30-min timeout.
- **Root cause:** Synchronous blob upload in the training loop while other ranks wait.
- **Fix:** Non-blocking uploads via background threads. Final uploads use `blocking=True` only after training loop ends.
- **Rule:** Never do blocking I/O between DDP collective operations.

### 4. DDP early-stop sync bug (downstream fine-tuning)
- **What happened:** Rank 0 hit `break` inside `if is_main:` block before reaching `dist.broadcast()`. Other ranks stuck waiting at broadcast forever.
- **Root cause:** Python `break` exits the loop immediately, skipping the broadcast that all ranks need to reach.
- **Fix:** Use `should_stop` flag instead of `break`. All ranks reach broadcast, then break.
- **Rule:** Never use `break` inside a rank-specific block in DDP. Always broadcast the stop decision first.

### 5. Gradient accumulation misaligned with scheduler/EMA (early Run 1 bug)
- **What happened:** Scheduler and EMA updated every micro-batch instead of every optimizer step. With accum_steps=2, scheduler ran 2x too fast.
- **Root cause:** Scheduler/EMA step was outside the accumulation guard.
- **Fix:** Only step scheduler/EMA when `(itr + 1) % accum_steps == 0`.
- **Rule:** Scheduler, EMA, and optimizer must all step at the same rate.

---

## Downstream Issues

### 6. Frozen encoder rep_diversity is the bottleneck, not probe capacity
- **What happened:** Increasing probe depth from 2 to 3 blocks gave only +0.1% test AUC (0.733 → 0.734).
- **Root cause:** The frozen encoder's representations are the limiting factor. More probe layers can't extract signal that isn't in the features.
- **Lesson:** When frozen probe performance plateaus, the fix is unfreezing the encoder, not adding more probe layers.

### 7. 100 slices OOM with encoder gradients
- **What happened:** Fine-tuning with 100 slices, batch_size=1 OOM'd (15 GB on 16 GB T4). 64 slices worked (~11 GB).
- **Root cause:** All 100 slice activations must stay in memory for backward through the probe. ~120 MB per slice activation graph.
- **Fix:** Use 32-64 slices for fine-tuning. 100 slices only for frozen probe (no grad, uses ~5 MB per slice).
- **Rule:** With ViT-B/16 encoder gradients on T4 16GB: max ~64 slices at batch_size=1.

---

## ImageNet Pretrained Init Issues

### 8. ImageNet ViT produces collapsed features for OCT (ongoing)
- **What happened:** Initialized encoder from ImageNet supervised ViT-B/16. rep_diversity=0.98 at epoch 1 (near-collapse). Loss dropped to 0.008 and barely recovered.
- **Root cause:** ImageNet ViT was trained on colorful natural images. OCT images are grayscale with repetitive structure. All OCT patches activate similar ImageNet features → near-identical representations → trivial prediction task.
- **Attempted fix (failed):** EMA=0.999 (slower target updates) + LR=0.0001 (gentle). This made it WORSE — too gentle to escape collapse.
- **Next attempt:** EMA=0.996 (standard) + LR=0.00025 (proven for OCT) to force faster adaptation away from collapsed ImageNet features.
- **Key insight:** With pretrained init, the encoder needs AGGRESSIVE updates to specialize for the target domain, not gentle ones. The opposite of what we initially assumed.

### 9. Investigated but NOT the cause: AMP and momentum schedule differences
- **AMP mismatch (target fp32, context fp16):** Investigated as potential collapse cause. NOT the issue — PyTorch auto-upcasts for loss computation. We intentionally use fp32 for target (more precise, no backward needed).
- **Cosine vs linear momentum:** Investigated as potential cause. NOT the issue — for EMA range 0.999→1.0, the difference is ~0.0003 max. Negligible.
- **Lesson:** Don't throw fixes at the wall. Identify the actual bottleneck (EMA + LR too gentle for domain shift) before changing working code.

---

## Code Differences from Official I-JEPA

Documented for reference. These are intentional differences, not bugs:

| Aspect | Official | Ours | Impact |
|--------|---------|------|--------|
| Momentum schedule | Linear | Cosine | Negligible for our EMA ranges |
| Target path AMP | Under autocast | Without autocast (fp32) | Slightly more precise targets |
| LayerNorm epsilon | 1e-6 | 1e-5 (PyTorch default) | Minor numerical difference |
| qkv_bias default | False | True | Both use True in practice |
| Pretrained init | Not supported | Custom loading path | Novel, needs careful tuning |
| CLS token | Has interpolation code (buggy for no-CLS) | Direct pos_embed add | Cleaner |

---

## General Rules

1. **I-JEPA loss is NOT a reliable metric.** It can be low due to collapse or high due to diverse targets. Monitor rep_diversity and cos_sim instead.
2. **Early stopping for I-JEPA pretraining is unusual.** Most papers train for fixed epochs (RETFound: 800, US-JEPA: 100). Use validation loss selection like US-JEPA if early stopping is needed.
3. **ImageNet init for I-JEPA is untested territory.** The official code doesn't support it. Domain adaptation (OCT) needs aggressive LR/EMA, not conservative ones.
4. **Always upload checkpoints to blob immediately.** Don't wait until end of training — jobs crash.
5. **DDP cleanup (barrier + destroy) must happen before any rank-specific code.** Otherwise NCCL timeouts are guaranteed.
