# Lessons Learned

Mistakes, debug-traps, and invariants we've paid to learn. Keep these visible so they don't sneak back in.

---

## Pretraining

### 1. LR=0.0005 too high for OCT
- **What happens:** Model learns fine during warmup, diverges once LR hits peak.
- **Why:** OCT is less diverse than ImageNet, gradients are more correlated, effective LR is higher than nominal. sqrt-scaling from the I-JEPA paper's LR overestimates.
- **Rule:** For OCT + ViT-B/16 + effective batch 512, **peak LR = 0.00025**.

### 2. Pre-warmup val loss is artificially low
- **What happens:** Before warmup ends, the EMA target encoder hasn't diverged from the online encoder. Prediction is trivial → val_loss looks great → nothing ever beats it again → patience counter / best-checkpoint save latches onto epoch 1.
- **Rule:** Every best-AUC / best-loss / patience decision must be **gated on `past_warmup`**. See `train_patch.py` + `eval_downstream.py` fine-tune path (commit `135ba2a`, off-by-one fix `0dcd9d0`).

### 3. Blocking blob uploads stall DDP
- **What happens:** Rank 0 uploads a 1.5 GB checkpoint synchronously, takes >5 min, other ranks block on the next collective → NCCL 30-min timeout → hang.
- **Rule:** Background-thread uploads during training, only `blocking=True` after the training loop ends.

### 4. DDP early-stop `break` skips `dist.broadcast`
- **What happens:** Rank 0 hits `break` inside `if is_main:` before reaching `dist.broadcast()`. Other ranks sit on the broadcast forever.
- **Rule:** Use a `should_stop` flag, broadcast, THEN break on all ranks.

### 5. Grad accumulation must gate scheduler + EMA
- **What happens:** Scheduler / EMA step every micro-batch. With `accum_steps=2`, scheduler runs 2× too fast.
- **Rule:** Only step scheduler / EMA inside `if (itr + 1) % accum_steps == 0:`.

### 6. DDP val-loss must be all-reduced across ranks
- **What happens:** Each rank sees a different `val_loss` (its own shard). Early-stop decisions diverge across ranks → NCCL hang at next collective.
- **Rule:** `dist.all_reduce(sum, count)` → global mean before any rank-comparison decision. Fixed in commit `135ba2a`.

---

## Downstream

### 7. Encoder representations are the bottleneck, not probe capacity
- **What happens:** Probe depth d=3 (21M) vs d=1 (7M) moves Val AUC by ~0.002. 3× capacity, 0× gain.
- **Rule:** When a frozen probe plateaus, don't scale the probe. Unfreeze the encoder or change the pretraining source.

### 8. 100 slices OOM with encoder gradients
- **What happens:** Unfrozen ViT-B/16 at bs=1 × 100 slices blows past 16 GB on T4. bs=1 × 64 slices fits (~11 GB).
- **Rule:** Frozen probe can use all 100 slices. Unfrozen fine-tune: max 64 on T4 16GB.

### 9. Eval preprocessing must match pretraining
- **What happens:** Pretraining normalizes with ImageNet mean/std; downstream forgets to. Frozen probe AUC drops ~10 pts.
- **Rule:** `imagenet_normalize()` before the encoder in both frozen and unfrozen paths. Applied in `eval_downstream.py`.

### 10. Attentive probes overfit small medical datasets
- **What happens:** Even d=1 (7M params) + weight_decay=0.05 + dropout=0.2 hits train AUC → 1.0 by epoch 10-15 on 6K samples. Val AUC peaks at epoch 4-8 then drifts.
- **Why:** 7M params / 6K samples ≈ 1200 params/sample. Small-data attentive probes are known to over-parameterize — see [Attention, Please! ICLR 2026](https://arxiv.org/abs/2506.10178).
- **Rule:** Frozen-probe overfit pattern is normal and invisible in the paper (only best-val is reported). Ceiling is the encoder, not the probe.

### 11. Print buffering hides training progress under `tee`
- **What happens:** Python `print()` is block-buffered (~4KB) when piped. Per-epoch progress sits in the buffer for ~40 epochs.
- **Rule:** `sys.stdout.reconfigure(line_buffering=True)` at `__main__` (or `flush=True` on every print). `train_patch.py` does it via a `log()` helper; `eval_downstream.py` sets line buffering globally (commit `61f08c3`).

### 12. Linear scaling rule for LR with batch size
- **What happens:** Copy a literature LR verbatim without accounting for batch size. Literature's bs=1024 at LR=1e-3 ≠ our bs=256 at LR=1e-3.
- **Rule:** `LR_ours = LR_ref × (bs_ours / bs_ref)`. For bs=256 from a bs=1024 reference, LR=4e-4.

### 13. Torchrun port conflict with multiple DDP jobs on same node
- **What happens:** Two DDP jobs on the same compute both bind port 29500. Second crashes with `Address already in use`.
- **Rule:** Unique `MASTER_PORT` per job if co-scheduled, or run sequentially.

### 14. Shell scripts need LF line endings for bash on Linux
- **What happens:** Windows-edited scripts get CRLF. AML Linux bash chokes with `$'\r': command not found`.
- **Rule:** `.gitattributes` has `*.sh text eol=lf`, but the working copy can still drift. Run `sed -i 's/\r$//' scripts/*.sh` before submitting.

---

## Fine-tuning

### 15. Layer-wise LR decay (LLRD) is standard for ViT fine-tuning
- **Rule:** γ=0.65 for ViT-B, γ=0.75 for ViT-L, MAE convention. Top encoder layers get the full base LR; bottom layers get ~base × γ^num_layers. Our single flat `lr_encoder` was leaving 500× LR on the table for top encoder layers. Implemented in `build_finetune_param_groups` (commit `f78876f`).

### 16. Warmup gate also applies to fine-tune best-save and patience
- **Rule:** Same bug class as pretraining #2. `epoch > warmup_epochs` (fine-tune loop is 1-indexed, not 0-indexed like pretraining — commit `0dcd9d0` fixes the off-by-one).

---

## Code Differences from Official I-JEPA

Intentional, not bugs:

| Aspect | Official | Ours | Impact |
|---|---|---|---|
| Momentum schedule | Linear | Cosine | Negligible for our EMA ranges |
| Target path AMP | Under autocast | fp32 (no autocast) | Slightly more precise targets |
| LayerNorm epsilon | 1e-6 | 1e-5 (PyTorch default) | Minor |
| CLS token | Interpolation code present | No CLS, direct pos_embed add | Cleaner, avoids the no-CLS interpolation bug |

---

## General rules

1. **I-JEPA loss is NOT a reliable quality metric.** Low loss can mean collapse; high loss can mean healthy learning. Monitor `rep_diversity` and `cos_sim` instead.
2. **Downstream AUC is the quality signal.** Use the linear probe sweep across ep25/50/75/100 to pick the pretraining checkpoint.
3. **No early stopping for final pretraining runs.** Literature standard (RETFound, V-JEPA) is fixed-epoch.
4. **Upload checkpoints to blob during training.** Don't wait for the end — jobs crash.
5. **DDP cleanup (barrier + destroy) must run on all ranks before any rank-specific code.**
6. **Never revert local-only configs** (blob-storage accounts, compute names) to placeholders when committing.
