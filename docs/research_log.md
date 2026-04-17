# Research Log

Reference log for paper writing. Each entry records a problem or decision, the investigation (what we read / consulted), the solution or decision, and outcomes. New entries appended at the top of each section; sections updated whenever we learn something worth keeping.

## Contents
- [Session Log](#session-log) — chronological problem/solution entries
- [Paper Bibliography](#paper-bibliography) — every paper referenced, with context
- [Backlog / Future Work](#backlog--future-work) — experiments planned but not yet run
- [Corrections](#corrections) — claims I got wrong and what the correct answer is

---

## Session Log

### 2026-04-17 — Literature-tuned frozen probe sweep

#### 1. Parallel 4-probe sweep hung silently after feature pre-compute
**Context**: Initial sweep `dreamy_basin_6rxm9myg2g` submitted with all 4 probes (ep25/50/75/100) running in parallel on 4 T4 GPUs via `CUDA_VISIBLE_DEVICES` isolation.
**Symptom**: After ~50 min of feature pre-compute all 4 probes finished caching features, then stdout went silent for 2h+. No results uploaded.
**Investigation**: Checked `std_log.txt` (last modified 2h earlier), `metrics-capability.log` (disk writes ~1.4 MB over 1.5h = essentially idle), blob storage (no `results.json` from any probe). Hypothesized RAM contention: 4 probes × 2 GB cached features + 4 DataLoader workers each forking with COW = 16 worker processes × 2 GB virtual memory.
**Solution**: Rewrote `scripts/run_linear_sweep.sh` to run probes sequentially on GPU 0. Removed background `&`, added feature_cache cleanup between probes. Commit `f02ccf1`.
**Tradeoff**: Wall time ~75 min → ~3.3h. Reliability > speed for this debugging phase.
**References**: Diagnosed from system logs; no external paper.

#### 2. Python `print()` block-buffered under `tee` → no live stdout
**Context**: Linear-sweep stdout showed pre-compute progress but nothing during actual probe training, giving appearance of a hang.
**Investigation**: Compared to pretraining logs which show per-epoch prints in real time. Found `src/train_patch.py:160` uses `print(msg, flush=True)` inside a `log()` helper. `src/eval_downstream.py` used plain `print()`. When stdout is piped through `tee`, Python defaults to block buffering (~4 KB).
**Solution**: Added `sys.stdout.reconfigure(line_buffering=True)` at `__main__` in `eval_downstream.py`. One-line fix, applies globally. Commit `61f08c3`.
**References**: Python docs on stdout buffering; diagnosed from trace.

#### 3. DDP val_loss could diverge across ranks → potential collective hang
**Context**: Code review after Opus 4.6 → 4.7 upgrade found `evaluate_val` in `train_patch.py` didn't aggregate val_loss across ranks.
**Investigation**: Each rank iterates its own `DistributedSampler` shard and computes a local mean. Early-stop decision on different values → ranks diverge at next collective → NCCL timeout after 30 min.
**Solution**: `all_reduce` sum + count, compute global mean. Same pattern applied to `run_patch_finetune` in `eval_downstream.py` for train_loss logging. Commit `135ba2a`.
**References**: Our own `lessons_learned.md` items #3 and #4 describe the historical NCCL timeout bugs; fix generalizes the pattern.

#### 4. `best` checkpoint saved pre-warmup with artificially low val_loss
**Context**: `run6_random_posfix.md` notes `jepa_patch-best.pth.tar` ended up being epoch 1. EMA target not yet diverged → trivial prediction task → val_loss low but not meaningful.
**Investigation**: Existing `past_warmup` guard only protected patience counting, not best-checkpoint writing.
**Solution**: Gated the entire best-ckpt saving block (including the patience decrement) on `past_warmup`. Commit `135ba2a`.
**References**: Our `lessons_learned.md` item #2 describes the warmup-epoch artificially-low loss. This fix extends that rule to checkpoint saving.

#### 5. d=3 attentive probe overfit catastrophically
**Context**: First sweep (`patient_grape_3v757g11t5`) ran probe with `PROBE_DEPTH=3` (21M trainable params), `WEIGHT_DECAY=0`, `dropout=0.1`. First probe (ep25) results: Train AUC 0.748 → 1.000 by epoch 20, Val AUC peaked 0.8437 at epoch 4 then drifted down, Val loss 0.49 → 1.70.
**Investigation**:
- Reviewed literature on attentive probes: [V-JEPA](https://arxiv.org/html/2404.08471v1) uses 4-block probe (larger than ours); [Attention, Please! ICLR 2026](https://arxiv.org/abs/2506.10178) states attentive probes are "over-parameterized and inefficient" as a known problem; I-JEPA original paper uses single-block probe.
- Re-read our own `lessons_learned.md` item #6: "The frozen encoder's representations are the limiting factor. More probe layers can't extract signal that isn't in the features."
**Solution**: Retuned to literature-standard config: `PROBE_DEPTH=1`, `WEIGHT_DECAY=0.05`, `dropout=0.2`, `EPOCHS=50`, `PATIENCE=15`, `BATCH_SIZE=256`, `LR_PROBE=LR_HEAD=4e-4` (linear-scaled from 1e-3 reference at bs=1024), `warmup_epochs=5` (10% of epochs per MAE convention). Commits `e1eb9e5` (script parameterization) + config updates.
**Outcome**: ep25 Val AUC improved marginally 0.8437 → 0.8460. Overfit delayed (best epoch 4 → 8) but not eliminated. Confirms probe capacity isn't the ceiling; encoder quality is.
**References**:
- [V-JEPA](https://arxiv.org/html/2404.08471v1) — 4-layer attentive probe with cross-attention final layer
- [Attention, Please! (ICLR 2026)](https://arxiv.org/abs/2506.10178) — attentive probe over-parameterization as a field-wide problem
- [Why Warmup the Learning Rate? (NeurIPS 2024)](https://arxiv.org/abs/2406.09405) — 10% warmup convention
- [On the Adequacy of Untuned Warmup for Adaptive Optimization](https://arxiv.org/pdf/1910.04209) — AdamW variance-stabilization rationale
- Our `lessons_learned.md` #6 — encoder features, not probe, is the bottleneck

#### 6. LR scaling: should literature 1e-3 apply verbatim?
**Context**: Initially proposed `LR=1e-3` to "match literature." User caught that literature's LR assumes their batch size, not ours.
**Investigation**: Linear scaling rule: `LR_ours = LR_ref × (bs_ours / bs_ref)`. Literature reference for attentive probes (I-JEPA, V-JEPA) is ~bs=1024 with LR~1e-3.
**Solution**: Scale LR to our batch size. Chose `bs=256`, `LR=4e-4` (linearly scaled). Kept single LR for probe and head (literature convention; head is only 2K params so Adam's adaptive scaling handles it).
**References**: Linear scaling rule is standard — Goyal et al. "Accurate, Large Minibatch SGD" (2017) is the canonical reference for SGD; similar principle applies (approximately) to AdamW.

#### 7. Literature-comparison check: is our 0.846 frozen-probe AUC reportable?
**Context**: Fear that overfit pattern (train AUC 1.0, val 0.85) would be flagged by reviewers.
**Investigation**:
- Papers don't publish per-epoch probe training curves; they report final val/test AUC at best-val-epoch.
- [Zhou 2025](https://arxiv.org/abs/2509.03421v1) Fig 5: frozen linear probes on retinal tasks average 0.761-0.787 across 10 tasks; our 0.846 on glaucoma is in the upper range.
- [RETFound on PAPILA (fundus)](https://www.nature.com/articles/s41586-023-06555-x): frozen probe AUC 0.86 — comparable to ours on OCT.
- RETFound fine-tuned on OCT glaucoma (2K scans, 50 epochs): AUC 0.91 → ~5% gain from fine-tune is literature-typical.
**Conclusion**: 0.846 frozen is publishable and competitive. Fine-tune is the lever for 0.88-0.91 territory.
**References**:
- [Zhou et al. 2025 (Generalist vs Specialist VFMs for Ocular Disease)](https://arxiv.org/abs/2509.03421v1)
- [RETFound (Zhou et al. Nature 2023)](https://www.nature.com/articles/s41586-023-06555-x)
- [RETFound-OCT glaucoma evaluation](https://www.medrxiv.org/content/10.1101/2024.08.04.24311475v1.full)

#### 8. DINOv2/US-JEPA "standard" citation error (correction)
**Context**: I claimed "DINOv2 and US-JEPA use d=1 attentive probe as standard."
**Investigation**: Fetched papers.
- DINOv2 (Oquab et al.): uses **linear classifier** on frozen features, no attentive probe.
- US-JEPA: uses **linear probing**, treats ultrasound frames individually (no temporal aggregation).
- V-JEPA IS the correct JEPA-family precedent: cross-attention + learnable query + 2-layer MLP + LN + linear.
**Correction**: Only I-JEPA and V-JEPA use attentive probes in this family. Cited in a paper, need to say "following V-JEPA's attentive probing protocol" not "following DINOv2."
**References**: [V-JEPA](https://arxiv.org/html/2404.08471v1), [DINOv2](https://arxiv.org/html/2304.07193v2), [US-JEPA](https://arxiv.org/html/2602.19322).

#### 9. Zhou 2025 paper — "Generalist vs Specialist" — summary of findings useful for framing
**Key results** (fine-tune AUROC averaged over 10 ocular + systemic tasks):
- RETFound-DINOv2 = 0.830 (winner)
- DINOv3-ViT-L = 0.816
- RETFound-MAE = 0.809
- DINOv2-ViT-giant = 0.800
**Inferences we can use (saves running experiments)**:
- DINO-family > MAE-family for retinal (RETFound-DINOv2 > RETFound-MAE by 0.021 AUC)
- DINOv3 > DINOv2 at equal backbone (train recipe > raw scale: DINOv3-L beats DINOv2-giant despite 3.5× smaller)
- Domain-specific continued pretraining still helps (RETFound-DINOv2 > generalist DINO)
- Fine-tune beats linear probe by 2-5% across the board
- RETFound-DINOv2 features most discriminative (mean pairwise sim 0.798 vs 0.975 for RETFound-MAE)
**Reference**: [Zhou et al. 2025 arxiv 2509.03421](https://arxiv.org/abs/2509.03421v1).

#### 10. Data protocol verification
**Context**: Paranoid check that pretraining doesn't leak into downstream Test split.
**Investigation**: `src/train_patch.py:244,254` uses only `Training/` and `Validation/` directories; Test/ never referenced. Downstream eval (`src/eval_downstream.py`) uses Training (probe train) + Validation (checkpoint select) + Test (final report). `src/datasets/oct_slices.py` confirms `glaucoma` is the label key.
**Confirmed clean protocol**:
- Pretraining: Training split slices (unlabeled SSL) + Validation for val-loss
- Probe train: Training split + glaucoma labels
- Val: Validation split (checkpoint selection via best val AUC)
- Test: Held out from everything, used only for final AUC reporting
**Reference**: No external paper; verified from our code.

#### 11. DINOv3 availability: same patch size as ours
**Context**: Checking feasibility of DINOv3-init for Phase 3 plan.
**Investigation**: Searched HF + DINOv3 model card. DINOv3 moved from patch=14 (DINOv2) → patch=16, directly matching our ViT-B/16.
- `facebook/dinov3-vitb16-pretrain-lvd1689m` available on HF (license click-through).
- 86M params, embed_dim=768, 12 heads, 12 depth — identical to our I-JEPA ViT-B.
- Architectural deltas: DINOv3 uses RoPE (rotary pos) + 4 register tokens. Our code uses sinusoidal pos, no registers.
**Implication**: Weight loading requires handling 2 deltas (skip register tokens; RoPE not loaded as parameter so no conflict). Transferring core transformer blocks is clean.
**Reference**: [DINOv3 paper arxiv 2508.10104](https://arxiv.org/html/2508.10104v1), [DINOv3-ViT-B/16 on HF](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m).

#### 12. RETFound-DINOv2 avoided architectural adaptation
**Context**: Asked how RETFound-DINOv2 handled mixing pretrained init + continued SSL.
**Investigation**: Fetched HF model card + inferred from published papers.
**Finding**: They kept DINOv2's architecture (ViT-L/14) AND DINOv2's SSL objective (self-distillation) — only the pretraining data changed (AlzEye retinal instead of LVD-142M natural). Standard "continued pretraining" recipe. No architectural gymnastics.
**Implication for our plan**: Mixing DINOv3-init with I-JEPA objective (what we proposed) is a more novel combination — no precedent for this specific mix. Three paths: (A) match RETFound exactly (use DINOv3's recipe on OCT, drop I-JEPA), (B) use DINOv2 init which maps cleaner to our I-JEPA arch, (C) adapt our code for DINOv3's RoPE + registers.
**Reference**: [RETFound-DINOv2 HF](https://huggingface.co/YukunZhou/RETFound_dinov2_meh), [Chia et al. NEJM AI 2024](https://ai.nejm.org/doi/10.1056/AIe2401024).

#### 13. SLIViT + Kermany cross-dataset evaluation plan
**Context**: SLIViT achieved ~86% fine-tune AUC on our FairVision. Hypothesis: their encoder is data-distribution-shifted.
**Investigation**:
- SLIViT: ConvNeXt + ViT integrator, pretrained on natural + 2D medical images including Kermany OCT. Published Nature BME.
- Kermany OCT dataset: 84,484 B-scans (2D slices, not 3D volumes), 4 classes (CNV/DME/DRUSEN/NORMAL), public on Mendeley Data CC-BY-4.0.
**Decision**: Add Kermany as a downstream benchmark (not pretraining source). Gives direct SLIViT comparison on their home turf. Cheap — just 4-class probe head, features already cacheable. Disease set differs from glaucoma (no glaucoma in Kermany).
**References**: [SLIViT paper pubmed 38045283](https://pubmed.ncbi.nlm.nih.gov/38045283/), [Kermany on Mendeley](https://data.mendeley.com/datasets/rscbjbr9sj/3).

#### 14. Pretraining data scope — narrow (glaucoma-only) vs broad (all FairVision 30K)
**Context**: Question raised about whether pretraining should use all 30K FairVision subjects (3 diseases) or just glaucoma 10K subset.
**Investigation**: SSL scales with data volume and diversity. RETFound used 1.6M mixed retinal images. DINOv2 used 142M curated diverse images. Broader pretraining data is established as better.
**Decision pending**: Keep downstream task binary glaucoma (literature comparability). For pretraining, consider expanding to full 30K in Phase 2+. Estimated cost ~3× compute; expected gain ~0.01-0.03 Val AUC for glaucoma probe. DINO-init might be larger-impact lever (0.03-0.05 expected).
**References**: [RETFound Nature 2023](https://www.nature.com/articles/s41586-023-06555-x), [DINOv2](https://arxiv.org/html/2304.07193v2), [FairVision paper](https://arxiv.org/abs/2310.02492).

---

## Paper Bibliography

### SSL Methods
- **I-JEPA** — Assran et al., CVPR 2023. [arxiv 2301.08243](https://arxiv.org/abs/2301.08243). Our pretraining method. ViT-H/14 → 73.3% ImageNet with attentive probe. Single-block attentive probe for evaluation.
- **V-JEPA** — Bardes et al., 2024. [arxiv 2404.08471](https://arxiv.org/html/2404.08471v1). Video JEPA. Uses 4-block attentive probe, last layer cross-attention with learnable query. Direct precedent for multi-frame/slice aggregation.
- **V-JEPA 2** — 2025. [arxiv 2506.09985](https://arxiv.org/abs/2506.09985). Successor to V-JEPA.
- **DINOv2** — Oquab et al., 2023. [arxiv 2304.07193](https://arxiv.org/html/2304.07193v2). Self-distillation + iBOT. Linear probe evaluation (not attentive). 142M curated image pretraining.
- **DINOv3** — Meta, 2025. [arxiv 2508.10104](https://arxiv.org/html/2508.10104v1). 1.7B images, Gram Anchoring, high-res adaptation. ViT-B/16 matches our arch. HF: `facebook/dinov3-vitb16-pretrain-lvd1689m`.
- **MAE** — He et al., CVPR 2022. [arxiv 2111.06377](https://arxiv.org/abs/2111.06377). Masked pixel reconstruction. Zhou 2025 shows MAE-family < DINO-family for retinal continued pretraining.

### Medical Foundation Models
- **RETFound (Nature 2023)** — Zhou et al. [Nature paper](https://www.nature.com/articles/s41586-023-06555-x). 1.6M retinal images, MAE pretraining, ViT-L/16. Key glaucoma numbers: fine-tuned OCT AUC ~0.91, PAPILA fundus frozen probe ~0.86.
- **RETFound-DINOv2** — Continued pretraining of DINOv2 on 904K AlzEye retinal images. Winner in Zhou 2025 eval across 10 retinal tasks (avg AUROC 0.830). HF: `YukunZhou/RETFound_dinov2_meh`.
- **Zhou et al. 2025 (Generalist vs Specialist VFMs)** — [arxiv 2509.03421](https://arxiv.org/abs/2509.03421v1). Key findings: DINOv3-L > DINOv2-giant despite 3.5× smaller; RETFound-DINOv2 > all generalists; fine-tune > linear probe by 2-5%.
- **SLIViT** — Nature BME. [pubmed 38045283](https://pubmed.ncbi.nlm.nih.gov/38045283/). ConvNeXt + ViT integrator. Pretrained on natural + 2D medical (incl. Kermany). [GitHub](https://github.com/cozygene/SLIViT).
- **Chia et al. NEJM AI 2024** — [DOI 10.1056/AIe2401024](https://ai.nejm.org/doi/10.1056/AIe2401024). Multimodal RETFound extension paper.

### Datasets
- **FairVision** — Luo et al. 2024. [arxiv 2310.02492](https://arxiv.org/abs/2310.02492). 30K subjects across glaucoma/AMD/DR, 10K per disease. Our pretraining + eval source (glaucoma subset).
- **Kermany OCT 2017** — Kermany et al. Cell 2018. [Mendeley rscbjbr9sj/3](https://data.mendeley.com/datasets/rscbjbr9sj/3). 84,484 B-scans, 4 classes (CNV/DME/DRUSEN/NORMAL), CC-BY-4.0. Candidate cross-dataset eval benchmark.

### Probe Design / Evaluation
- **Attention, Please! Revisiting Attentive Probing for MIM** — ICLR 2026. [arxiv 2506.10178](https://arxiv.org/abs/2506.10178). Documents that attentive probes are widely over-parameterized. Proposes "Efficient Probing" (EP): multi-query cross-attention, ~200K params vs typical 7M+.
- **Linear Classifier Probes** — Alain & Bengio, 2016. [arxiv 1610.01644](https://arxiv.org/pdf/1610.01644). Foundational probe methodology paper.

### Optimization / Training
- **Why Warmup the Learning Rate?** — NeurIPS 2024. [arxiv 2406.09405](https://arxiv.org/html/2406.09405v1). Mechanisms of warmup in adaptive optimizers; supports 10% warmup convention.
- **On the Adequacy of Untuned Warmup for Adaptive Optimization** — 2019. [arxiv 1910.04209](https://arxiv.org/pdf/1910.04209). AdamW variance-estimate stabilization rationale for warmup.

---

## Backlog / Future Work

### Phase 2: Foundation-model baselines on FairVision OCT (cheap, ~hours each)
- [ ] DINOv3-ViT-B/16 frozen probe on FairVision glaucoma (HF weights, license click-through)
- [ ] RETFound-DINOv2 frozen probe (ViT-L/14, need dim/embed adaption for 1024→768 or use L-size probe)
- [ ] RETFound-MAE frozen probe (optional — Zhou 2025 shows it underperforms RETFound-DINOv2)
- [ ] SLIViT encoder probe (already have data point: ~86% fine-tune; frozen number TBD)

### Phase 2.5: Cross-dataset evaluation (new benchmark, low effort)
- [ ] Download Kermany OCT 2017 from Mendeley
- [ ] Add 4-class probe head (CNV/DME/DRUSEN/NORMAL)
- [ ] Evaluate our I-JEPA encoder on Kermany — cross-dataset generalization test
- [ ] Compare with SLIViT's reported Kermany numbers

### Phase 3: DINO-init + I-JEPA continuation (expensive, days)
- [ ] Decide init source:
  - DINOv2 ViT-L/14: cleaner arch match (no RoPE/registers), needs ViT-B→L upgrade and patch-embed retrain
  - DINOv3 ViT-B/16: patch size matches, but has RoPE + 4 register tokens → I-JEPA code changes needed
- [ ] Implementation path (A/B/C from session entry #12)
- [ ] Run continuation SSL on FairVision Training
- [ ] Compare vs RETFound-DINOv2 (same-init, different continuation SSL test)

### Phase 4: 3D-aware SSL (novel contribution — independent of init)
- [ ] Multi-view / axial-based 3D OCT pretraining
- [ ] Volume-level masking instead of per-slice masking
- [ ] Comparison vs current per-slice 2D approach

### Phase 5 (optional): Broader pretraining data
- [ ] Download full FairVision 30K (AMD + DR + Glaucoma)
- [ ] Pretrain on combined dataset, probe on glaucoma (tests data-scope hypothesis)
- [ ] 3 × binary probes (glaucoma/AMD/DR) on same encoder for multi-task eval

### Immediate / this session
- [x] Commit DDP + flush fixes (`135ba2a`, `61f08c3`)
- [x] Sequential sweep rewrite (`f02ccf1`)
- [x] Literature-tuned d=1 config + script parameterization (`e1eb9e5` + config)
- [ ] `busy_roof_xjmvcyb7pm` sweep completion
- [ ] Download sweep CSVs, pick best val-AUC epoch across ep25/50/75/100
- [ ] Fine-tune on best checkpoint (attentive probe d=1 + linear head, unfrozen encoder)
- [ ] Mean-pool + linear head ablation (no attention) — tests whether attentive probe earns its keep
- [ ] Generate sweep plots, update docs, commit

### Open questions for future experiments
- [ ] Does mean-pool + linear match attentive probe Val AUC? (if yes → drop attention)
- [ ] Does fine-tune break the ~0.85 frozen-probe ceiling to 0.89+? (literature suggests yes)
- [ ] Does full 30K multi-disease pretraining help glaucoma probe? (expected +0.01-0.03)
- [ ] Does DINO-init (v2 or v3) + I-JEPA continuation beat random-init I-JEPA? (expected +0.03-0.05)
- [ ] Does 3D volume-level SSL beat per-slice approach? (open research question)

---

## Corrections

Claims I made that turned out to be wrong, and the correct answer. Keeping these visible so they don't sneak back into the paper.

- **"DINOv2 standard uses attentive probe"** → WRONG. DINOv2 paper uses linear classifier on frozen features. Only I-JEPA and V-JEPA (and some successors) use attentive probes.
- **"US-JEPA uses d=1 attentive probe"** → WRONG. US-JEPA uses linear probing; treats ultrasound frames individually, no temporal aggregation in eval.
- **"Literature LR for probes is 1e-3"** (without batch-size scaling) → Incomplete. Literature LR assumes bs~1024. Linear scaling rule: `LR_ours = LR_ref × (bs_ours / bs_ref)`. We use bs=256 → LR=4e-4.
- **Initial time estimate of "50 min" for parallel sweep** → WRONG. Pre-compute alone is ~50 min per probe; didn't account for data download + checkpoint pull + training. Actual: ~75 min per probe sequential, ~3.3h for 4-probe sweep.

---

*Log maintained by us during development. Append new problems/solutions/references here rather than to chat — this is the durable source when we sit down to write the paper.*
