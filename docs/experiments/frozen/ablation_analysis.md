# Probe Ablation — Statistical Analysis (full 2×3 matrix)

Paired bootstrap (B=2000, stratified, same resample indices across methods) on the 3000-volume FairVision Test split. Six runs total: 3 probes × {frozen, fine-tune}. All use the same ep100 random-init I-JEPA ViT-B/16 encoder checkpoint.

## Point Test AUC

| Run | Probe | Params | Test AUC |
|---|---|---|---|
| Frozen | AttentiveProbe d=1 + Linear | 7.17M | 0.8706 |
| Frozen | MeanPool + Linear | 2.3K | 0.8746 |
| Frozen | CrossAttnPool + Linear | 277K | 0.8791 |
| Fine-tune LLRD γ=0.5 | AttentiveProbe d=1 + Linear | 86M + 7.17M | **0.8878** |
| Fine-tune LLRD γ=0.5 | CrossAttnPool + Linear | 86M + 277K | **0.8872** |
| Fine-tune LLRD γ=0.5 | MeanPool + Linear | 86M + 2.3K | **0.8868** |

## Paired comparisons — frozen regime

| Comparison | Δ Test AUC | 95% CI | two-sided p | |
|---|---|---|---|---|
| CrossAttnPool − d=1 | **+0.0085** | [+0.003, +0.014] | 0.004 | ** |
| CrossAttnPool − MeanPool | +0.0046 | [−0.001, +0.010] | 0.088 | ns |
| MeanPool − d=1 | +0.0041 | [−0.001, +0.010] | 0.163 | ns |

Frozen takeaway: **CrossAttnPool significantly beats d=1** at 26× fewer params. MeanPool is within noise of both d=1 and CrossAttnPool — the "CrossAttnPool > MeanPool" edge is directional but not significant under a two-sided test.

## Paired comparisons — fine-tune regime

| Comparison | Δ Test AUC | 95% CI | two-sided p | |
|---|---|---|---|---|
| FT d=1 − FT CrossAttnPool | +0.0005 | [−0.004, +0.005] | 0.81 | ns |
| FT d=1 − FT MeanPool | +0.0009 | [−0.004, +0.005] | 0.69 | ns |
| FT CrossAttnPool − FT MeanPool | +0.0004 | [−0.001, +0.002] | 0.63 | ns |

Fine-tune takeaway: **the probe architecture is noise.** All three fine-tune runs tie within 0.001 AUC of each other. No pair is statistically distinguishable. MeanPool (0 probe params) matches d=1 (7.17M probe params).

## Fine-tune uplift over frozen (same probe)

| Probe | Frozen | Fine-tune | Δ | two-sided p | |
|---|---|---|---|---|---|
| d=1 AttentiveProbe | 0.8706 | 0.8878 | **+0.0172** | <0.001 | *** |
| MeanPool | 0.8746 | 0.8868 | **+0.0122** | <0.001 | *** |
| CrossAttnPool | 0.8791 | 0.8872 | **+0.0080** | 0.009 | ** |

Fine-tune uplift is statistically real on every probe, but scales inversely with how well the probe did frozen. d=1 had the most ceiling to recover (over-parameterized frozen); CrossAttnPool was already near ceiling frozen.

## Cross-regime — does fine-tune beat the best frozen probe?

| Comparison | Δ Test AUC | 95% CI | two-sided p | |
|---|---|---|---|---|
| FT MeanPool − frozen CrossAttnPool | +0.0077 | [+0.002, +0.013] | 0.013 | * |
| FT d=1 − frozen CrossAttnPool | +0.0087 | [+0.003, +0.014] | 0.004 | ** |
| FT CrossAttnPool − frozen CrossAttnPool | +0.0080 | [+0.002, +0.014] | 0.009 | ** |

Even fine-tune + MeanPool (zero probe parameters) beats the best frozen probe significantly. The fine-tune uplift is not about the probe — it's about encoder adaptation.

## Interpretation — the two-regime story

1. **MeanPool vs d=1 under frozen is NOT significant** (two-sided p=0.16). We cannot claim MeanPool beats d=1. What we CAN claim: 3000× more probe params in d=1 buy no measurable AUC gain on this task. d=1 is not earning its capacity on 6K volumes.

2. **CrossAttnPool is significantly better than d=1 frozen** (+0.009, p=0.004). This replicates the [ICLR 2026 "Attention, Please!"](https://arxiv.org/abs/2506.10178) finding on medical OCT: standard attentive probes are over-parameterized for frozen-probe protocols.

3. **Under fine-tune, probe architecture is irrelevant.** All three fine-tune variants land within 0.001 of each other (pairwise p > 0.6). Whatever slice-weighting the attention probes provide frozen is absorbed by encoder top-block + encoder.norm adaptation under fine-tune.

4. **Fine-tune uplift is real for every probe, dominated by encoder-side adaptation.** MeanPool's +0.012 uplift with zero probe parameters proves the gain isn't probe-driven. The encoder.norm LayerNorm (at full base LR 2e-4) plus slight top-block adaptation (LR ~1e-4 after warmup) is what closes the gap.

5. **The fine-tune-vs-LP gap is ~1-2 pp on FairVision OCT glaucoma**, below Zhou 2025's 2-4% range for retinal accuracy. Our frozen-probe ceiling is closer to the fine-tune ceiling than Zhou 2025 would predict, consistent with the encoder already being near-saturated at ep100 SSL.

## Parameter-per-sample intuition (frozen regime only)

With 6000 training volumes:

| Probe | Params / sample | Regime |
|---|---|---|
| d=1 AttentiveProbe | 1,200 | Severe over-parameterization |
| CrossAttnPool | 46 | Tight but workable |
| MeanPool | 0.4 | Heavy under-parameterization, safe |

Under fine-tune the 86M encoder dominates and this ratio becomes ill-defined — the denominator "unique trainable capacity per sample" includes encoder top-block updates, which operate at very low LR. Effectively the encoder is a learned-slowly-adapted feature extractor, not a 86M-capacity learner.

## Paper claims (safe)

- **CrossAttnPool significantly outperforms the standard I-JEPA-style d=1 AttentiveProbe** at 26× fewer parameters under frozen-probe evaluation (+0.009 AUC, p=0.004).
- **Under fine-tune, the probe architecture is irrelevant.** A zero-probe-parameter mean-pool matches a 7.17M-parameter attentive probe on Test AUC (pairwise p > 0.6). Fine-tune uplift is entirely encoder-side.
- **For frozen-probe protocols**, CrossAttnPool is Pareto-optimal. **For fine-tune protocols**, MeanPool is Pareto-optimal (zero probe params).
- **Fine-tuning with MAE-style LLRD γ=0.5 adds +0.008-0.017 Test AUC** over the corresponding frozen probe, consistent with (but below) Zhou 2025's 2-4% retinal fine-tune-vs-LP gap.

## Paper claims (requires more data)

- **"d=1 overfits"** — directionally consistent (MeanPool's +0.004 over d=1, CrossAttnPool's +0.009 over d=1) but only the CrossAttnPool gap is individually significant. Would benefit from multi-seed replication.
- **"Slice positional information is strictly required"** — CrossAttnPool beats MeanPool under frozen only marginally (p=0.09 two-sided). Under fine-tune they tie (pos_embed-equipped CrossAttnPool vs position-blind MeanPool both at ~0.887). Not a decisive claim.

## Reproducibility

Analysis computed from `test_predictions.npz` saved by each of 6 AML runs. 2000 bootstrap resamples with stratified positive/negative resampling matching the test-set 48.5%/51.5% prevalence. Seed 42. Paired deltas computed on the same resample indices across methods (standard correlated-AUC test). All p-values reported two-sided unless noted otherwise.
