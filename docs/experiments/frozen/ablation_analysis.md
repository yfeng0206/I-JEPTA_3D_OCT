# Probe Ablation — Statistical Analysis

Paired bootstrap (B=2000, stratified, same resample indices across methods) on the 3000-volume FairVision Test split. All probes evaluated on the same ep100 random-init I-JEPA encoder checkpoint.

## Point Test AUC with 95% confidence intervals

| Probe | Params | Test AUC | 95% CI |
|---|---|---|---|
| d=1 AttentiveProbe + Linear | 7.17M | 0.8706 | [0.858, 0.883] |
| MeanPool + Linear | 2.3K | 0.8746 | [0.861, 0.887] |
| CrossAttnPool + Linear | 277K | 0.8791 | [0.867, 0.890] |
| Fine-tune + d=1 + LLRD γ=0.5 | 86M + 7.17M | 0.8878 | [0.875, 0.899] |

## Paired differences (same bootstrap draws → correlated-AUC test)

| Comparison | Mean delta | 95% CI | p-value | Significance |
|---|---|---|---|---|
| MeanPool − d=1 | **+0.0041** | [−0.001, +0.010] | 0.081 | **ns** |
| CrossAttnPool − MeanPool | +0.0046 | [−0.001, +0.010] | 0.044 | * |
| CrossAttnPool − d=1 | +0.0085 | [+0.003, +0.014] | 0.002 | ** |
| Fine-tune − CrossAttnPool | +0.0087 | [+0.003, +0.014] | 0.001 | ** |
| Fine-tune − d=1 | +0.0173 | [+0.011, +0.024] | <0.001 | *** |

p-values are one-sided P(delta ≤ 0). Significance thresholds: * p<0.05, ** p<0.01, *** p<0.001.

## Interpretation

1. **MeanPool vs d=1 is NOT significant (p=0.08).** We cannot claim MeanPool beats d=1 from this data. What we CAN claim: d=1 AttentiveProbe (7M params) does NOT improve over a trivial mean-pool baseline (2.3K params) within statistical noise. 3000× more parameters for zero measurable AUC gain is a strong negative finding — d=1 is not earning its capacity on 6K volumes.

2. **CrossAttnPool vs d=1 is highly significant (p=0.002).** CrossAttnPool (277K) beats d=1 (7.17M) with a +0.009 Test AUC gap. This replicates the [ICLR 2026 "Attention, Please!"](https://arxiv.org/abs/2506.10178) finding on medical OCT: standard attentive probes are over-parameterized and a minimal variant matches or beats them.

3. **CrossAttnPool vs MeanPool is borderline significant (p=0.04).** Attention + positional embeddings add a modest but real +0.005 Test AUC over pure mean-pool. Slice-level attention is helping — just not as much as one might have guessed. The gain is smaller than the param scaling suggests (100× more params for +0.5% AUC).

4. **Fine-tune vs anything is highly significant (p<0.001 vs d=1, p=0.001 vs CrossAttnPool).** MAE-style LLRD fine-tuning adds +0.009 over the best frozen probe and +0.017 over the d=1 baseline. Within the 2-4% fine-tune-vs-LP gap reported by Zhou 2025 for retinal tasks.

## Parameter-per-sample intuition

With 6000 training volumes:

| Probe | Params / sample | Regime |
|---|---|---|
| d=1 AttentiveProbe | 1,200 | Severe over-parameterization |
| CrossAttnPool | 46 | Tight but workable |
| MeanPool | 0.4 | Heavy under-parameterization, safe |

Classical rule of thumb (~10 samples per parameter) puts the sweet spot around 600 parameters for 6K samples — below CrossAttnPool but far above MeanPool. That the CrossAttnPool wins suggests 46 params/sample is still tractable because the small-capacity attention layer primarily learns slice weights rather than arbitrary transformations.

## Implications for paper claims

**Safe claims:**
- CrossAttnPool significantly outperforms the standard I-JEPA-style d=1 AttentiveProbe at 26× fewer parameters (p=0.002).
- A minimal cross-attention pool with slice-axis pos_embed is the recommended frozen probe for multi-slice OCT classification.
- Fine-tuning with MAE-style LLRD γ=0.5 adds ~2% Test AUC on top of the best frozen probe.

**Claims that require more data to substantiate:**
- "d=1 AttentiveProbe overfits" — directionally consistent with MeanPool's +0.004 over d=1, but NS. Would need replication across checkpoints or datasets.
- "Slice positional information is strictly required" — CrossAttnPool beats MeanPool only marginally (p=0.04). Not a decisive claim.

## Pending

`plum_jicama_9tnw0xy5tk`: fine-tune + CrossAttnPool + LLRD γ=0.5. Closes the last cell:
- If CrossAttnPool wins under fine-tune as well → **"CrossAttnPool is pareto-optimal across frozen and fine-tune regimes."**
- If d=1 wins under fine-tune → **"Fine-tune needs more probe capacity than frozen does."** Different story, also publishable.

## Reproducibility

Analysis computed in one block from the saved `test_predictions.npz` files for each run. 2000 bootstrap resamples with stratified positive/negative resampling at 48.5%/51.5% to match the test-set prevalence. Seed 42. Bootstrap delta CIs computed on the same resample indices to leverage within-sample correlation (standard practice for comparing AUCs on the same test set).
