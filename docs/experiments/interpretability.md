# Interpretability — Occlusion Attribution on the 3 Fine-Tune Probes

Architecture-agnostic occlusion attribution for the three fine-tune probes that tied at Test AUC ~0.887. Goal: validate that the probes actually use the same signal, and find what that signal is at the slice, patch, and pixel level.

Pipeline: [`scripts/interpretability.py`](../../scripts/interpretability.py). Results are derived from two AML jobs (initial attribution + patch aggregate over 3000 volumes) plus local post-hoc analyses.

## Method

```
OCT volume → (64 slices, 3, 256, 256)
 → frozen-ish encoder → 256 patch tokens/slice
 → mean over patches → (64, 768) slice tokens F
 → probe → (768,) pooled vector
 → LayerNorm + Linear → logit

Inverse (all architecture-agnostic, all causal):
  Slice-level:   for s in 0..63:  zero F[s], measure Δlogit
  Window-level:  for w in 0..57:  zero F[w..w+6], measure Δlogit  (W=7)
  Patch-level:   for p in 0..255: leave-one-out on patch tokens of target slice,
                                  substitute altered mean into F, measure Δlogit
```

Occlusion is used instead of gradient × input because the CrossAttnPool and d=1 probes are non-linear in F — gradient attribution gives misleading numbers there. Occlusion remains valid for all three probes.

## Key findings at a glance

| # | Claim | Figure / sheet |
|---|---|---|
| 1 | All 3 probes agree on **which slices** matter | [`slice_contribution_curves.png`](../../results/summary/slice_contribution_curves.png) |
| 2 | Wrong predictions use the **same pattern with weaker signal**, not different anatomy | [`slice_contribution_by_outcome.png`](../../results/summary/slice_contribution_by_outcome.png) |
| 3 | The pattern is statistically robust (tight bootstrap CI at n=1466) | [`slice_contribution_ci.png`](../../results/summary/slice_contribution_ci.png) |
| 4 | **Window occlusion (W=7) amplifies the signal ~7× and cleans it** | [`04_window_occlusion_W7.png`](../../results/summary/04_window_occlusion_W7.png) |
| 5 | Per-patch attribution concentrates on the B-scan center | [`05_patch_aggregate.png`](../../results/summary/05_patch_aggregate.png) |
| 6 | Individual B-scans show clinical landmarks (RNFL thinning, cup excavation) | [`heatmap_grid.png`](../../results/summary/heatmap_grid.png) |
| 7 | **Probes agree at slice granularity, not patch granularity** (r=0.94 → r=0.10) | [`09_cross_probe_patch_agreement.png`](../../results/summary/09_cross_probe_patch_agreement.png) |
| 8 | Window occlusion recovers 25× more signal than single-slice for MeanPool | [`10_completeness_window.png`](../../results/summary/10_completeness_window.png) |
| 9 | **14-65% of patches are statistically non-zero** (95% bootstrap CI) | [`11_patch_ci_significance.png`](../../results/summary/11_patch_ci_significance.png) |
| 10 | Attribution structure is nearly invariant to prediction confidence | [`13_attribution_vs_confidence.png`](../../results/summary/13_attribution_vs_confidence.png) |

## Cross-model agreement — full picture

| Pair | Slice-level r | Patch-level r (per-volume, 2 slices) | Interpretation |
|---|---|---|---|
| MeanPool vs CrossAttnPool | **0.94** | 0.08 / 0.10 | **Same slices, different patches** |
| MeanPool vs d=1 | 0.53 | 0.11 / 0.10 | d=1 is noisier throughout |
| CrossAttnPool vs d=1 | 0.59 | 0.09 / 0.10 | Same |

The probes robustly agree on WHICH slices are informative, but each picks a different patch subset within those slices. The disease signal is redundantly distributed across multiple patch groups within each informative slice — each probe learns its own subset.

## Completeness under occlusion

Median `|sum(contribs)| / |baseline_logit|` ratio:

| Model | Single-slice (W=1) | Window (W=7) | Amplification |
|---|---|---|---|
| MeanPool | **1.3%** | 32.4% | **25×** |
| CrossAttnPool | 6.0% | 52.0% | 8.6× |
| d=1 AttentiveProbe | 48.6% | **304.9%** | 6.3× |

- Under single-slice zero-mask, MeanPool contribs explain only 1.3% of the logit because removing 1 of 64 pool inputs barely moves the mean. Window W=7 recovers 25× more signal.
- d=1 under window occlusion **overshoots** (sum > 3× logit) — direct evidence that its self-attention nonlinearly amplifies large perturbations. The choice of occlusion primitive matters more for d=1 than for MeanPool.
- **Actionable**: window occlusion is the correct primitive for slice-level attribution on mean-pool models.

## Population-level two-peak structure — with caveat

Population-averaged slice attribution shows a bimodal structure: peak at native position ~63, dip at ~95, peak at ~137. A naive reading maps this to superior + inferior optic disc rim.

**This interpretation is NOT supported by per-volume data.** Correlation between per-volume contribs at the two peaks is **slightly negative**:

| Model | r (glaucoma class) |
|---|---|
| MeanPool | −0.22 |
| CrossAttnPool | −0.07 |
| d=1 | −0.14 |

If both peaks came from the same bilateral anatomic structure, per-volume contribs should positively correlate (a diseased eye has signal at both rims). Negative correlation suggests the peaks reflect **OD/OS laterality mixing**: right-eye and left-eye scans are stored with flipped axial orientation, so each contributes to a different peak. Population average shows both; individual volumes show one.

**Until OD/OS flipping is implemented** (detect disc laterality from the SLO, then reorient), the "bilateral disc rim" reading should not be claimed. See `12_disc_rim_symmetry.png`.

## Errors are weaker-signal, not wrong-anatomy

Stratifying by TP / FN / TN / FP: FN curves are scaled-down TP curves (same shape, smaller amplitude). Same for FP vs TN. The ~20% error rate at threshold 0.5 comes from signal-strength saturation on hard cases, not from the model attending to different slices. Also consistent with finding 10: attribution shape is confidence-invariant (|Pearson r| ≤ 0.25 between peak contrib magnitude and |logit|).

## Paper-ready claims

**Safe claims**:
- All 3 probes, despite 0 → 7.17M probe params, converge on the same slice-level attribution structure (MeanPool↔CrossAttnPool slice-level r = 0.94).
- A trivial MeanPool + Linear is Pareto-optimal for the fine-tune regime.
- Errors come from weaker-signal, not wrong-anatomy; the attribution pattern is confidence-invariant.
- Window occlusion (W=7) is the correct attribution primitive for mean-pool-based models; single-slice zero-mask systematically under-estimates signal.

**Claims to avoid without more evidence**:
- "The model discovers superior + inferior disc rim" — unsupported without OD/OS flipping.
- "The three probes look at the same pixels" — they don't (patch-level r ≈ 0.10).

## Reproducibility

All .npz outputs and per-slice contribution tables are on blob at `ijepa-interpretability/`. Local post-hoc analyses (bootstrap CI, window occlusion, deeper correlations) are regenerated by [`scripts/deeper_interpretability_analysis.py`](../../scripts/deeper_interpretability_analysis.py) reading from a local archive of the AML outputs.

Single-seed per FT run is a known limitation. Cross-architecture agreement (r = 0.94 MP↔CA at slice level) makes single-seed less of a concern but multi-seed would strengthen the paper.
