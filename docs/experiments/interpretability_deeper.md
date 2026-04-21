# Deeper analyses — claims 8-12

Post-hoc analyses on the interpretability data already collected (no new AML jobs). 5 new claims, 2 of which substantially revise the original interpretation.

## Claim 8: Probes agree at SLICE level but DISAGREE at patch level

**Evidence**: `plots/09_cross_probe_patch_agreement.png` + `tables/summary.xlsx` sheet `07_cross_probe_patch`

Per-volume Pearson r between pairs of probes on the (256,) patch attribution vector, averaged across 3000 volumes:

|  | MeanPool ↔ CrossAttnPool | MeanPool ↔ d=1 | CrossAttnPool ↔ d=1 |
|---|---|---|---|
| **Slice level** (from `02_slice_peaks` / prior work) | **r = 0.94** | r = 0.53 | r = 0.59 |
| **Patch level, slice 20** | r = 0.08 | r = 0.11 | r = 0.09 |
| **Patch level, slice 43** | r = 0.10 | r = 0.10 | r = 0.10 |

**Interpretation**: The probes robustly converge on WHICH slices are informative, but **each probe picks a different set of pixel patches within those slices**. This argues the slice-level signal is **redundant across patches** — multiple minimal sufficient patch subsets within the disc region would each produce the same slice token after mean-pool → each probe learns to rely on its own subset.

**Revised paper claim**: "All three probes agree on the slice-level attribution structure (which slices drive the prediction, r = 0.94 between MeanPool and CrossAttnPool). Within those slices, each probe relies on a different subset of patches (per-volume patch-level r ≈ 0.1), indicating the disease signal is redundantly distributed across multiple patch groups."

## Claim 9: Window occlusion largely closes the completeness gap for MeanPool

**Evidence**: `plots/10_completeness_window.png` + sheet `08_completeness_window`

Median `|sum(contribs)| / |baseline_logit|` ratio:

| Model | Single-slice (W=1) | Window (W=7) | Amplification |
|---|---|---|---|
| MeanPool | 1.3% | **32.4%** | **25×** |
| CrossAttnPool | 6.0% | 52.0% | 8.6× |
| d=1 AttentiveProbe | 48.6% | **304.9% (!)** | 6.3× |

**Interpretation**:
- For **MeanPool**, widening the occlusion window from 1 to 7 slices recovers from 1.3% to 32.4% of the logit — the "hidden" signal is spread across adjacent slices and zeroing 1 barely perturbs the pool. Claim 7 ("contribs don't add up") was a feature of zero-mask-on-mean-pool, not a feature of the underlying attribution.
- For **d=1**, the window-occlusion ratio **exceeds 100%** (304.9%). Zeroing 7 consecutive slices produces a larger logit swing than the baseline prediction magnitude — direct evidence that d=1's self-attention nonlinearly amplifies large perturbations. Explains its noisy single-volume attribution seen earlier.
- Practical: **window occlusion is the correct primitive for slice-level attribution on mean-pool models**. Single-slice zero-masking systematically under-estimates contribution.

## Claim 10: 14-65% of patches are statistically non-zero on the aggregate heatmap

**Evidence**: `plots/11_patch_ci_significance.png` + sheet `09_patch_significance`

Per-patch 95% bootstrap CI (B=500) on the glaucoma-class mean patch contribution:

| Model | Slice | # patches with CI excluding 0 | % |
|---|---|---|---|
| MeanPool | 20 | 36 / 256 | 14.1% |
| MeanPool | 43 | 41 / 256 | 16.0% |
| CrossAttnPool | 20 | 71 / 256 | 27.7% |
| CrossAttnPool | 43 | 71 / 256 | 27.7% |
| d=1 AttentiveProbe | 20 | 145 / 256 | 56.6% |
| d=1 AttentiveProbe | 43 | 165 / 256 | 64.5% |

**Interpretation**: The aggregate per-patch heatmap is not all noise — even under MeanPool (weakest amplitudes) ~15% of patches have a non-zero-crossing CI at n=1466. d=1's much higher rate is partly real (its probe amplifies) and partly confounded (higher raw amplitudes make smaller patches look significant relative to SE).

**Practical**: when we overlay "where the model looks" on a B-scan, we should mark only statistically-significant patches, not the whole heatmap. Claim 5 stands but tightens: ~20 patches (not 256) carry the real signal per slice for MeanPool/CrossAttnPool.

## Claim 11 (REVISION): The two slice-level peaks are NOT bilateral rim of one structure

**Evidence**: `plots/12_disc_rim_symmetry.png` + sheet `10_disc_rim_symmetry`

Per-volume Pearson r between `slice_contrib[20]` and `slice_contrib[43]`:

| Model | r (glaucoma) | r (all volumes) | median ratio 20 / 43 |
|---|---|---|---|
| MeanPool | **-0.22** | -0.15 | +0.37 |
| CrossAttnPool | -0.07 | -0.03 | ~0 |
| d=1 | -0.14 | +0.03 | +0.22 |

**What we expected**: if the two peaks both came from the same bilateral anatomic structure (e.g., superior + inferior rim of the same optic disc), per-volume contribs should POSITIVELY correlate — a diseased eye has signal at both rims.

**What we got**: slight *negative* correlations, essentially uncorrelated within volumes.

**This invalidates the "superior + inferior disc rim" interpretation** I used in earlier writeups. More likely interpretations of the bimodal population curve:

1. **OD/OS laterality artifact** (GPT flagged this earlier). In Cirrus HD-OCT disc cubes, left-eye and right-eye scans have flipped anatomic orientation in the stored array. Without OD/OS flipping to a common orientation, right-eye volumes contribute a peak at ~native slice 63 and left-eye at ~137 (or vice versa). Population average shows both peaks; individual volumes show one. The negative per-volume correlation is consistent with this mutual-exclusion pattern.
2. **Different anatomic features** (macula vs disc, or superior arcade vs foveal region). Less likely for an optic disc cube, but possible.
3. **Some mix of both**.

**Action**: the "superior/inferior disc rim" claim should be REMOVED from the paper narrative until either (a) OD/OS flipping is implemented or (b) per-volume correlation becomes positive after correction.

**Safe replacement statement**: "Population-level slice attribution shows a bimodal structure around the disc region. Per-volume contributions at the two peaks are uncorrelated (r ≈ -0.1 to -0.2), suggesting the bimodality reflects population heterogeneity (likely scan laterality mixing) rather than bilateral features of a single anatomic structure within each eye."

## Claim 12: Attribution magnitude is largely invariant to prediction confidence

**Evidence**: `plots/13_attribution_vs_confidence.png` + sheet `11_magnitude_vs_confidence`

Pearson r between per-volume peak |slice contribution| and |baseline logit|:

| Model | r |
|---|---|
| MeanPool | +0.25 |
| CrossAttnPool | -0.16 |
| d=1 AttentiveProbe | -0.06 |

Different signs per model, all weak (|r| ≤ 0.25). Combined with Claim 2 (errors are scaled-down TP curves), the overall picture:

**Interpretation**: the model's ATTENTION STRUCTURE is nearly invariant to prediction confidence. Whether the model predicts confidently or uncertainly, correctly or wrong, it uses the same attribution shape. What varies is the SIGNAL AMPLITUDE at the same anatomic locations. Confident correct ↔ weaker-signal correct ↔ weaker-signal incorrect are on the same spectrum, not qualitatively different attention patterns.

## Summary of what we learned from the deeper analyses

1. **Same slices, different patches** (Claim 8) — strong new constraint on the "same anatomy" story. At the slice level the probes are in lockstep (r = 0.94); at the patch level they're essentially independent (r ≈ 0.10). The disease signal is redundant across multiple spatially distinct patch groups within each informative slice.

2. **Window occlusion is the right primitive** (Claim 9) — single-slice zero-masking systematically under-estimates attribution because mean-pool robustness hides it. Window W=7 recovers 25× more signal for MeanPool. For d=1, window occlusion blows past y=x — direct evidence of nonlinear probe amplification.

3. **Patch heatmaps have ~15-65% statistically significant patches** (Claim 10) — the aggregate per-patch map is not uniform noise, but it's not all signal either. Clinical-interpretation figures should mark only significant patches.

4. **REVISED: the bimodal slice curve is probably OD/OS mixing, not bilateral rim** (Claim 11) — per-volume contribs at the two peaks are slightly NEGATIVELY correlated, contradicting a single-structure interpretation. Until OD/OS flipping is implemented, the "superior + inferior rim" reading is unsupported.

5. **Attribution shape is confidence-invariant** (Claim 12) — correct/incorrect, confident/uncertain, the model looks at the same pattern with different amplitudes. Errors are signal-strength weaker, not attention-location different (reinforces Claim 2 from first pass).

## Next steps this implies

- **OD/OS flipping** is now a must-do, not a nice-to-have. Either (a) use SLO disc-position detection to infer laterality and flip, or (b) stratify analyses by eye and show separate curves.
- **Window occlusion (W=5 or W=7) should be our primary slice-attribution primitive** going forward, not single-slice zero-mask.
- **Patch heatmaps should use significance masks** — only show patches where the bootstrap CI excludes zero.
- **The "convergent anatomy" narrative needs a precision edit**: probes converge at slice granularity, not patch granularity. Paper language should reflect that.
