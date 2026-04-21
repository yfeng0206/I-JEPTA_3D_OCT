"""Deeper post-hoc analyses on existing interpretability data.

Produces:
  plots/
    09_cross_probe_patch_agreement.png     A — per-volume patch-level r between probe pairs
    10_completeness_window.png             H — sum(window_contrib) vs logit (completeness recovery)
    11_patch_ci_significance.png           I — per-patch 95% bootstrap CI + significance mask
    12_disc_rim_symmetry.png               G — peak_20 vs peak_43 per volume (symmetry test)
    13_attribution_vs_confidence.png       C — peak contrib magnitude vs |baseline logit|
  tables/summary.xlsx  new sheets:
    07_cross_probe_patch      pairwise r per (model_a, model_b, slice)
    08_completeness_window    window-occlusion completeness
    09_patch_significance     # significant patches per (model, slice)
    10_disc_rim_symmetry      per-volume peak correlation + mean ratio
    11_magnitude_vs_confidence Pearson r per model
  FINDINGS_DEEP.md            narrative of claims 8-12

Run: python _deeper_analysis.py
"""
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = r'C:\Users\garyfeng\OneDrive - Microsoft\Desktop\I-JEPA_results_presentation'
ARCHIVE = r'C:\Users\garyfeng\OneDrive - Microsoft\Desktop\I-JEPA_archive'
PLOTS   = os.path.join(RESULTS, 'plots')
TABLES  = os.path.join(RESULTS, 'tables')

names    = ['meanpool', 'crossattn', 'd1']
displays = {'meanpool': 'FT + MeanPool',
            'crossattn': 'FT + CrossAttnPool',
            'd1': 'FT + AttentiveProbe d=1'}
colors   = {'meanpool': '#d62728', 'crossattn': '#1f77b4', 'd1': '#2ca02c'}
axial    = np.linspace(0, 199, 64)

# ---- Load everything once ----
slice_contrib = {}
features      = {}
window_contrib = {}
patch_aggregate = {}
for n in names:
    slice_contrib[n] = np.load(os.path.join(ARCHIVE, '04_interpretability',
                                            'slice_contributions', f'slice_contrib_{n}.npz'))
    features[n] = np.load(os.path.join(ARCHIVE, '04_interpretability',
                                       'features', f'features_{n}.npz'))
    window_contrib[n] = np.load(os.path.join(ARCHIVE, '04_interpretability',
                                             'window_occlusion', f'window_contrib_{n}_W7.npz'))
    patch_aggregate[n] = {}
    for s in [20, 43]:
        patch_aggregate[n][s] = np.load(os.path.join(
            RESULTS, 'patch_aggregate', f'patch_aggregate_{n}_slice{s:02d}.npz'))

# =============================================================================
# A — Cross-probe PATCH-level correlation (per-volume, averaged across volumes)
# =============================================================================
print('\n[A] cross-probe patch-level agreement')
rows_A = []
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, slice_s in zip(axes, [20, 43]):
    mat = np.ones((3, 3))
    for i, na in enumerate(names):
        for j, nb in enumerate(names):
            if j < i:
                continue
            if i == j:
                continue
            pa = patch_aggregate[na][slice_s]['patch_contrib']   # (3000, 256)
            pb = patch_aggregate[nb][slice_s]['patch_contrib']
            # Per-volume Pearson correlation between the (256,) patch vectors
            rs = []
            for v in range(pa.shape[0]):
                a = pa[v]; b = pb[v]
                if a.std() < 1e-10 or b.std() < 1e-10:
                    continue
                rs.append(np.corrcoef(a, b)[0, 1])
            r_mean = float(np.mean(rs))
            mat[i, j] = mat[j, i] = r_mean
            rows_A.append({
                'model_a': displays[na],
                'model_b': displays[nb],
                'target_slice_subset': slice_s,
                'target_slice_native': int(axial[slice_s]),
                'mean_per_volume_pearson_r': round(r_mean, 4),
                'n_volumes_used': len(rs),
            })
            print(f'  slice {slice_s:>2} ({na:10s} vs {nb:10s}) : mean per-vol r = {r_mean:+.3f}')

    im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{mat[i, j]:+.2f}', ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='white' if abs(mat[i, j]) > 0.5 else 'black')
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels([displays[n] for n in names], fontsize=9, rotation=15)
    ax.set_yticklabels([displays[n] for n in names], fontsize=9)
    ax.set_title(f'subset slice {slice_s} (native ~{int(axial[slice_s])}/199)',
                 fontsize=10)
fig.suptitle('Cross-probe PATCH-level agreement\n'
             '(mean per-volume Pearson r between probes on (256,) patch vectors)',
             fontsize=12)
plt.colorbar(im, ax=axes, fraction=0.03)
fig.savefig(os.path.join(PLOTS, '09_cross_probe_patch_agreement.png'),
            dpi=140, bbox_inches='tight')
plt.close(fig)

# =============================================================================
# H — Completeness under window occlusion (W=7)
# =============================================================================
print('\n[H] completeness under window occlusion')
rows_H = []
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, n in zip(axes, names):
    logits = features[n]['logits']
    sum_w  = window_contrib[n]['window_contrib'].sum(axis=1)   # sum across 58 windows
    sum_s  = slice_contrib[n]['contrib'].sum(axis=1)            # compare single-slice
    labels = features[n]['labels']

    ratio_s = float(np.median(np.abs(sum_s) / (np.abs(logits) + 1e-9)))
    ratio_w = float(np.median(np.abs(sum_w) / (np.abs(logits) + 1e-9)))

    ax.scatter(logits[labels == 0], sum_w[labels == 0], s=5, alpha=0.35,
               color='steelblue', label=f'healthy (n={int((labels == 0).sum())})')
    ax.scatter(logits[labels == 1], sum_w[labels == 1], s=5, alpha=0.35,
               color='darkred', label=f'glaucoma (n={int((labels == 1).sum())})')
    xlim = max(np.abs(logits).max(), 0.1)
    ax.plot([-xlim, xlim], [-xlim, xlim], 'k--', lw=0.8, alpha=0.5,
            label='y = x (full additivity)')
    ax.axhline(0, color='k', lw=0.3, alpha=0.4)
    ax.axvline(0, color='k', lw=0.3, alpha=0.4)
    ax.set_xlim(-xlim, xlim)
    ax.set_xlabel('Baseline logit', fontsize=10)
    if ax is axes[0]:
        ax.set_ylabel('Sum of window contributions (58 windows, W=7)', fontsize=10)
    ax.set_title(f'{displays[n]}\n'
                 f'single-slice ratio = {ratio_s:.1%} → window(W=7) = {ratio_w:.1%}',
                 fontsize=10)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.2)

    rows_H.append({
        'model': displays[n],
        'median_ratio_single_slice': round(ratio_s, 4),
        'median_ratio_window_W7':    round(ratio_w, 4),
        'amplification_factor':      round(ratio_w / max(ratio_s, 1e-9), 2),
    })
    print(f'  {n:10s}: single={ratio_s:.1%}  window={ratio_w:.1%}  '
          f'amp={ratio_w / max(ratio_s, 1e-9):.1f}x')

fig.suptitle('Completeness gap under window occlusion (W=7)\n'
             'Window occlusion recovers more of the logit than single-slice zero-masking — '
             'but still far from y = x (full additivity)',
             fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, '10_completeness_window.png'), dpi=140, bbox_inches='tight')
plt.close(fig)

# =============================================================================
# I — Per-patch bootstrap CI, count significant patches
# =============================================================================
print('\n[I] per-patch bootstrap CI (B=500)')
rows_I = []
B = 500
rng = np.random.default_rng(42)

fig, axes = plt.subplots(3, 2, figsize=(10, 14))
for i, n in enumerate(names):
    for j, slice_s in enumerate([20, 43]):
        pa = patch_aggregate[n][slice_s]
        contrib = pa['patch_contrib']         # (3000, 256)
        labels  = pa['labels']
        pos     = contrib[labels == 1]         # (1466, 256)

        # Bootstrap per-patch mean
        boot = np.zeros((B, 256), dtype=np.float32)
        for b in range(B):
            idx = rng.integers(0, len(pos), size=len(pos))
            boot[b] = pos[idx].mean(axis=0)
        mean = boot.mean(axis=0)
        lo = np.percentile(boot, 2.5,  axis=0)
        hi = np.percentile(boot, 97.5, axis=0)
        # "Significant" if CI does NOT cross zero
        pos_sig = (lo > 0).sum()
        neg_sig = (hi < 0).sum()
        sig_mask = (lo > 0) | (hi < 0)
        n_sig = int(sig_mask.sum())
        print(f'  {n:10s} slice {slice_s:>2}: {n_sig}/256 patches sig '
              f'({pos_sig} pos, {neg_sig} neg, 95% CI excludes 0)')

        # Overlay mean heatmap with significance outline
        ax = axes[i, j]
        heat = mean.reshape(16, 16)
        sig_grid = sig_mask.reshape(16, 16)
        vmax = max(np.abs(heat).max(), 1e-12)
        ax.imshow(heat, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                  interpolation='nearest')
        # Mark significant patches with a green dot
        ys, xs = np.where(sig_grid)
        ax.scatter(xs, ys, s=18, marker='s', edgecolor='lime',
                   facecolor='none', linewidths=1.0)
        ax.set_title(f'{displays[n]} | subset {slice_s} (native ~{int(axial[slice_s])})\n'
                     f'{n_sig}/256 patches with 95% CI excluding 0',
                     fontsize=9)
        ax.axis('off')

        rows_I.append({
            'model':             displays[n],
            'target_slice':      slice_s,
            'native_slice':      int(axial[slice_s]),
            'n_sig_patches':     n_sig,
            'n_pos_sig':         int(pos_sig),
            'n_neg_sig':         int(neg_sig),
            'fraction_sig':      round(n_sig / 256, 3),
            'max_abs_mean':      float(np.abs(mean).max()),
        })

fig.suptitle('Aggregate patch heatmap with 95% bootstrap CI significance\n'
             '(lime squares = patch 95% CI does NOT cross zero, B=500)',
             fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, '11_patch_ci_significance.png'),
            dpi=140, bbox_inches='tight')
plt.close(fig)

# =============================================================================
# G — Disc rim symmetry (peak at subset 20 vs peak at subset 43)
# =============================================================================
print('\n[G] disc rim symmetry (subset 20 vs subset 43 per volume)')
rows_G = []
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, n in zip(axes, names):
    C = slice_contrib[n]['contrib']          # (3000, 64)
    labels = slice_contrib[n]['labels']
    c20 = C[:, 20]
    c43 = C[:, 43]

    pos_mask = labels == 1
    r_pos = float(np.corrcoef(c20[pos_mask], c43[pos_mask])[0, 1])
    r_all = float(np.corrcoef(c20, c43)[0, 1])
    mean_ratio = float(np.median(c20[pos_mask] / (c43[pos_mask] + 1e-9)))

    ax.scatter(c20[labels == 0], c43[labels == 0], s=5, alpha=0.3,
               color='steelblue', label=f'healthy (n={int((labels == 0).sum())})')
    ax.scatter(c20[pos_mask], c43[pos_mask], s=5, alpha=0.3,
               color='darkred', label=f'glaucoma (n={int(pos_mask.sum())})')
    lim = max(np.abs(c20).max(), np.abs(c43).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.7, alpha=0.5, label='y = x')
    ax.axhline(0, color='k', lw=0.3, alpha=0.4)
    ax.axvline(0, color='k', lw=0.3, alpha=0.4)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('Contrib at subset 20 (native ~63, superior rim)', fontsize=9)
    if ax is axes[0]:
        ax.set_ylabel('Contrib at subset 43 (native ~137, inferior rim)', fontsize=9)
    ax.set_title(f'{displays[n]}\n'
                 f'r_glau = {r_pos:+.3f}  |  r_all = {r_all:+.3f}',
                 fontsize=10)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.2)

    rows_G.append({
        'model':                  displays[n],
        'pearson_r_glaucoma':     round(r_pos, 4),
        'pearson_r_all':          round(r_all, 4),
        'median_ratio_subset20_over_43': round(mean_ratio, 3),
    })
    print(f'  {n:10s}: r_glau = {r_pos:+.3f}  r_all = {r_all:+.3f}  '
          f'median(20/43) = {mean_ratio:+.2f}')

fig.suptitle('Disc rim symmetry test: do the two peaks come from the same anatomic structure?\n'
             'Correlated across volumes → same structure (both rims of the disc). Uncorrelated → independent signals.',
             fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, '12_disc_rim_symmetry.png'), dpi=140, bbox_inches='tight')
plt.close(fig)

# =============================================================================
# C — Attribution magnitude vs prediction confidence
# =============================================================================
print('\n[C] attribution magnitude vs |baseline logit|')
rows_C = []
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, n in zip(axes, names):
    C = slice_contrib[n]['contrib']
    logits = features[n]['logits']
    labels = slice_contrib[n]['labels']

    peak_abs = np.abs(C).max(axis=1)         # per-volume peak |slice contrib|
    r_all = float(np.corrcoef(peak_abs, np.abs(logits))[0, 1])

    ax.scatter(np.abs(logits[labels == 0]), peak_abs[labels == 0],
               s=5, alpha=0.3, color='steelblue',
               label=f'healthy (n={int((labels == 0).sum())})')
    ax.scatter(np.abs(logits[labels == 1]), peak_abs[labels == 1],
               s=5, alpha=0.3, color='darkred',
               label=f'glaucoma (n={int((labels == 1).sum())})')
    ax.set_xlabel('|Baseline logit| (prediction confidence)', fontsize=10)
    if ax is axes[0]:
        ax.set_ylabel('Per-volume peak |slice contribution|', fontsize=10)
    ax.set_title(f'{displays[n]}\n'
                 f'Pearson r = {r_all:+.3f}', fontsize=10)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.2)

    rows_C.append({
        'model':       displays[n],
        'pearson_r_peak_contrib_vs_abs_logit': round(r_all, 4),
        'median_peak_contrib_pos':  float(np.median(peak_abs[labels == 1])),
        'median_peak_contrib_neg':  float(np.median(peak_abs[labels == 0])),
    })
    print(f'  {n:10s}: r = {r_all:+.3f}')

fig.suptitle('Attribution magnitude vs prediction confidence\n'
             'Strong positive r → high-confidence cases have more concentrated attribution. '
             'Near-zero r → attribution shape is invariant to confidence.',
             fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, '13_attribution_vs_confidence.png'),
            dpi=140, bbox_inches='tight')
plt.close(fig)

# =============================================================================
# Excel: rebuild from scratch with all 11 sheets
# =============================================================================
print('\n[xlsx] rebuilding with 11 sheets')
xlsx = os.path.join(TABLES, 'summary.xlsx')

# Reconstruct sheets 1-6 from the same raw data sources used in _build.py + _consolidate.py
W = 7
headline_rows = []
for n in names:
    d_c = slice_contrib[n]; d_f = features[n]
    headline_rows.append({
        'model':   displays[n], 'probe': n,
        'n_glaucoma': int((d_c['labels'] == 1).sum()),
        'n_healthy':  int((d_c['labels'] == 0).sum()),
        'test_auc':    float(d_f['test_auc']) if 'test_auc' in d_f else None,
        'median_abs_logit': float(np.median(np.abs(d_f['logits']))),
    })

slice_peaks = []
for n in names:
    mean_pos = slice_contrib[n]['mean_pos']
    top3 = np.argsort(mean_pos)[-3:][::-1]
    for rank, s in enumerate(top3, 1):
        slice_peaks.append({
            'model': displays[n], 'rank': rank,
            'subset_idx': int(s), 'native_slice': int(axial[s]),
            'mean_pos_contrib': float(mean_pos[s]),
            'mean_neg_contrib': float(slice_contrib[n]['mean_neg'][s]),
        })

window_peaks = []
for n in names:
    c = window_contrib[n]['window_contrib']; y = window_contrib[n]['labels']
    mp = c[y == 1].mean(axis=0)
    top3 = np.argsort(mp)[-3:][::-1]
    for rank, i in enumerate(top3, 1):
        window_peaks.append({
            'model': displays[n], 'rank': rank,
            'window_start_subset': int(i), 'window_end_subset': int(i + W - 1),
            'center_subset': int(i + (W - 1) // 2),
            'center_native': int(axial[i + (W - 1) // 2]),
            'peak_pos_contrib': float(mp[i]),
        })

patch_peaks = []
for n in names:
    for s in [20, 43]:
        p = patch_aggregate[n][s]
        pos_mean = p['patch_contrib'][p['labels'] == 1].mean(axis=0)
        order = np.argsort(pos_mean)[-5:][::-1]
        for rank, idx in enumerate(order, 1):
            patch_peaks.append({
                'model': displays[n], 'target_slice': s,
                'native_slice': int(axial[s]), 'rank': rank,
                'patch_idx': int(idx),
                'patch_row': int(idx // 16), 'patch_col': int(idx % 16),
                'pos_mean_contrib': float(pos_mean[idx]),
            })

confusion_rows = []
completeness_rows = []
for n in names:
    labels = features[n]['labels']; probs = features[n]['probs']; logits = features[n]['logits']
    pred = (probs >= 0.5).astype(int)
    tp = int(((labels == 1) & (pred == 1)).sum())
    fn = int(((labels == 1) & (pred == 0)).sum())
    tn = int(((labels == 0) & (pred == 0)).sum())
    fp = int(((labels == 0) & (pred == 1)).sum())
    confusion_rows.append({
        'model': displays[n],
        'TP': tp, 'FN': fn, 'TN': tn, 'FP': fp,
        'accuracy_at_0.5':    (tp + tn) / len(labels),
        'sensitivity_at_0.5': tp / max(tp + fn, 1),
        'specificity_at_0.5': tn / max(tn + fp, 1),
    })
    sum_c = slice_contrib[n]['contrib'].sum(axis=1)
    ratio = np.abs(sum_c) / (np.abs(logits) + 1e-9)
    completeness_rows.append({
        'model': displays[n],
        'median_|baseline_logit|':  float(np.median(np.abs(logits))),
        'median_|sum(slice_contrib)|': float(np.median(np.abs(sum_c))),
        'median_ratio_sum_over_logit': float(np.median(ratio)),
        'mean_ratio_sum_over_logit':   float(ratio.mean()),
        'interpretation': f'Median single-slice contribs explain '
                          f'{np.median(ratio)*100:.1f}% of the baseline logit.',
    })

# Retry loop for OneDrive-held files
import time
for attempt in range(5):
    try:
        if os.path.exists(xlsx):
            os.remove(xlsx)
        break
    except (OSError, PermissionError):
        time.sleep(1.5)
else:
    print(f'  WARN: could not delete {xlsx}; writing anyway (may error if locked)')

with pd.ExcelWriter(xlsx, engine='openpyxl') as writer:
    pd.DataFrame(headline_rows   ).to_excel(writer, sheet_name='01_headline_auc',          index=False)
    pd.DataFrame(slice_peaks     ).to_excel(writer, sheet_name='02_slice_peaks',            index=False)
    pd.DataFrame(window_peaks    ).to_excel(writer, sheet_name='03_window_peaks',           index=False)
    pd.DataFrame(patch_peaks     ).to_excel(writer, sheet_name='04_patch_peaks',            index=False)
    pd.DataFrame(confusion_rows  ).to_excel(writer, sheet_name='05_confusion_matrix',       index=False)
    pd.DataFrame(completeness_rows).to_excel(writer, sheet_name='06_completeness',          index=False)
    pd.DataFrame(rows_A          ).to_excel(writer, sheet_name='07_cross_probe_patch',      index=False)
    pd.DataFrame(rows_H          ).to_excel(writer, sheet_name='08_completeness_window',    index=False)
    pd.DataFrame(rows_I          ).to_excel(writer, sheet_name='09_patch_significance',     index=False)
    pd.DataFrame(rows_G          ).to_excel(writer, sheet_name='10_disc_rim_symmetry',      index=False)
    pd.DataFrame(rows_C          ).to_excel(writer, sheet_name='11_magnitude_vs_confidence', index=False)
    for ws in writer.sheets.values():
        ws.freeze_panes = 'A2'
print(f'  wrote {xlsx}  (11 sheets)')

# =============================================================================
# Resize plots that exceed the 2576x2576 viewer limit
# =============================================================================
print('\n[resize] capping plots at 2200 px')
from PIL import Image
for name in ['09_cross_probe_patch_agreement.png',
             '10_completeness_window.png',
             '11_patch_ci_significance.png',
             '12_disc_rim_symmetry.png',
             '13_attribution_vs_confidence.png']:
    p = os.path.join(PLOTS, name)
    im = Image.open(p)
    if max(im.size) > 2200:
        r = 2200 / max(im.size)
        im = im.resize((int(im.size[0] * r), int(im.size[1] * r)), Image.LANCZOS)
        im.save(p, optimize=True)
        print(f'  resized {name} -> {im.size}')

print('\nDONE')
