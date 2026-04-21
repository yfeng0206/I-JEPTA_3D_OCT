"""Local post-hoc: demonstrate the effect of shared-scale (B) and
slice-mean-subtraction (C) on patch-occlusion heatmaps.

The current heatmap_grid.png uses per-image color scaling
(`vmax = |contrib|.max()` per subplot). That makes a map whose deltas are
all in, say, +0.002..+0.008 look saturated blue — indistinguishable from a
genuinely flat map. This script rebuilds six curated examples using the
aggregate patch-occlusion data (slices 20 and 43, 3000 volumes, 3 models)
and renders 3 variants per example:

  Col 1  "original"   per-image zero-centered scale   (vmax per subplot)
  Col 2  "B"          shared zero-centered scale      (one vmax across grid)
  Col 3  "B + C"      Δ_local(p) = Δ(p) − mean(Δ)     (shared scale)

Each subplot title includes min/mean/max/std (A) so the raw dynamic range
is visible.

Runs on the `patch_aggregate_*.npz` files under I-JEPA_results_presentation/
patch_aggregate/ — no AML job needed. Writes the figure to both:
  results/summary/heatmap_grid_BC.png           (repo)
  I-JEPA_results_presentation/plots/14_heatmap_BC.png   (presentation)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


PRESENT = r'C:\Users\garyfeng\OneDrive - Microsoft\Desktop\I-JEPA_results_presentation'
REPO    = r'C:\Users\garyfeng\OneDrive - Microsoft\Desktop\I-JEPA'

PATCH_DIR = os.path.join(PRESENT, 'patch_aggregate')
OUT_REPO    = os.path.join(REPO, 'results', 'summary', 'heatmap_grid_BC.png')
OUT_PRESENT = os.path.join(PRESENT, 'plots', '14_heatmap_BC.png')

MODELS = [
    ('meanpool',  'FT + MeanPool'),
    ('crossattn', 'FT + CrossAttnPool'),
    ('d1',        'FT + AttentiveProbe d=1'),
]
SLICES = [20, 43]             # the two peaks; native ~63 and ~137
AXIAL = np.linspace(0, 199, 64)


def load_patch_data():
    """Return dict keyed by (model_name, slice_idx) -> npz dict."""
    data = {}
    for n, _ in MODELS:
        for s in SLICES:
            p = os.path.join(PATCH_DIR, f'patch_aggregate_{n}_slice{s:02d}.npz')
            d = np.load(p)
            data[(n, s)] = {
                'patch_contrib': d['patch_contrib'],  # (3000, 256)
                'baseline':      d['baseline'],
                'labels':        d['labels'],
                'probs':         d['probs'],
            }
    return data


def pick_curated(data):
    """For each (model, {TP, TN}) pick one volume and the slice (20 or 43)
    where its |mean patch contrib| is largest.

    Returns a list of 6 tuples: (model_name, class, vol_idx, slice_idx).
    """
    picks = []
    for n, _ in MODELS:
        # Merge the two slices into a (N, 2, 256) array for easier selection
        pc_20 = data[(n, 20)]['patch_contrib']     # (3000, 256)
        pc_43 = data[(n, 43)]['patch_contrib']
        labels = data[(n, 20)]['labels']            # same ordering
        probs  = data[(n, 20)]['probs']

        # Per-volume "signal magnitude" at each slice = |mean over patches|
        mag_20 = np.abs(pc_20.mean(axis=1))
        mag_43 = np.abs(pc_43.mean(axis=1))
        # For each vol, pick the slice with the larger magnitude
        slice_choice = np.where(mag_43 > mag_20, 43, 20)
        mag_best = np.maximum(mag_20, mag_43)

        # TP: label=1 & prob >= 0.5, sort by prob desc × mag desc
        tp_mask = (labels == 1) & (probs >= 0.5)
        tn_mask = (labels == 0) & (probs <  0.5)

        # Rank by (prob * mag) for TP, ((1 - prob) * mag) for TN
        tp_score = np.where(tp_mask, probs * mag_best, -np.inf)
        tn_score = np.where(tn_mask, (1 - probs) * mag_best, -np.inf)

        tp_vol = int(np.argmax(tp_score))
        tn_vol = int(np.argmax(tn_score))

        picks.append((n, 'TP', tp_vol, int(slice_choice[tp_vol])))
        picks.append((n, 'TN', tn_vol, int(slice_choice[tn_vol])))

    return picks


def render(picks, data):
    # Gather per-cell (16,16) maps under three transforms first so we can
    # compute shared vmax for B and B+C.
    cells = []           # each: dict(name_disp, cls, vol, slice, raw, local, label, prob)
    for (n, cls, vol, s) in picks:
        pc = data[(n, s)]['patch_contrib'][vol]   # (256,)
        label = int(data[(n, s)]['labels'][vol])
        prob  = float(data[(n, s)]['probs'][vol])
        raw   = pc.reshape(16, 16)
        local = (pc - pc.mean()).reshape(16, 16)
        cells.append({
            'name': n, 'cls': cls, 'vol': vol, 'slice': s,
            'label': label, 'prob': prob,
            'raw': raw, 'local': local,
            'mean': float(pc.mean()), 'std': float(pc.std()),
            'min':  float(pc.min()),  'max':  float(pc.max()),
        })

    vmax_B  = max(np.abs(c['raw']).max()   for c in cells) + 1e-12
    vmax_BC = max(np.abs(c['local']).max()  for c in cells) + 1e-12
    print(f'[shared scales] B: ±{vmax_B:.4f}   B+C: ±{vmax_BC:.4f}')

    displays = dict(MODELS)
    n_rows = len(cells)
    fig, axes = plt.subplots(n_rows, 3, figsize=(11, 2.8 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    col_titles = [
        'Col 1: per-image zero-centered scale\n(current default, artifact-prone)',
        f'Col 2 — B: shared scale ±{vmax_B:.3f}\n(honest cross-cell comparison)',
        f'Col 3 — B + C: Δ_local = Δ − mean(Δ)\nshared scale ±{vmax_BC:.3f} (reveals local structure)',
    ]

    for row, c in enumerate(cells):
        native = int(AXIAL[c['slice']])
        row_label = (f'{displays[c["name"]]}\n'
                     f'{c["cls"]} vol={c["vol"]:04d}  slice={c["slice"]} '
                     f'(native ~{native}/199)\n'
                     f'label={c["label"]}  prob={c["prob"]:.2f}')

        # Per-image stats caption
        stats = (f'min {c["min"]:+.3f}  max {c["max"]:+.3f}\n'
                 f'mean {c["mean"]:+.3f}  std {c["std"]:.3f}')

        # Col 1: per-image scale
        v1 = max(np.abs(c['raw']).max(), 1e-12)
        im0 = axes[row, 0].imshow(c['raw'], cmap='RdBu_r',
                                  vmin=-v1, vmax=v1, interpolation='nearest')
        axes[row, 0].set_ylabel(row_label, fontsize=8)
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
        axes[row, 0].set_title(f'±{v1:.3f} (per-image)\n{stats}', fontsize=7)
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.02)

        # Col 2: shared B scale
        im1 = axes[row, 1].imshow(c['raw'], cmap='RdBu_r',
                                  vmin=-vmax_B, vmax=vmax_B, interpolation='nearest')
        pct_same_sign = 100 * (c['raw'].flatten() * np.sign(c['mean']) > 0).mean()
        axes[row, 1].set_xticks([]); axes[row, 1].set_yticks([])
        axes[row, 1].set_title(
            f'shared ±{vmax_B:.3f}\n'
            f'{pct_same_sign:.0f}% patches share sign of mean',
            fontsize=7,
        )
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.02)

        # Col 3: B + C (shared scale, mean-subtracted)
        im2 = axes[row, 2].imshow(c['local'], cmap='RdBu_r',
                                  vmin=-vmax_BC, vmax=vmax_BC, interpolation='nearest')
        axes[row, 2].set_xticks([]); axes[row, 2].set_yticks([])
        axes[row, 2].set_title(
            f'shared ±{vmax_BC:.3f}\n'
            f'local spread std={c["std"]:.3f}',
            fontsize=7,
        )
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.02)

    # Column headers as text above row 0
    for col, t in enumerate(col_titles):
        axes[0, col].annotate(
            t, xy=(0.5, 1.35), xycoords='axes fraction',
            ha='center', va='bottom', fontsize=9, fontweight='bold',
        )

    fig.suptitle(
        'Effect of shared color scale (B) and slice-mean subtraction (C) '
        'on patch-occlusion heatmaps\n'
        '(same raw Δlogit values; only the visualization changes)',
        fontsize=11, y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    for path in (OUT_REPO, OUT_PRESENT):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=140, bbox_inches='tight')
        print(f'wrote {path}  ({os.path.getsize(path)/1e3:.0f} KB)')
    plt.close(fig)


def main():
    data = load_patch_data()
    picks = pick_curated(data)
    print('curated:')
    for (n, cls, vol, s) in picks:
        print(f'  {n:10s} {cls}  vol={vol:04d}  slice={s}  '
              f'(native ~{int(AXIAL[s])})')
    render(picks, data)


if __name__ == '__main__':
    main()
