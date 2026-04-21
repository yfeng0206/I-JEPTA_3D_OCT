"""Test the OD/OS storage-mirror hypothesis on per-volume slice-contribution curves.

Prior observation (docs/experiments/interpretability.md): population-averaged slice
attribution is bimodal with peaks at native positions ~63 and ~137. A naive "bilateral
disc rim" reading would predict positive per-volume correlation between the two peaks;
observed is slightly negative (−0.07 to −0.22). That points at OD/OS axial-storage
mixing: right- and left-eye scans stored with flipped slice-axis direction.

FairVision SLOs are disc-centred, so a simple disc-position detector can't reconstruct
laterality. This script tests the hypothesis indirectly:

  1. K-means (k=2) on L2-normalised per-volume 64-dim contrib curves (glaucoma only).
  2. Mirror test: corr(cluster1_mean, flip(cluster2_mean)) should be high if clusters
     are laterality-flipped versions of each other.
  3. Pseudo-OD/OS realignment: flip the cluster with the higher-slice-idx peak and
     recompute the population mean. If OD/OS mirror is the dominant story, the
     bimodal curve should collapse toward a single peak.

Outputs:
  results/summary/14_odos_mirror_test.png    mirror evidence
  results/summary/15_odos_aligned_curves.png realignment evidence

Reads slice_contrib_<probe>.npz from I-JEPA_archive/04_interpretability_v2_fp32/.
"""
import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


REPO     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE  = r'C:\Users\garyfeng\OneDrive - Microsoft\Desktop\I-JEPA_archive\04_interpretability_v2_fp32\slice_contributions'
OUT_DIR  = os.path.join(REPO, 'results', 'summary')
PROBES   = ['meanpool', 'crossattn', 'd1']
AXIAL    = np.linspace(0, 199, 64)


def cluster_and_flip(C_glau):
    """Return (cluster_labels, flip_mask) for per-volume contribs C_glau (N, 64).
    flip_mask is True for the cluster whose mean peak is at the higher slice index
    (so flipping puts both clusters' peaks at the lower slice index)."""
    C_n = C_glau / (np.linalg.norm(C_glau, axis=1, keepdims=True) + 1e-9)
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    lbl = km.fit_predict(C_n)
    m0 = C_glau[lbl == 0].mean(axis=0)
    m1 = C_glau[lbl == 1].mean(axis=0)
    flip_cluster = 0 if int(np.argmax(m0)) > int(np.argmax(m1)) else 1
    flip_mask = lbl == flip_cluster
    return lbl, flip_mask


def mirror_test():
    print('=== Test 1: per-volume cluster means; mirror correlation ===')
    print(f'{"probe":10s}  {"corr(c1,c2)":>12s}  {"corr(c1,flip(c2))":>18s}  {"n1":>4s} {"n2":>4s}  verdict')
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for i, probe in enumerate(PROBES):
        d = np.load(os.path.join(ARCHIVE, f'slice_contrib_{probe}.npz'))
        C = d['contrib'][d['labels'] == 1]
        lbl, _ = cluster_and_flip(C)
        c1 = C[lbl == 0].mean(axis=0); n1 = int((lbl == 0).sum())
        c2 = C[lbl == 1].mean(axis=0); n2 = int((lbl == 1).sum())
        r_raw = float(np.corrcoef(c1, c2)[0, 1])
        r_mir = float(np.corrcoef(c1, c2[::-1])[0, 1])
        verdict = 'MIRROR' if r_mir > 0.7 and r_mir - r_raw > 0.4 else (
            'same-shape' if r_raw > 0.7 else 'noisy')
        print(f'{probe:10s}  {r_raw:>+12.3f}  {r_mir:>+18.3f}  {n1:>4d} {n2:>4d}  {verdict}')

        ax = axes[i, 0]
        ax.plot(AXIAL, c1, color='steelblue', lw=2, label=f'cluster 1 (n={n1})')
        ax.plot(AXIAL, c2, color='darkred',   lw=2, label=f'cluster 2 (n={n2})')
        ax.axhline(0, color='k', lw=0.4, alpha=0.5)
        ax.set_xlabel('slice native position (/199)')
        ax.set_ylabel('mean slice Δlogit')
        ax.set_title(f'{probe} — raw cluster means  corr(c1,c2)={r_raw:+.3f}')
        ax.legend(); ax.grid(alpha=0.3)

        ax = axes[i, 1]
        ax.plot(AXIAL, c1,          color='steelblue', lw=2, label='cluster 1')
        ax.plot(AXIAL, c2[::-1], '--', color='darkred',  lw=2, label='cluster 2 FLIPPED')
        ax.axhline(0, color='k', lw=0.4, alpha=0.5)
        ax.set_xlabel('slice native position (/199)')
        ax.set_title(f'{probe} — mirror test  corr(c1,flip(c2))={r_mir:+.3f}')
        ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle('OD/OS mixing hypothesis: k=2 clustering on per-volume slice curves (glaucoma)\n'
                 'If mixing is real, right column shows two curves overlapping after flip.', fontsize=11)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, '14_odos_mirror_test.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')


def realignment_test():
    print('\n=== Test 2: pseudo-OD/OS realignment ===')
    fig, axes = plt.subplots(3, 1, figsize=(10, 11))
    for i, probe in enumerate(PROBES):
        d = np.load(os.path.join(ARCHIVE, f'slice_contrib_{probe}.npz'))
        C = d['contrib'][d['labels'] == 1]
        _, flip_mask = cluster_and_flip(C)
        aligned = C.copy()
        aligned[flip_mask] = aligned[flip_mask][:, ::-1]

        raw_mean = C.mean(axis=0)
        aligned_mean = aligned.mean(axis=0)
        def n_peaks(curve):
            peaks, _ = find_peaks(curve, prominence=0.2 * (curve.max() - curve.min()))
            return len(peaks)
        print(f'  {probe:10s}  peaks: raw={n_peaks(raw_mean)}  aligned={n_peaks(aligned_mean)}  '
              f'(flipped {int(flip_mask.sum())}/{len(C)})')

        ax = axes[i]
        ax.plot(AXIAL, raw_mean, color='black', lw=2,
                label=f'raw population mean ({len(C)} glaucoma vols)')
        ax.plot(AXIAL, aligned_mean, color='darkgreen', lw=2,
                label=f'after flipping cluster-2 ({int(flip_mask.sum())} vols)')
        ax.axhline(0, color='k', lw=0.4, alpha=0.5)
        for v in (63, 95, 137):
            ax.axvline(v, color='gray', ls=':', alpha=0.5)
        ax.set_xlabel('slice native position (/199)')
        ax.set_ylabel('mean slice Δlogit')
        ax.set_title(f'{probe}: raw BIMODAL → after OD/OS realignment')
        ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle('Pseudo-OD/OS realignment: k=2 cluster label as laterality, one cluster flipped.',
                 fontsize=10)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, '15_odos_aligned_curves.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    mirror_test()
    realignment_test()
