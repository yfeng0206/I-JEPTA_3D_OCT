"""Reassemble heatmap_grid.png from 6 curated individual-volume overlays.

Phase 3 of interpretability.py writes 60 per-volume × 3-slice overlay PNGs
to `<output>/heatmaps_<model>/vol####_slice##.png`. The published
`results/summary/heatmap_grid.png` is a manually-curated 6-row composite
(1 TP + 1 TN per probe) of those PNGs with annotation text on top of each
row.

This script reproduces that composite from a given AML run's blob output
prefix. Used to refresh the grid after the fp16 autocast fix was re-run
(AML job yellow_hook_nppdpmd29y).

Usage:
    python scripts/assemble_heatmap_grid.py \\
        --blob-prefix ijepa-interpretability/interpretability_20260421_HHMMSS
"""

import argparse
import io
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(REPO, 'results', 'summary', 'heatmap_grid.png')

# (model_key, display_prefix, vol_idx, slice_subset_idx, label_text,
#  pred, slice_contrib, description, native_axial)
ROWS = [
    ('meanpool',  'meanpool',  661, 40, 'glaucoma', 0.98, +0.109, 'RNFL thinning visible',     126),
    ('meanpool',  'meanpool',  388,  7, 'healthy',  0.18, -0.075, 'Smooth healthy retina',      22),
    ('crossattn', 'crossattn', 1592, 54, 'glaucoma', 0.94, +0.170, 'Localized pathology',       170),
    ('crossattn', 'crossattn', 1991,  6, 'healthy',  0.19, -0.055, 'Peripheral normal',          18),
    ('d1',        'd1',         100, 45, 'glaucoma', 0.94, +0.188, 'Disc rim B-scan',           142),
    ('d1',        'd1',        2469, 49, 'healthy',  0.07, -0.432, 'Strong healthy suppressor', 154),
]

BLOB_ACCOUNT = 'turbosxnpesu01'
BLOB_CONTAINER = 'azureml-blobstore-d2ed9711-de3b-4979-a764-8ba7535b1da0'


def download_png(blob_prefix, model, vol, slice_idx):
    """Pull <blob_prefix>/heatmaps_<model>/vol####_slice##.png bytes."""
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobClient

    blob_name = f'{blob_prefix}/heatmaps_{model}/vol{vol:04d}_slice{slice_idx:02d}.png'
    cred = DefaultAzureCredential()
    bc = BlobClient(
        account_url=f'https://{BLOB_ACCOUNT}.blob.core.windows.net',
        container_name=BLOB_CONTAINER,
        blob_name=blob_name,
        credential=cred,
    )
    return bc.download_blob().readall()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--blob-prefix', required=True,
                    help='AML output blob prefix, e.g. '
                         'ijepa-interpretability/interpretability_20260421_HHMMSS')
    ap.add_argument('--local-dir', default=None,
                    help='Optional: read already-downloaded PNGs from here '
                         'instead of blob (useful for local re-runs). Expected '
                         'layout: <local-dir>/heatmaps_<model>/vol####_slice##.png')
    args = ap.parse_args()

    # Fetch the 6 PNGs
    imgs = []
    for (model, _disp, vol, s, *_) in ROWS:
        if args.local_dir:
            p = os.path.join(args.local_dir, f'heatmaps_{model}',
                             f'vol{vol:04d}_slice{s:02d}.png')
            print(f'reading {p}')
            imgs.append(Image.open(p))
        else:
            print(f'downloading {model} vol{vol:04d} slice{s:02d}...')
            data = download_png(args.blob_prefix, model, vol, s)
            imgs.append(Image.open(io.BytesIO(data)))

    # Compose 6-row figure. Each row shows the downloaded PNG with a 3-line
    # annotation above it. The PNG is already (slice | heatmap overlay).
    fig_h_per_row = 3.5
    fig, axes = plt.subplots(6, 1, figsize=(10, fig_h_per_row * 6))

    for ax, img, (model, disp, vol, s, lbl, pred, contrib, desc, axial) in zip(
            axes, imgs, ROWS):
        ax.imshow(np.asarray(img))
        ax.axis('off')
        sign = '+' if contrib >= 0 else ''
        direction = 'pushes toward glaucoma' if contrib > 0 else 'pushes toward healthy'
        title = (f'[{disp}]  vol {vol:04d}  slice {s}  (axial ~{axial}/199)\n'
                 f'label={lbl}  pred={pred:.2f}  slice contribution={sign}{contrib:.3f} ({direction})\n'
                 f'{desc}')
        ax.set_title(title, fontsize=9, loc='left', pad=6)

    fig.suptitle(
        'Occlusion attribution — representative B-scans across the 3 fine-tune probes\n'
        'Left pane of each row: original B-scan. Right pane: per-patch Δlogit overlay.',
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(OUT_PATH, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'\nwrote {OUT_PATH}  ({os.path.getsize(OUT_PATH)/1e3:.0f} KB)')


if __name__ == '__main__':
    main()
