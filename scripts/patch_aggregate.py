#!/usr/bin/env python
"""Aggregate patch-level occlusion attribution on target slices across the
full test set.

For each fine-tune model and each target subset slice (default: 20 and 43,
which correspond to native volume positions ~63 and ~137 — the two peaks in
the cross-model slice-attribution curve):

  For every volume in the Test split (3000 volumes):
    1. Re-forward the target slice through the fine-tuned encoder,
       keeping per-patch tokens (256, 768).
    2. Splice the patch-token set into the cached (64, 768) slice features
       for that volume; compute baseline logit.
    3. For each of 256 patches: zero the patch, recompute slice-token mean,
       substitute into the cached (64, 768), run probe + head, measure
       Δlogit. Batched as (256, 64, 768) per volume.
    4. Save the (256,) attribution vector, plus label + prediction.

Output: patch_aggregate_<model>_slice<s>.npz with arrays:
  patch_contrib:  (N_vols, 256) float32     — per-volume per-patch Δlogit
  baseline:       (N_vols,)     float32     — logit with full volume
  labels:         (N_vols,)     int8
  probs:          (N_vols,)     float32     — sigmoid of baseline logit
  target_slice_idx: int

Also emits a summary figure: 3 models × 2 slices, showing mean per-patch
attribution for TP vs TN (glaucoma mean minus healthy mean = diff map).
"""

import argparse
import gc
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.models.vision_transformer import VisionTransformer
from src.models.attentive_pool_minimal import CrossAttnPool, MeanPool
from src.datasets.oct_volumes import OCTVolumeDataset
from src.helper import _VIT_CONFIGS
from src.eval_downstream import AttentiveProbe, LinearHead, imagenet_normalize


MODELS = [
    {'name': 'meanpool',  'probe_type': 'mean_pool',       'probe_kwargs': {}},
    {'name': 'crossattn', 'probe_type': 'cross_attn_pool', 'probe_kwargs': {'head_dim': 64}},
    {'name': 'd1',        'probe_type': 'attentive',       'probe_kwargs': {'num_heads': 12, 'depth': 1}},
]


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

def build_probe(probe_type, num_slices, embed_dim, probe_kwargs):
    if probe_type == 'mean_pool':
        return MeanPool(num_slices=num_slices, embed_dim=embed_dim)
    if probe_type == 'cross_attn_pool':
        return CrossAttnPool(num_slices=num_slices, embed_dim=embed_dim,
                             **probe_kwargs)
    if probe_type == 'attentive':
        return AttentiveProbe(num_slices=num_slices, embed_dim=embed_dim,
                              **probe_kwargs)
    raise ValueError(f'Unknown probe_type={probe_type!r}')


def load_model(ckpt_path, probe_type, probe_kwargs, num_slices, device):
    vit_cfg = _VIT_CONFIGS['vit_base']
    encoder = VisionTransformer(
        img_size=256, patch_size=16,
        embed_dim=vit_cfg['embed_dim'],
        depth=vit_cfg['depth'],
        num_heads=vit_cfg['num_heads'],
    )
    probe = build_probe(probe_type, num_slices, vit_cfg['embed_dim'], probe_kwargs)
    head = LinearHead(in_dim=vit_cfg['embed_dim'])

    # weights_only=False for PyTorch 2.6+ compatibility (our checkpoints
    # include a numpy.float val_auc that's not in the default safe-globals
    # list; we trust the source).
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except TypeError:
        # PyTorch <2.0 doesn't accept the weights_only kwarg
        ckpt = torch.load(ckpt_path, map_location='cpu')
    encoder.load_state_dict(ckpt['encoder'])
    probe.load_state_dict(ckpt['probe'])
    head.load_state_dict(ckpt['head'])
    encoder.to(device).eval()
    probe.to(device).eval()
    head.to(device).eval()
    return encoder, probe, head


# --------------------------------------------------------------------------
# Per-volume patch occlusion on a single target slice
# --------------------------------------------------------------------------

@torch.no_grad()
def patch_occlusion_volume(encoder, probe, head, F_v, slice_pixels, target_idx, device):
    """Compute per-patch attribution on one target slice of one volume.

    Args:
        encoder: the fine-tuned ViT (eval mode).
        probe: the slice-aggregation probe (eval mode).
        head: the LinearHead (eval mode).
        F_v: (S, D) tensor, cached per-slice features for this volume.
        slice_pixels: (3, H, W) tensor, raw pixel values of the target slice (in [0, 1]).
        target_idx: int, subset slice index (0..S-1) being attributed.
        device: torch device.

    Returns:
        patch_contrib: np.ndarray (P,) float32 — baseline - logit_masked_p for p in 0..P-1.
        baseline_logit: float — logit when slice's mean is reconstructed from all patches.
    """
    S, D = F_v.shape

    # 1) Re-forward the single target slice, preserving per-patch tokens
    x = slice_pixels.unsqueeze(0).to(device)        # (1, 3, H, W)
    with autocast():
        out = encoder(imagenet_normalize(x))         # (1, P, D)
    patches = out.squeeze(0).float()                 # (P, D)
    P = patches.size(0)

    # 2) Baseline slice token = mean over all 256 patches (matches what the
    #    model computed during the forward pass). Note this is re-computed
    #    here in fp32 for numerical consistency with the masked forwards.
    total = patches.sum(0)                           # (D,)
    baseline_slice = total / P                       # (D,)

    # 3) Build (P+1, S, D): row 0 = baseline, rows 1..P = each patch removed
    F_v_dev = F_v.float().to(device)                 # (S, D)
    F_stack = F_v_dev.unsqueeze(0).expand(P + 1, -1, -1).clone()  # (P+1, S, D)
    F_stack[0, target_idx] = baseline_slice

    # For each p: remove patch p from the mean
    # alt[p] = (total - patches[p]) / (P - 1)
    # Vectorised: alt = (total.unsqueeze(0) - patches) / (P - 1)   shape (P, D)
    alt_slices = (total.unsqueeze(0) - patches) / (P - 1)          # (P, D)
    F_stack[1:, target_idx] = alt_slices

    # 4) Forward through probe + head in one batch
    with autocast():
        pooled = probe(F_stack)                      # (P+1, D)
        logits = head(pooled).squeeze(-1)            # (P+1,)
    logits = logits.float().cpu().numpy()

    baseline_logit = float(logits[0])
    masked_logits = logits[1:]                       # (P,)
    patch_contrib = (baseline_logit - masked_logits).astype(np.float32)

    return patch_contrib, baseline_logit


# --------------------------------------------------------------------------
# Main loop: one pass per model
# --------------------------------------------------------------------------

def process_model(m, args, test_dataset, device):
    name = m['name']
    print(f'\n=== {name} ===', flush=True)

    feat_path = os.path.join(args.features_dir, f'features_{name}.npz')
    cached = np.load(feat_path)
    F_cache = cached['features']                     # (N, S, D) fp16
    labels = cached['labels']
    probs = cached['probs']
    N, S, D = F_cache.shape
    assert S == args.num_slices, (S, args.num_slices)

    ckpt_path = os.path.join(args.model_dir, name, 'best_model.pt')
    print(f'  loading {ckpt_path}')
    encoder, probe, head = load_model(
        ckpt_path, m['probe_type'], m['probe_kwargs'],
        args.num_slices, device,
    )

    # Pre-allocate per-target-slice output
    P = 256
    patch_contribs = {s: np.zeros((N, P), dtype=np.float32)
                      for s in args.target_slices}
    baselines = {s: np.zeros(N, dtype=np.float32)
                 for s in args.target_slices}

    loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                        num_workers=2, pin_memory=True)

    t0 = time.time()
    for i, (volume, _label) in enumerate(loader):
        volume = volume.squeeze(0)                   # (S, 3, H, W)
        F_v = torch.from_numpy(F_cache[i])           # (S, D) fp16

        for s in args.target_slices:
            pc, bl = patch_occlusion_volume(
                encoder, probe, head,
                F_v, volume[s], s, device,
            )
            patch_contribs[s][i] = pc
            baselines[s][i] = bl

        if (i + 1) % 200 == 0:
            dt = time.time() - t0
            print(f'  {i+1}/{N}  ({dt:.0f}s, {(i+1)/dt:.2f} vol/s)',
                  flush=True)

    # Save per-slice outputs
    for s in args.target_slices:
        out = os.path.join(
            args.output_dir, f'patch_aggregate_{name}_slice{s:02d}.npz',
        )
        np.savez(
            out,
            patch_contrib=patch_contribs[s],
            baseline=baselines[s],
            labels=labels,
            probs=probs,
            target_slice_idx=np.int32(s),
        )
        print(f'  wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)',
              flush=True)

    # Free memory before next model
    del encoder, probe, head, F_cache, cached
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# Aggregate figure
# --------------------------------------------------------------------------

def make_figure(output_dir, model_names, target_slices, th=0.5):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    displays = {
        'meanpool':  'FT + MeanPool',
        'crossattn': 'FT + CrossAttnPool',
        'd1':        'FT + AttentiveProbe d=1',
    }
    axial_of = lambda s: int(np.linspace(0, 199, 64)[s])

    # Layout: 3 rows (one per slice, but we only have 2 slices, doubled into
    # columns instead — easier to read as rows=slices, 3 columns per model).
    # Use: 2 target slices × 3 models rows = 6 rows. Each row shows
    #   [B-scan-less heatmap glaucoma | healthy | diff]
    n_rows = len(target_slices) * len(model_names)
    fig, axes = plt.subplots(n_rows, 3, figsize=(11, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    row = 0
    for s in target_slices:
        for m in model_names:
            d = np.load(
                os.path.join(output_dir, f'patch_aggregate_{m}_slice{s:02d}.npz'),
            )
            P = d['patch_contrib']                   # (N, 256)
            y = d['labels']
            # Per-class means
            glau_mean = P[y == 1].mean(axis=0).reshape(16, 16)
            heal_mean = P[y == 0].mean(axis=0).reshape(16, 16)
            diff = glau_mean - heal_mean
            vmax = max(np.abs(glau_mean).max(),
                       np.abs(heal_mean).max(),
                       np.abs(diff).max()) + 1e-12

            titles = [
                f'glaucoma mean (n={(y==1).sum()})',
                f'healthy mean  (n={(y==0).sum()})',
                'diff = glaucoma − healthy',
            ]
            for col, arr, tt in zip(range(3),
                                     [glau_mean, heal_mean, diff],
                                     titles):
                ax = axes[row, col]
                im = ax.imshow(arr, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                               interpolation='nearest')
                ax.set_title(f'{displays[m]}  | subset {s} (native ~{axial_of(s)}/199)\n{tt}',
                             fontsize=9)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            row += 1

    fig.suptitle('Aggregate per-patch attribution across 3000 Test volumes\n'
                 '(occlusion Δlogit per 16x16 patch, averaged per class)',
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(output_dir, 'patch_aggregate_figure.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')


# --------------------------------------------------------------------------
# Entry
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir',     required=True)
    ap.add_argument('--model-dir',    required=True,
                    help='Dir with <name>/best_model.pt for each of 3 FT models')
    ap.add_argument('--features-dir', required=True,
                    help='Dir with features_<name>.npz from plucky_soccer')
    ap.add_argument('--output-dir',   required=True)
    ap.add_argument('--num-slices',   type=int, default=64)
    ap.add_argument('--slice-size',   type=int, default=256)
    ap.add_argument('--target-slices', type=int, nargs='+', default=[20, 43],
                    help='Subset indices to aggregate patch attribution on '
                         '(default: peak and secondary peak of the cross-model '
                         'contribution curve — native positions ~63 and ~137)')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=== patch_aggregate.py ===')
    print(f'  device:         {device}')
    print(f'  data:           {args.data_dir}')
    print(f'  models:         {args.model_dir}')
    print(f'  features:       {args.features_dir}')
    print(f'  output:         {args.output_dir}')
    print(f'  target_slices:  {args.target_slices}')

    test_dataset = OCTVolumeDataset(
        os.path.join(args.data_dir, 'Test'),
        num_slices=args.num_slices, slice_size=args.slice_size,
        return_label=True,
    )
    print(f'  Test volumes:   {len(test_dataset)}')

    # Sequential per model (matches never-queue + RAM-safe rule)
    for m in MODELS:
        process_model(m, args, test_dataset, device)

    make_figure(args.output_dir,
                [m['name'] for m in MODELS],
                args.target_slices)

    print('\n=== done ===')


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    main()
