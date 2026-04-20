#!/usr/bin/env python
"""Slice- and patch-level occlusion attribution for the 3 fine-tune runs.

Runs four phases sequentially in one process:

  Phase 1 — Feature extraction (per model, streaming):
    Load best_model.pt for each of the 3 FT runs. Encode all N test volumes,
    caching per-slice features (N, S, 768) as float16. Also run probe+head
    forward at full sequence to recompute predictions (so we don't need to
    download test_predictions.npz separately).

  Phase 2 — Slice occlusion attribution (per model, feature-level):
    For each volume and each slice s in 0..S-1, zero X[s] and re-run
    probe+head to get the altered logit. contribution[s] = logit_full −
    logit_masked_s. Architecture-agnostic; correct for MeanPool,
    CrossAttnPool, and AttentiveProbe d=1. Batched 1 volume at a time as
    (S, S, D).

  Phase 3 — Patch occlusion heatmaps (per model, 20 selected volumes):
    Select 10 confident true-positive + 10 confident true-negative volumes.
    Identify top-3 slices by |contribution|. Re-forward those slices through
    the (fine-tuned) encoder with per-patch tokens preserved (no patch
    mean-pool). For each of 256 patches, zero it, recompute the slice-token
    mean, substitute into the cached (S, D) feature tensor, run probe+head,
    delta logit → per-patch contribution. Reshape to (16, 16) and save with
    overlay on the original slice image.

  Phase 4 — Plots:
    - slice_contribution_curves.png: 3 FT architectures overlaid, disease vs
      healthy mean contribution per slice index.
    - heatmap grids: one PNG per model, 20 vols × 3 slices each with the
      overlay + original slice side by side.

All phases stream, cache float16 where possible, and explicitly release
model state between models to stay within ~4 GB RAM / 2 GB VRAM.

Invocation:
    python scripts/interpretability.py \\
        --data-dir /tmp/fairvision_data/data \\
        --model-dir /tmp/models \\
        --output-dir /tmp/interp_out \\
        --num-slices 64 \\
        --slice-size 256 \\
        --chunk-size 16
"""

import argparse
import gc
import json
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
# AttentiveProbe, LinearHead, imagenet_normalize all live in eval_downstream
# to avoid code duplication.
from src.eval_downstream import AttentiveProbe, LinearHead, imagenet_normalize


# --------------------------------------------------------------------------
# Model definitions (mirrors each run's config so state_dict loads cleanly)
# --------------------------------------------------------------------------

MODELS = [
    {
        'name':        'meanpool',
        'probe_type':  'mean_pool',
        'probe_kwargs': {},
        'blob_path':   'ijepa-downstream/downstream_patch_s64_ep50_bs1_linear_20260419_060137',
        'display':     'FT + MeanPool',
    },
    {
        'name':        'crossattn',
        'probe_type':  'cross_attn_pool',
        'probe_kwargs': {'head_dim': 64},
        'blob_path':   'ijepa-downstream/downstream_patch_s64_ep50_bs1_linear_20260418_192249',
        'display':     'FT + CrossAttnPool',
    },
    {
        'name':        'd1',
        'probe_type':  'attentive',
        'probe_kwargs': {'num_heads': 12, 'depth': 1},
        'blob_path':   'ijepa-downstream/downstream_patch_s64_ep50_bs1_linear_20260418_035940',
        'display':     'FT + AttentiveProbe d=1',
    },
]


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
    """Load encoder, probe, head from a saved best_model.pt."""
    vit_cfg = _VIT_CONFIGS['vit_base']
    encoder = VisionTransformer(
        img_size=256, patch_size=16,
        embed_dim=vit_cfg['embed_dim'],
        depth=vit_cfg['depth'],
        num_heads=vit_cfg['num_heads'],
    )
    probe = build_probe(probe_type, num_slices, vit_cfg['embed_dim'], probe_kwargs)
    head = LinearHead(in_dim=vit_cfg['embed_dim'])

    ckpt = torch.load(ckpt_path, map_location='cpu')
    encoder.load_state_dict(ckpt['encoder'])
    probe.load_state_dict(ckpt['probe'])
    head.load_state_dict(ckpt['head'])

    encoder.to(device).eval()
    probe.to(device).eval()
    head.to(device).eval()

    return encoder, probe, head, ckpt.get('val_auc', None)


# --------------------------------------------------------------------------
# Phase 1 — feature extraction
# --------------------------------------------------------------------------

@torch.no_grad()
def encode_volume_per_slice(encoder, volume, chunk_size, device):
    """volume: (S, 3, H, W) → (S, D) fp16 CPU."""
    flat = volume.to(device)
    parts = []
    for j in range(0, flat.size(0), chunk_size):
        chunk = imagenet_normalize(flat[j:j + chunk_size])
        with autocast():
            out = encoder(chunk)                    # (chunk, 256, D)
        parts.append(out.mean(dim=1).to(torch.float16).cpu())
    return torch.cat(parts, dim=0)                  # (S, D)


@torch.no_grad()
def phase1_extract(model_cfg, dataset, num_slices, embed_dim, chunk_size,
                   model_dir, output_dir, device):
    """Encode all test volumes and compute predictions for one model.

    Writes features_<name>.npz with:
      features: (N, S, D) float16
      labels:   (N,)      int8
      probs:    (N,)      float32   (predicted sigmoid)
      logits:   (N,)      float32
    """
    name = model_cfg['name']
    ckpt_path = os.path.join(model_dir, name, 'best_model.pt')
    print(f'[phase1/{name}] loading {ckpt_path}')
    encoder, probe, head, saved_val = load_model(
        ckpt_path, model_cfg['probe_type'], model_cfg['probe_kwargs'],
        num_slices, device,
    )

    n_vols = len(dataset)
    features = torch.zeros(n_vols, num_slices, embed_dim, dtype=torch.float16)
    labels   = torch.zeros(n_vols, dtype=torch.int8)
    logits   = torch.zeros(n_vols, dtype=torch.float32)

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=2, pin_memory=True)

    t0 = time.time()
    for i, (volume, label) in enumerate(loader):
        volume = volume.squeeze(0)                  # (S, 3, H, W)
        per_slice = encode_volume_per_slice(
            encoder, volume, chunk_size, device)    # (S, D) fp16

        # Predict
        x = per_slice.float().unsqueeze(0).to(device)   # (1, S, D)
        with autocast():
            pooled = probe(x)                           # (1, D)
            logit = head(pooled).squeeze()              # scalar
        logits[i] = logit.item()
        features[i] = per_slice
        labels[i] = label.item()

        if (i + 1) % 200 == 0:
            dt = time.time() - t0
            print(f'[phase1/{name}]   {i+1}/{n_vols}  ({dt:.0f}s, '
                  f'{(i+1)/dt:.1f} vol/s)', flush=True)

    probs = torch.sigmoid(logits).numpy()

    # Sanity: predicted Test AUC should match the reported number
    from sklearn.metrics import roc_auc_score
    test_auc = roc_auc_score(labels.numpy(), probs)
    sv_str = f'{saved_val:.4f}' if saved_val is not None else 'N/A'
    print(f'[phase1/{name}] recomputed test_auc = {test_auc:.4f} '
          f'(saved val_auc = {sv_str})')

    out_path = os.path.join(output_dir, f'features_{name}.npz')
    np.savez(out_path,
             features=features.numpy(), labels=labels.numpy(),
             probs=probs, logits=logits.numpy(),
             test_auc=np.float32(test_auc))
    print(f'[phase1/{name}] saved {out_path} '
          f'({os.path.getsize(out_path)/1e6:.1f} MB)')

    # Keep encoder around for Phase 3 of THIS model, but free when caller
    # moves to the next model.
    return encoder, probe, head


# --------------------------------------------------------------------------
# Phase 2 — slice occlusion attribution
# --------------------------------------------------------------------------

@torch.no_grad()
def phase2_slice_occlusion(model_cfg, features_path, probe, head, device,
                           output_dir):
    """For each volume, zero each slice one at a time, compute delta logit.

    Batches the S masked variants per volume into a single forward:
        variants ∈ (S, S, D), variants[s, s] = 0, else features.
    Head + probe run on all S variants in one call.
    """
    name = model_cfg['name']
    data = np.load(features_path)
    features = data['features']                     # (N, S, D) float16
    labels   = data['labels']
    logits_full = data['logits']                    # (N,) precomputed full-model logits

    N, S, D = features.shape
    contrib = np.zeros((N, S), dtype=np.float32)
    # Mask matrix: row 0 = baseline (no occlusion). Rows 1..S zero slice s=r-1.
    # Doing baseline + masked in one batch keeps the two numerically consistent
    # under autocast / fp16 rounding.
    mask = np.ones((S + 1, S), dtype=np.float32)
    for r in range(1, S + 1):
        mask[r, r - 1] = 0.0

    t0 = time.time()
    for i in range(N):
        F = features[i].astype(np.float32)                          # (S, D)
        variants = F[None, :, :] * mask[:, :, None]                 # (S+1, S, D)
        x = torch.from_numpy(variants).to(device)
        with autocast():
            pooled = probe(x)                                       # (S+1, D)
            logits_all = head(pooled).squeeze(-1)                   # (S+1,)
        logits_all = logits_all.float().cpu().numpy()
        baseline = logits_all[0]
        masked_logit = logits_all[1:]                               # (S,)

        contrib[i] = baseline - masked_logit                        # (S,)

        if (i + 1) % 500 == 0:
            dt = time.time() - t0
            print(f'[phase2/{name}]   {i+1}/{N}  ({dt:.0f}s)', flush=True)

    # Aggregate per class
    mean_pos = contrib[labels == 1].mean(axis=0)
    mean_neg = contrib[labels == 0].mean(axis=0)
    out_path = os.path.join(output_dir, f'slice_contrib_{name}.npz')
    np.savez(out_path,
             contrib=contrib, labels=labels,
             mean_pos=mean_pos.astype(np.float32),
             mean_neg=mean_neg.astype(np.float32))
    print(f'[phase2/{name}] saved {out_path}')
    return contrib


# --------------------------------------------------------------------------
# Phase 3 — patch occlusion heatmaps (subset)
# --------------------------------------------------------------------------

@torch.no_grad()
def phase3_patch_heatmaps(model_cfg, features_path, contrib, dataset,
                          encoder, probe, head, output_dir, device,
                          n_tp=10, n_tn=10, top_k_slices=3):
    """For selected volumes, compute per-patch contribution on top-k slices."""
    name = model_cfg['name']
    data = np.load(features_path)
    labels = data['labels']
    probs = data['probs']

    tp_idx = np.where((labels == 1) & (probs > 0.7))[0]
    tn_idx = np.where((labels == 0) & (probs < 0.3))[0]
    np.random.RandomState(42).shuffle(tp_idx)
    np.random.RandomState(42).shuffle(tn_idx)
    picks = list(tp_idx[:n_tp]) + list(tn_idx[:n_tn])
    print(f'[phase3/{name}] selected {len(picks)} volumes '
          f'({len(tp_idx[:n_tp])} TP, {len(tn_idx[:n_tn])} TN)')

    heatmap_dir = os.path.join(output_dir, f'heatmaps_{name}')
    os.makedirs(heatmap_dir, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t0 = time.time()
    for k, vol_idx in enumerate(picks):
        # Cached per-slice features for this volume (altered slice will
        # substitute into this).
        F = data['features'][vol_idx].astype(np.float32)        # (S, D)

        # Load raw pixel volume to re-forward selected slices
        volume, label = dataset[vol_idx]                         # (S, 3, H, W), scalar
        volume = volume.to(device)

        top_slices = np.argsort(np.abs(contrib[vol_idx]))[-top_k_slices:]
        top_slices = sorted(top_slices.tolist())

        # Re-forward the selected slices with per-patch tokens preserved
        selected = imagenet_normalize(volume[top_slices])        # (K, 3, H, W)
        with autocast():
            patch_tokens = encoder(selected)                     # (K, 256, D)
        patch_tokens = patch_tokens.float()                      # (K, 256, D)

        for k_idx, s in enumerate(top_slices):
            patches = patch_tokens[k_idx]                         # (256, D)
            P = patches.shape[0]
            total = patches.sum(dim=0)                             # (D,)
            alt_slice = (total.unsqueeze(0) - patches) / (P - 1)   # (P, D)

            # Build (P, S, D) with slice s replaced by each alternative
            template = torch.from_numpy(F).to(device)              # (S, D)
            template = template.unsqueeze(0).expand(P, -1, -1).clone()  # (P, S, D)
            template[:, s, :] = alt_slice

            with autocast():
                pooled = probe(template)                            # (P, D)
                alt_logits = head(pooled).squeeze(-1)              # (P,)
            alt_logits = alt_logits.float().cpu().numpy()

            # contribution of patch p = full_logit − logit_with_p_removed
            patch_contrib = data['logits'][vol_idx] - alt_logits    # (256,)
            heatmap = patch_contrib.reshape(16, 16)

            # Save figure: original slice + heatmap overlay, side-by-side
            slice_img = volume[s].cpu().numpy()                     # (3, H, W) in [0,1]
            slice_img = np.transpose(slice_img, (1, 2, 0))          # (H, W, 3)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(slice_img, cmap='gray')
            axes[0].set_title(f'Slice {s}')
            axes[0].axis('off')

            # Upsample heatmap from (16,16) to (256,256) for overlay
            from scipy.ndimage import zoom
            heat_up = zoom(heatmap, 16, order=1)                    # (256, 256)
            vmax = np.abs(heatmap).max() + 1e-9
            axes[1].imshow(slice_img, cmap='gray', alpha=0.6)
            axes[1].imshow(heat_up, cmap='RdBu_r', alpha=0.5,
                           vmin=-vmax, vmax=vmax)
            axes[1].set_title(
                f'Patch contrib (Δlogit)\n'
                f'label={int(labels[vol_idx])}  pred={probs[vol_idx]:.2f}  '
                f'slice_contrib={contrib[vol_idx, s]:+.3f}'
            )
            axes[1].axis('off')

            out = os.path.join(
                heatmap_dir, f'vol{vol_idx:04d}_slice{s:02d}.png')
            plt.tight_layout()
            fig.savefig(out, dpi=100, bbox_inches='tight')
            plt.close(fig)

        if (k + 1) % 5 == 0:
            dt = time.time() - t0
            print(f'[phase3/{name}]   {k+1}/{len(picks)}  ({dt:.0f}s)',
                  flush=True)

    print(f'[phase3/{name}] heatmaps written to {heatmap_dir}')


# --------------------------------------------------------------------------
# Phase 4 — plots (cross-architecture)
# --------------------------------------------------------------------------

def phase4_plots(output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {'meanpool': '#d62728', 'crossattn': '#1f77b4', 'd1': '#2ca02c'}
    displays = {'meanpool': 'FT + MeanPool',
                'crossattn': 'FT + CrossAttnPool',
                'd1': 'FT + AttentiveProbe d=1'}
    for name in ['meanpool', 'crossattn', 'd1']:
        path = os.path.join(output_dir, f'slice_contrib_{name}.npz')
        if not os.path.exists(path):
            print(f'[phase4] skip {name}: {path} missing')
            continue
        d = np.load(path)
        x = np.arange(len(d['mean_pos']))
        ax.plot(x, d['mean_pos'], color=colors[name],
                linestyle='-',  linewidth=2.0,
                label=f'{displays[name]} (glaucoma, n={int((d["labels"]==1).sum())})')
        ax.plot(x, d['mean_neg'], color=colors[name],
                linestyle='--', linewidth=1.2, alpha=0.6,
                label=f'{displays[name]} (healthy,  n={int((d["labels"]==0).sum())})')

    ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Slice index (0..63, linspace(0, 199, 64))')
    ax.set_ylabel('Mean Δlogit contribution per slice (occlusion)')
    ax.set_title('Slice-level occlusion attribution — 3 fine-tune probes\n'
                 '(how much does each slice push the logit toward its class?)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out = os.path.join(output_dir, 'slice_contribution_curves.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[phase4] saved {out}')


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir',   required=True)
    ap.add_argument('--model-dir',  required=True,
                    help='Directory with <name>/best_model.pt for each of 3 models')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--num-slices', type=int, default=64)
    ap.add_argument('--slice-size', type=int, default=256)
    ap.add_argument('--chunk-size', type=int, default=16)
    ap.add_argument('--n-tp', type=int, default=10)
    ap.add_argument('--n-tn', type=int, default=10)
    ap.add_argument('--top-k-slices', type=int, default=3)
    ap.add_argument('--skip-existing', action='store_true',
                    help='Skip Phase 1 for models whose features_*.npz already exists')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'=== Interpretability run ===')
    print(f'  device:      {device}')
    print(f'  data:        {args.data_dir}')
    print(f'  models:      {args.model_dir}')
    print(f'  output:      {args.output_dir}')
    print(f'  num_slices:  {args.num_slices}')
    print()

    test_dataset = OCTVolumeDataset(
        os.path.join(args.data_dir, 'Test'),
        num_slices=args.num_slices, slice_size=args.slice_size,
        return_label=True,
    )
    print(f'Test dataset: {len(test_dataset)} volumes')
    print()

    vit_cfg = _VIT_CONFIGS['vit_base']

    for model_cfg in MODELS:
        name = model_cfg['name']
        feat_path = os.path.join(args.output_dir, f'features_{name}.npz')
        slice_path = os.path.join(args.output_dir, f'slice_contrib_{name}.npz')

        # Phase 1
        if args.skip_existing and os.path.exists(feat_path) and os.path.exists(slice_path):
            print(f'=== {name}: Phase 1 + 2 cache exists, skip → Phase 3 ===')
            # Still need encoder/probe/head for Phase 3
            ckpt_path = os.path.join(args.model_dir, name, 'best_model.pt')
            encoder, probe, head, _ = load_model(
                ckpt_path, model_cfg['probe_type'], model_cfg['probe_kwargs'],
                args.num_slices, device,
            )
            contrib = np.load(slice_path)['contrib']
        else:
            print(f'=== {name}: Phase 1 (encode) ===')
            encoder, probe, head = phase1_extract(
                model_cfg, test_dataset, args.num_slices,
                vit_cfg['embed_dim'], args.chunk_size,
                args.model_dir, args.output_dir, device,
            )

            print(f'=== {name}: Phase 2 (slice occlusion) ===')
            contrib = phase2_slice_occlusion(
                model_cfg, feat_path, probe, head, device, args.output_dir,
            )

        print(f'=== {name}: Phase 3 (patch occlusion heatmaps) ===')
        phase3_patch_heatmaps(
            model_cfg, feat_path, contrib, test_dataset,
            encoder, probe, head, args.output_dir, device,
            n_tp=args.n_tp, n_tn=args.n_tn,
            top_k_slices=args.top_k_slices,
        )

        # Free before next model
        del encoder, probe, head, contrib
        gc.collect()
        torch.cuda.empty_cache()

    # Phase 4
    print('=== Phase 4: cross-model plots ===')
    phase4_plots(args.output_dir)

    # Summary
    summary = {'models': {}}
    for m in MODELS:
        name = m['name']
        path = os.path.join(args.output_dir, f'features_{name}.npz')
        if os.path.exists(path):
            d = np.load(path)
            summary['models'][name] = {
                'test_auc': float(d['test_auc']) if 'test_auc' in d else None,
                'n_volumes': int(len(d['labels'])),
                'displays': m['display'],
            }
    with open(os.path.join(args.output_dir, 'interpretability_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    main()
