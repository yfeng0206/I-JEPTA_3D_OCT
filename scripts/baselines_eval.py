#!/usr/bin/env python
"""Frozen-probe evaluation of foundation-model baselines on FairVision.

Sequential pipeline (RAM-safe, per never-queue rule inside one process):

  for encoder in [dinov3, octcube]:
      1. Build adapter, load weights.
      2. Pre-compute features for Train/Val/Test, cache to disk as fp16.
      3. Free encoder (del, gc, torch.cuda.empty_cache()).
      4. Train a frozen probe + LinearHead on cached features:
           - per_slice_2d encoders: CrossAttnPool(S=num_slices, D=embed_dim) + Linear
           - volume_3d encoders:    Identity + Linear (features already pooled)
      5. Evaluate on Test split. Save results.json + test_predictions.npz.
      6. Delete caches + probe, next encoder.

Output directory layout:

    <output_dir>/
        <encoder_name>/
            features_train.npz
            features_val.npz
            features_test.npz
            results.json
            test_predictions.npz
            train_log.csv
            best_probe.pt

At the end, a summary.csv collecting all encoder test_auc values is written
so the comparison is in one place.
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
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.encoders import build_adapter
from src.models.attentive_pool_minimal import CrossAttnPool
from src.datasets.oct_volumes import OCTVolumeDataset
from src.eval_downstream import LinearHead


# --------------------------------------------------------------------------
# Feature pre-computation
# --------------------------------------------------------------------------

@torch.no_grad()
def precompute_features(adapter, dataset, num_slices, embed_dim, layout):
    """Encode every volume in the split through `adapter`. Streams one
    volume at a time to keep RAM bounded."""
    N = len(dataset)
    # Output shape depends on layout
    if layout == 'per_slice_2d':
        features = np.zeros((N, num_slices, embed_dim), dtype=np.float16)
    else:
        features = np.zeros((N, embed_dim), dtype=np.float16)
    labels = np.zeros(N, dtype=np.int8)

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=2, pin_memory=True)
    t0 = time.time()
    for i, (volume, label) in enumerate(loader):
        volume = volume.squeeze(0)              # (S, 3, H, W)
        out = adapter.encode_volume(volume)      # (S, D) or (D,)
        features[i] = out.numpy()
        labels[i] = int(label.item())
        if (i + 1) % 500 == 0:
            dt = time.time() - t0
            print(f'    {i+1}/{N}  ({dt:.0f}s, {(i+1)/dt:.2f} vol/s)',
                  flush=True)
    return features, labels


# --------------------------------------------------------------------------
# Probe training on cached features
# --------------------------------------------------------------------------

def train_probe(train_feats, train_labels, val_feats, val_labels,
                num_slices, embed_dim, layout, device,
                epochs=50, patience=15, lr=4e-4, weight_decay=0.05,
                batch_size=128, warmup_epochs=5):
    """Train CrossAttnPool+Linear (per-slice) or Linear (volume) on
    cached features. Standard frozen-probe schedule matching our other runs."""
    if layout == 'per_slice_2d':
        probe = CrossAttnPool(num_slices=num_slices, embed_dim=embed_dim,
                              head_dim=64).to(device)
        probe_desc = f'CrossAttnPool(S={num_slices}, D={embed_dim})'
    else:
        probe = nn.Identity().to(device)
        probe_desc = 'Identity (volume-level features)'
    head = LinearHead(in_dim=embed_dim).to(device)

    print(f'    probe: {probe_desc}   head: LinearHead({embed_dim})')

    # Wrap features in TensorDatasets for standard training
    train_ds = TensorDataset(
        torch.from_numpy(train_feats).float(),
        torch.from_numpy(train_labels).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(val_feats).float(),
        torch.from_numpy(val_labels).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    trainable = list(probe.parameters()) + list(head.parameters())
    trainable = [p for p in trainable if p.requires_grad]
    if not trainable:  # Identity probe
        trainable = list(head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = max(1, warmup_epochs * len(train_loader))
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        import math
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * prog)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    best_auc = 0.0
    best_state = None
    best_epoch = 0
    patience_cnt = 0
    log_rows = []

    for epoch in range(1, epochs + 1):
        probe.train(); head.train()
        tot_loss, n = 0.0, 0
        for feats, lbls in train_loader:
            feats = feats.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            with autocast():
                pooled = probe(feats)                    # (B, D)
                logits = head(pooled).squeeze(-1)        # (B,)
                loss = criterion(logits, lbls)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            tot_loss += loss.item() * lbls.size(0)
            n += lbls.size(0)
        train_loss = tot_loss / max(n, 1)

        probe.eval(); head.eval()
        all_lbl, all_prob = [], []
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for feats, lbls in val_loader:
                feats = feats.to(device, non_blocking=True)
                lbls = lbls.to(device, non_blocking=True)
                with autocast():
                    pooled = probe(feats)
                    logits = head(pooled).squeeze(-1)
                    loss = criterion(logits, lbls)
                val_loss_sum += loss.item() * lbls.size(0)
                val_n += lbls.size(0)
                all_lbl.append(lbls.cpu().numpy())
                all_prob.append(torch.sigmoid(logits).float().cpu().numpy())
        val_loss = val_loss_sum / max(val_n, 1)
        all_lbl = np.concatenate(all_lbl); all_prob = np.concatenate(all_prob)
        val_auc = roc_auc_score(all_lbl, all_prob) if len(np.unique(all_lbl)) >= 2 else 0.5

        improved = val_auc > best_auc
        marker = ' *' if improved else ''
        print(f'    Epoch {epoch:2d}/{epochs}  '
              f'Train: {train_loss:.4f}  Val: {val_loss:.4f}  AUC: {val_auc:.4f}{marker}',
              flush=True)
        log_rows.append({'epoch': epoch, 'train_loss': train_loss,
                         'val_loss': val_loss, 'val_auc': val_auc})
        if improved:
            best_auc = val_auc
            best_state = {'probe': probe.state_dict(),
                          'head': head.state_dict()}
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1
            if epoch > warmup_epochs and patience_cnt >= patience:
                print(f'    Early stopping at epoch {epoch} (patience={patience})')
                break

    # Restore best state for test eval
    if best_state is not None:
        probe.load_state_dict(best_state['probe'])
        head.load_state_dict(best_state['head'])

    return probe, head, best_epoch, best_auc, log_rows


@torch.no_grad()
def evaluate_test(probe, head, test_feats, test_labels, device, batch_size=128):
    probe.eval(); head.eval()
    ds = TensorDataset(torch.from_numpy(test_feats).float(),
                       torch.from_numpy(test_labels).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
    all_lbl, all_prob = [], []
    for feats, lbls in loader:
        feats = feats.to(device, non_blocking=True)
        with autocast():
            pooled = probe(feats)
            logits = head(pooled).squeeze(-1)
        all_lbl.append(lbls.numpy())
        all_prob.append(torch.sigmoid(logits).float().cpu().numpy())
    all_lbl = np.concatenate(all_lbl); all_prob = np.concatenate(all_prob)
    auc = roc_auc_score(all_lbl, all_prob)
    pred = (all_prob >= 0.5).astype(np.int64)
    tp = ((all_lbl == 1) & (pred == 1)).sum()
    fn = ((all_lbl == 1) & (pred == 0)).sum()
    tn = ((all_lbl == 0) & (pred == 0)).sum()
    fp = ((all_lbl == 0) & (pred == 1)).sum()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {
        'test_auc': float(auc),
        'sensitivity': float(sens),
        'specificity': float(spec),
        'confusion': {'TP': int(tp), 'FN': int(fn), 'TN': int(tn), 'FP': int(fp)},
    }, all_lbl, all_prob


# --------------------------------------------------------------------------
# Per-encoder workflow
# --------------------------------------------------------------------------

def run_encoder(encoder_name, args, datasets, device):
    print(f'\n{"="*70}\n=== Encoder: {encoder_name} ===\n{"="*70}', flush=True)

    out_dir = os.path.join(args.output_dir, encoder_name)
    os.makedirs(out_dir, exist_ok=True)

    adapter_kwargs = {
        'device': device,
        'weights_path': args.encoder_weights.get(encoder_name),
    }
    adapter = build_adapter(encoder_name, **adapter_kwargs)
    print(f'  built: {adapter}')

    # ---- Phase 1: pre-compute features for Train / Val / Test ----
    feats = {}
    for split, ds in datasets.items():
        cache_path = os.path.join(out_dir, f'features_{split}.npz')
        if os.path.exists(cache_path) and not args.force:
            print(f'  [cache hit] {cache_path}')
            data = np.load(cache_path)
            feats[split] = (data['features'], data['labels'])
            continue
        print(f'  encoding {split} ({len(ds)} volumes)...')
        t0 = time.time()
        f, y = precompute_features(
            adapter, ds, args.num_slices, adapter.embed_dim, adapter.input_layout,
        )
        np.savez(cache_path, features=f, labels=y)
        print(f'  wrote {cache_path} ({os.path.getsize(cache_path)/1e6:.0f} MB, '
              f'{time.time()-t0:.0f}s)')
        feats[split] = (f, y)

    # ---- Phase 2: free encoder before probe training ----
    embed_dim = adapter.embed_dim
    layout = adapter.input_layout
    adapter.cleanup()
    del adapter
    gc.collect()
    torch.cuda.empty_cache()
    print(f'  encoder freed; training probe on cached features')

    # ---- Phase 3: train probe + head on cached features ----
    train_f, train_y = feats['train']
    val_f, val_y = feats['val']
    test_f, test_y = feats['test']
    probe, head, best_epoch, best_val_auc, log_rows = train_probe(
        train_f, train_y, val_f, val_y,
        num_slices=args.num_slices, embed_dim=embed_dim, layout=layout,
        device=device,
        epochs=args.epochs, patience=args.patience,
        lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, warmup_epochs=args.warmup_epochs,
    )

    # ---- Phase 4: test eval ----
    test_metrics, test_lbl, test_prob = evaluate_test(
        probe, head, test_f, test_y, device, batch_size=args.batch_size,
    )
    print(f'  Test AUC: {test_metrics["test_auc"]:.4f}  '
          f'Sens: {test_metrics["sensitivity"]:.3f}  '
          f'Spec: {test_metrics["specificity"]:.3f}')

    # ---- Save outputs ----
    torch.save({
        'probe': probe.state_dict(),
        'head': head.state_dict(),
        'best_epoch': best_epoch,
        'best_val_auc': best_val_auc,
        'encoder': encoder_name,
        'embed_dim': embed_dim,
        'layout': layout,
    }, os.path.join(out_dir, 'best_probe.pt'))

    np.savez(os.path.join(out_dir, 'test_predictions.npz'),
             labels=test_lbl.astype(np.int8), probs=test_prob.astype(np.float32))

    import csv
    with open(os.path.join(out_dir, 'train_log.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss', 'val_auc'])
        w.writeheader()
        w.writerows(log_rows)

    results = {
        'encoder': encoder_name,
        'embed_dim': embed_dim,
        'layout': layout,
        'num_slices': args.num_slices,
        'best_epoch': best_epoch,
        'best_val_auc': float(best_val_auc),
        **test_metrics,
        'config': {
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'patience': args.patience,
            'warmup_epochs': args.warmup_epochs,
        },
    }
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  wrote {out_dir}/results.json')

    # ---- Cleanup ----
    del probe, head, feats, train_f, val_f, test_f
    gc.collect()
    torch.cuda.empty_cache()
    return results


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir',    required=True,
                    help='Root with Training/, Validation/, Test/ subdirs')
    ap.add_argument('--output-dir',  required=True)
    ap.add_argument('--encoders',    nargs='+', required=True,
                    help='Encoder names to evaluate, e.g. dinov3 octcube')
    ap.add_argument('--num-slices',  type=int, default=64)
    ap.add_argument('--slice-size',  type=int, default=256)
    # Per-encoder weights path (optional per encoder; some load from HF ID)
    ap.add_argument('--octcube-weights',  default=None)
    ap.add_argument('--dinov3-weights',   default=None)
    # Probe training hyperparameters
    ap.add_argument('--epochs',        type=int, default=50)
    ap.add_argument('--patience',      type=int, default=15)
    ap.add_argument('--lr',            type=float, default=4e-4)
    ap.add_argument('--weight-decay',  type=float, default=0.05)
    ap.add_argument('--batch-size',    type=int, default=128)
    ap.add_argument('--warmup-epochs', type=int, default=5)
    ap.add_argument('--force',         action='store_true',
                    help='Re-encode even if features cache exists')
    args = ap.parse_args()

    args.encoder_weights = {
        'octcube':         args.octcube_weights,
        'dinov3':          args.dinov3_weights,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=== baselines_eval.py ===')
    print(f'  device:      {device}')
    print(f'  data:        {args.data_dir}')
    print(f'  output:      {args.output_dir}')
    print(f'  encoders:    {args.encoders}')
    print(f'  num_slices:  {args.num_slices}')

    datasets = {
        'train': OCTVolumeDataset(os.path.join(args.data_dir, 'Training'),
                                  num_slices=args.num_slices,
                                  slice_size=args.slice_size,
                                  return_label=True),
        'val':   OCTVolumeDataset(os.path.join(args.data_dir, 'Validation'),
                                  num_slices=args.num_slices,
                                  slice_size=args.slice_size,
                                  return_label=True),
        'test':  OCTVolumeDataset(os.path.join(args.data_dir, 'Test'),
                                  num_slices=args.num_slices,
                                  slice_size=args.slice_size,
                                  return_label=True),
    }
    for k, ds in datasets.items():
        print(f'  {k}: {len(ds)} volumes')

    all_results = []
    for enc_name in args.encoders:
        try:
            r = run_encoder(enc_name, args, datasets, device)
            all_results.append(r)
        except Exception as e:
            print(f'\n[ERROR on {enc_name}] {type(e).__name__}: {e}\n'
                  f'    Continuing with next encoder.\n', flush=True)
            import traceback
            traceback.print_exc()

    # Summary
    print('\n=== SUMMARY ===')
    for r in all_results:
        print(f'  {r["encoder"]:20s}  Test AUC = {r["test_auc"]:.4f}  '
              f'(Sens {r["sensitivity"]:.3f}  Spec {r["specificity"]:.3f})')

    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nwrote {args.output_dir}/summary.json')


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    main()
