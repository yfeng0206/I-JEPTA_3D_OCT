"""
Downstream glaucoma classification using pretrained I-JEPA encoder.

Supports both patch-level and slice-level pretrained models:
  - Patch-level: each slice is encoded by the frozen ViT, mean-pooled to one
    token per slice, then a trainable attentive probe (single transformer
    block with learnable [CLS] token) aggregates across slices, followed by
    a linear classifier.  Follows the I-JEPA evaluation protocol (Assran
    et al., 2023).
  - Slice-level: slices are encoded by frozen ConvNeXt + frozen slice encoder,
    then mean-pooled and classified by a trainable MLP head.

Usage:
    # Patch-level pretrained -> AttentiveProbe + Linear
    python eval_downstream.py --config configs/downstream_patch.yaml

    # Slice-level pretrained -> MLP only
    python eval_downstream.py --config configs/downstream_slice.yaml

Compatible with PyTorch 1.13.1 and Python 3.8.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

import yaml

# ImageNet normalization (must match pretraining transforms in src/transforms.py)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def imagenet_normalize(x):
    """Normalize a batch of [0,1] images to ImageNet mean/std.

    Args:
        x: (B, 3, H, W) tensor in [0, 1] range.
    Returns:
        (B, 3, H, W) tensor normalized to ImageNet distribution.
    """
    mean = IMAGENET_MEAN.to(x.device, x.dtype)
    std = IMAGENET_STD.to(x.device, x.dtype)
    return (x - mean) / std

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from src.models.vision_transformer import (
    VisionTransformer, SliceEncoder, Block, VIT_EMBED_DIMS,
)
from src.models.feature_extractor import FrozenFeatureExtractor
from src.datasets.oct_volumes import OCTVolumeDataset
from src.helper import _VIT_CONFIGS
from src.utils.distributed import init_distributed


# ---------------------------------------------------------------------------
# Attentive probe for patch-level downstream (I-JEPA paper design)
# ---------------------------------------------------------------------------

class AttentiveProbe(nn.Module):
    """Slice-level attention probe for 3D OCT volume aggregation.

    Adapted from the I-JEPA attentive probe (Assran et al., 2023).  The
    paper uses a single block because patch tokens already carry global
    context from 12 encoder layers.  Our slice tokens are independently
    encoded, so we default to ``depth=2`` to give the model a chance to
    learn inter-slice relationships (configurable for ablation).

    Input:  (B, num_slices, embed_dim) -- one token per slice.
    Output: (B, embed_dim) -- volume representation from CLS token.

    Parameters (depth=2, dim=768):
        cls_token:  1 x 768          =       768
        pos_embed:  101 x 768        =    77,568
        2 x Block (SA + MLP):      ~14,175,744
        final norm:                      1,536
        Total:                     ~14,255,616
    """

    def __init__(self, num_slices=100, embed_dim=768, num_heads=12, depth=2):
        super(AttentiveProbe, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_slices + 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4.0, qkv_bias=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)   # (B, S+1, D)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # CLS token -> (B, D)


# ---------------------------------------------------------------------------
# Classification heads
# ---------------------------------------------------------------------------

class LinearHead(nn.Module):
    """Linear classification head (I-JEPA paper protocol)."""

    def __init__(self, in_dim, out_dim=1):
        super(LinearHead, self).__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(self.norm(x))


class MLPHead(nn.Module):
    """Two-layer MLP classification head."""

    def __init__(self, in_dim, hidden_dim=256, out_dim=1, dropout=0.1):
        super(MLPHead, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# LR schedule with warmup
# ---------------------------------------------------------------------------

def cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Evaluation (works on cached feature tensors)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(probe, head, loader, criterion, device, return_predictions=False):
    """Run evaluation on cached features.

    Returns:
        (loss, auc) or (loss, auc, labels, probs) if return_predictions=True.
    """
    probe.eval()
    head.eval()

    total_loss = 0.0
    n_samples = 0
    all_labels = []
    all_probs = []

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        with autocast():
            pooled = probe(features)             # (B, D)
            logits = head(pooled).squeeze(-1)    # (B,)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits)
        total_loss += loss.item() * labels.size(0)
        n_samples += labels.size(0)
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    avg_loss = total_loss / max(n_samples, 1)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) >= 2 else 0.5
    if return_predictions:
        return avg_loss, auc, all_labels, all_probs
    return avg_loss, auc


# ---------------------------------------------------------------------------
# Feature pre-computation (one-time cost, cached to disk)
# ---------------------------------------------------------------------------

def precompute_features(encoder, data_dir, split, num_slices, slice_size,
                        device, chunk_size=50, cache_dir=None):
    """Encode all volumes in a split with the frozen ViT and cache to disk.

    Returns:
        features: (N, num_slices, embed_dim) float32
        labels:   (N,) long
    """
    cache_path = None
    if cache_dir:
        cache_path = os.path.join(cache_dir, '%s_s%d.pt' % (split, num_slices))
        if os.path.exists(cache_path):
            print('  Loading cached %s features from %s' % (split, cache_path))
            data = torch.load(cache_path, map_location='cpu')
            return data['features'], data['labels']

    split_dir = os.path.join(data_dir, split)
    dataset = OCTVolumeDataset(
        split_dir, num_slices=num_slices, slice_size=slice_size, return_label=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=4, pin_memory=True)

    all_features = []
    all_labels = []

    encoder.eval()
    t0 = time.time()
    with torch.no_grad():
        for i, (volume, label) in enumerate(loader):
            volume = volume.to(device)       # (1, S, 3, H, W)
            flat = volume.squeeze(0)          # (S, 3, H, W)

            parts = []
            for j in range(0, flat.size(0), chunk_size):
                chunk = flat[j:j + chunk_size]
                chunk = imagenet_normalize(chunk)  # match pretraining distribution
                with autocast():
                    out = encoder(chunk)      # (chunk, patches, D)
                parts.append(out.mean(dim=1).cpu())  # (chunk, D)

            all_features.append(torch.cat(parts, dim=0))  # (S, D)
            all_labels.append(label.squeeze())

            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                print('    %s: %d/%d volumes (%.0fs)'
                      % (split, i + 1, len(dataset), elapsed))

    features = torch.stack(all_features)     # (N, S, D)
    labels = torch.stack(all_labels).long()  # (N,)
    elapsed = time.time() - t0
    print('  %s: %d volumes encoded in %.0fs (%.1f vol/s)'
          % (split, len(dataset), elapsed, len(dataset) / max(elapsed, 1)))

    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save({'features': features, 'labels': labels}, cache_path)
        size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        print('  Cached to %s (%.1f MB)' % (cache_path, size_mb))

    return features, labels


# ---------------------------------------------------------------------------
# Diagnostic plots (generated at end of training)
# ---------------------------------------------------------------------------

def _save_diagnostic_plots(output_dir, test_labels, test_probs, test_auc,
                           val_labels, val_probs):
    """Generate ROC curve, confusion matrix, and prediction histogram."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, confusion_matrix
    except ImportError:
        print('  Skipping plots (matplotlib not available)')
        return

    if test_labels is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. ROC curve
    ax = axes[0]
    fpr, tpr, thresholds = roc_curve(test_labels, test_probs)
    ax.plot(fpr, tpr, 'b-', linewidth=2, label='Test AUC = %.3f' % (test_auc or 0))
    if val_labels is not None:
        fpr_v, tpr_v, _ = roc_curve(val_labels, val_probs)
        val_auc = roc_auc_score(val_labels, val_probs)
        ax.plot(fpr_v, tpr_v, 'g--', linewidth=1.5, label='Val AUC = %.3f' % val_auc)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random (0.5)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    # 2. Confusion matrix at threshold=0.5
    ax = axes[1]
    preds = (test_probs >= 0.5).astype(int)
    cm = confusion_matrix(test_labels, preds)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix (threshold=0.5)')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Non-Glaucoma', 'Glaucoma'])
    ax.set_yticklabels(['Non-Glaucoma', 'Glaucoma'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=16)

    # 3. Prediction histogram
    ax = axes[2]
    ax.hist(test_probs[test_labels == 0], bins=30, alpha=0.6, color='blue',
            label='Non-Glaucoma (n=%d)' % (test_labels == 0).sum(), density=True)
    ax.hist(test_probs[test_labels == 1], bins=30, alpha=0.6, color='red',
            label='Glaucoma (n=%d)' % (test_labels == 1).sum(), density=True)
    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold=0.5')
    ax.set_xlabel('P(Glaucoma)')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Distribution')
    ax.legend()

    fig.tight_layout()
    plot_path = os.path.join(output_dir, 'diagnostic_plots.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved diagnostic_plots.png')


# ---------------------------------------------------------------------------
# Patch-level downstream
# ---------------------------------------------------------------------------

def run_patch_downstream(config, device):
    """Downstream glaucoma classification with I-JEPA pretrained encoder.

    Protocol (matched to SLIViT baseline for fair comparison):
      1. Pre-compute: encode all volumes with frozen ViT, cache to disk
      2. Train: AttentiveProbe (2 blocks) + LinearHead on cached features
      3. Early stop on val AUC, patience=5
      4. Evaluate best model on test set, report test AUC
    """
    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    log_cfg = config['logging']

    output_dir = log_cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 70)
    print('Downstream Classification — I-JEPA Attentive Probe')
    print('=' * 70)

    # ---- Load pretrained encoder -------------------------------------------
    vit_cfg = _VIT_CONFIGS[model_cfg['encoder_name']]
    encoder = VisionTransformer(
        img_size=model_cfg['crop_size'],
        patch_size=model_cfg['patch_size'],
        embed_dim=vit_cfg['embed_dim'],
        depth=vit_cfg['depth'],
        num_heads=vit_cfg['num_heads'],
    ).to(device)

    ckpt_path = model_cfg['encoder_checkpoint']
    print('Loading encoder from %s ...' % ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt['target_encoder'])
    print('  Loaded target_encoder weights (epoch %d)' % ckpt.get('epoch', -1))

    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    embed_dim = vit_cfg['embed_dim']
    num_slices = data_cfg['num_slices']

    # ---- Pre-compute features (one-time) -----------------------------------
    print('\n--- Pre-computing features with frozen encoder ---')
    slice_size = data_cfg.get('slice_size', 256)
    chunk_size = data_cfg.get('encode_chunk_size', 50)
    cache_dir = os.path.join(output_dir, 'feature_cache')

    train_feats, train_labels = precompute_features(
        encoder, data_cfg['data_dir'], 'Training',
        num_slices, slice_size, device, chunk_size, cache_dir)
    val_feats, val_labels = precompute_features(
        encoder, data_cfg['data_dir'], 'Validation',
        num_slices, slice_size, device, chunk_size, cache_dir)
    test_feats, test_labels = precompute_features(
        encoder, data_cfg['data_dir'], 'Test',
        num_slices, slice_size, device, chunk_size, cache_dir)

    # Free encoder from GPU after feature extraction
    encoder.cpu()
    torch.cuda.empty_cache()

    n_pos = int(train_labels.sum().item())
    n_neg = len(train_labels) - n_pos
    print('  Train: %d volumes (%d pos, %d neg, %.1f%% prevalence)'
          % (len(train_labels), n_pos, n_neg, 100.0 * n_pos / len(train_labels)))
    print('  Val:   %d volumes' % len(val_labels))
    print('  Test:  %d volumes' % len(test_labels))

    # ---- Data loaders on cached features -----------------------------------
    batch_size = data_cfg.get('batch_size', 16)

    train_loader = DataLoader(
        TensorDataset(train_feats, train_labels.float()),
        batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(
        TensorDataset(val_feats, val_labels.float()),
        batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(
        TensorDataset(test_feats, test_labels.float()),
        batch_size=batch_size, shuffle=False, pin_memory=True)

    # ---- Attentive probe + head --------------------------------------------
    print('\n--- Model ---')
    probe = AttentiveProbe(
        num_slices=num_slices,
        embed_dim=embed_dim,
        num_heads=model_cfg.get('probe_num_heads', 12),
        depth=model_cfg.get('probe_depth', 2),
    ).to(device)

    head_type = model_cfg.get('head_type', 'linear')
    if head_type == 'mlp':
        head = MLPHead(in_dim=embed_dim, dropout=train_cfg.get('dropout', 0.1)).to(device)
    else:
        head = LinearHead(in_dim=embed_dim).to(device)

    probe_params = sum(p.numel() for p in probe.parameters())
    head_params = sum(p.numel() for p in head.parameters())
    enc_params = sum(p.numel() for p in encoder.parameters())
    print('  Frozen encoder:  %s params' % format(enc_params, ','))
    print('  Attentive probe: %s params (trainable, depth=%d)'
          % (format(probe_params, ','), model_cfg.get('probe_depth', 2)))
    print('  Head (%s):     %s params (trainable)' % (head_type, format(head_params, ',')))
    print('  Total trainable: %s' % format(probe_params + head_params, ','))

    # ---- Optimizer (matched to SLIViT protocol) ----------------------------
    param_groups = [
        {'params': probe.parameters(), 'lr': train_cfg.get('lr_probe', 1e-4)},
        {'params': head.parameters(), 'lr': train_cfg.get('lr_head', 1e-3)},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=train_cfg.get('weight_decay', 0.01))
    scheduler = cosine_schedule_with_warmup(
        optimizer, train_cfg.get('warmup_epochs', 3),
        train_cfg['epochs'], len(train_loader),
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    # ---- CSV logger --------------------------------------------------------
    csv_path = os.path.join(output_dir, 'train_log.csv')
    csv_file = open(csv_path, 'w')
    csv_file.write('epoch,train_loss,train_auc,val_loss,val_auc,lr,elapsed_s\n')
    csv_file.flush()

    # ---- Training loop -----------------------------------------------------
    print('\n--- Training ---')
    best_auc = 0.0
    patience_counter = 0
    patience = train_cfg.get('patience', 5)
    epochs = train_cfg['epochs']

    for epoch in range(1, epochs + 1):
        probe.train()
        head.train()
        total_loss = 0.0
        n_samples = 0
        train_labels_epoch = []
        train_probs_epoch = []

        t0 = time.time()
        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                pooled = probe(features)         # (B, D)
                logits = head(pooled).squeeze(-1)
                loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item() * labels.size(0)
            n_samples += labels.size(0)
            with torch.no_grad():
                train_labels_epoch.append(labels.cpu())
                train_probs_epoch.append(torch.sigmoid(logits).cpu())

        elapsed = time.time() - t0
        train_loss = total_loss / max(n_samples, 1)
        train_labels_np = torch.cat(train_labels_epoch).numpy()
        train_probs_np = torch.cat(train_probs_epoch).numpy()
        train_auc = roc_auc_score(train_labels_np, train_probs_np) if len(np.unique(train_labels_np)) >= 2 else 0.5

        val_loss, val_auc = evaluate(probe, head, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]['lr']

        improved = val_auc > best_auc
        marker = ' *' if improved else ''
        print('Epoch %2d/%d (%4.1fs) | Train: %.4f (AUC %.3f) | Val: %.4f | AUC: %.4f | LR: %.2e%s'
              % (epoch, epochs, elapsed, train_loss, train_auc, val_loss, val_auc,
                 current_lr, marker))

        csv_file.write('%d,%.6f,%.6f,%.6f,%.6f,%.8f,%.1f\n'
                       % (epoch, train_loss, train_auc, val_loss, val_auc, current_lr, elapsed))
        csv_file.flush()

        if improved:
            best_auc = val_auc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'probe': probe.state_dict(),
                'head': head.state_dict(),
                'val_auc': val_auc,
                'val_loss': val_loss,
            }, os.path.join(output_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping at epoch %d (patience=%d)' % (epoch, patience))
                break

    csv_file.close()

    # ---- Test evaluation (with best model) ---------------------------------
    print('\n--- Test Evaluation ---')
    best_path = os.path.join(output_dir, 'best_model.pt')
    test_loss = test_auc = best_epoch = None
    test_labels = test_probs = None
    val_labels_final = val_probs_final = None

    if not os.path.exists(best_path):
        print('ERROR: best_model.pt not found — no epoch improved over AUC=0')
        best_epoch = 0
    else:
        best_ckpt = torch.load(best_path, map_location=device)
        probe.load_state_dict(best_ckpt['probe'])
        head.load_state_dict(best_ckpt['head'])
        best_epoch = best_ckpt['epoch']

        # Val predictions (for ROC curve)
        val_loss_f, val_auc_f, val_labels_final, val_probs_final = evaluate(
            probe, head, val_loader, criterion, device, return_predictions=True)

        # Test predictions
        test_loss, test_auc, test_labels, test_probs = evaluate(
            probe, head, test_loader, criterion, device, return_predictions=True)

        print('Best epoch: %d  |  Val AUC: %.4f  |  TEST AUC: %.4f'
              % (best_epoch, best_auc, test_auc))

        # Sensitivity / specificity at threshold=0.5
        if test_labels is not None:
            test_preds = (test_probs >= 0.5).astype(int)
            tp = ((test_preds == 1) & (test_labels == 1)).sum()
            tn = ((test_preds == 0) & (test_labels == 0)).sum()
            fp = ((test_preds == 1) & (test_labels == 0)).sum()
            fn = ((test_preds == 0) & (test_labels == 1)).sum()
            sensitivity = tp / max(tp + fn, 1)
            specificity = tn / max(tn + fp, 1)
            print('  Sensitivity: %.4f  |  Specificity: %.4f  (threshold=0.5)' % (sensitivity, specificity))

    # ---- Save predictions --------------------------------------------------
    if test_labels is not None:
        np.savez(os.path.join(output_dir, 'test_predictions.npz'),
                 labels=test_labels, probs=test_probs)
        print('  Saved test_predictions.npz (%d samples)' % len(test_labels))
    if val_labels_final is not None:
        np.savez(os.path.join(output_dir, 'val_predictions.npz'),
                 labels=val_labels_final, probs=val_probs_final)
        print('  Saved val_predictions.npz (%d samples)' % len(val_labels_final))

    # ---- Generate diagnostic plots -----------------------------------------
    _save_diagnostic_plots(output_dir, test_labels, test_probs, test_auc,
                           val_labels_final, val_probs_final)

    # ---- Save results ------------------------------------------------------
    results = {
        'mode': 'patch',
        'head_type': head_type,
        'num_slices': num_slices,
        'probe_depth': model_cfg.get('probe_depth', 2),
        'best_epoch': best_epoch,
        'best_val_auc': best_auc,
        'test_auc': test_auc,
        'test_loss': test_loss,
        'sensitivity': float(sensitivity) if test_labels is not None else None,
        'specificity': float(specificity) if test_labels is not None else None,
        'probe_params': probe_params,
        'head_params': head_params,
        'config': config,
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nResults saved to %s' % output_dir)
    print('  best_val_auc = %.4f' % best_auc)
    print('  test_auc     = %.4f' % (test_auc if test_auc else 0))
    print('  (SLIViT baseline: 0.869 test AUC)')

    return results


# ---------------------------------------------------------------------------
# Slice-level downstream
# ---------------------------------------------------------------------------

def evaluate_slice(encode_fn, head, loader, criterion, device):
    """Evaluate slice-level downstream (non-cached path)."""
    head.eval()
    total_loss = 0.0
    n_samples = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for volumes, labels in loader:
            volumes = volumes.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            features = encode_fn(volumes)
            logits = head(features).squeeze(-1)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
            total_loss += loss.item() * labels.size(0)
            n_samples += labels.size(0)
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    avg_loss = total_loss / max(n_samples, 1)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) >= 2 else 0.5
    return avg_loss, auc


def run_slice_downstream(config, device):
    """Downstream evaluation using a slice-level I-JEPA pretrained encoder."""
    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    log_cfg = config['logging']

    output_dir = log_cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 70)
    print('Downstream Classification (slice-level pretrained)')
    print('=' * 70)

    # ---- Frozen feature extractor ------------------------------------------
    fe_checkpoint = model_cfg.get('fe_checkpoint', None)
    feature_extractor = FrozenFeatureExtractor(checkpoint_path=fe_checkpoint).to(device)

    # ---- Load pretrained slice encoder -------------------------------------
    slice_encoder = SliceEncoder(
        num_slices=data_cfg['num_slices'],
        embed_dim=model_cfg['enc_dim'],
        depth=model_cfg['enc_depth'],
        num_heads=model_cfg['enc_heads'],
    ).to(device)

    ckpt_path = model_cfg['slice_encoder_checkpoint']
    print('Loading slice encoder from %s ...' % ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    slice_encoder.load_state_dict(ckpt['target_encoder'])
    print('  Loaded target_encoder weights (epoch %d)' % ckpt.get('epoch', -1))

    if model_cfg.get('freeze_encoder', True):
        for p in slice_encoder.parameters():
            p.requires_grad = False
        slice_encoder.eval()

    embed_dim = model_cfg['enc_dim']

    # ---- MLP head ----------------------------------------------------------
    head = MLPHead(in_dim=embed_dim).to(device)
    head_params = sum(p.numel() for p in head.parameters())
    print('  Head params: %s' % format(head_params, ','))

    # ---- Encode function ---------------------------------------------------
    @torch.no_grad()
    def encode_fn(volumes):
        """Encode volume: frozen ConvNeXt -> frozen slice encoder -> mean pool."""
        B, S, C, H, W = volumes.shape
        flat = volumes.reshape(B * S, C, H, W)
        slice_features = feature_extractor(flat)  # (B*S, 768)
        slice_features = slice_features.reshape(B, S, -1)  # (B, S, 768)
        encoded = slice_encoder(slice_features)  # (B, S, D)
        pooled = encoded.mean(dim=1)  # (B, D)
        return pooled

    # ---- Datasets ----------------------------------------------------------
    num_slices = data_cfg['num_slices']
    slice_size = data_cfg.get('slice_size', 256)

    train_dataset = OCTVolumeDataset(
        os.path.join(data_cfg['data_dir'], 'Training'),
        num_slices=num_slices, slice_size=slice_size, return_label=True,
    )
    val_dataset = OCTVolumeDataset(
        os.path.join(data_cfg['data_dir'], 'Validation'),
        num_slices=num_slices, slice_size=slice_size, return_label=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=data_cfg['batch_size'],
                              shuffle=True, num_workers=data_cfg['num_workers'],
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=data_cfg['batch_size'],
                            shuffle=False, num_workers=data_cfg['num_workers'],
                            pin_memory=True)

    print('  Train: %d volumes' % len(train_dataset))
    print('  Val:   %d volumes' % len(val_dataset))

    # ---- Optimizer ---------------------------------------------------------
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=train_cfg.get('lr', 1e-3),
        weight_decay=train_cfg.get('weight_decay', 0.01),
    )
    scheduler = cosine_schedule_with_warmup(
        optimizer, warmup_epochs=3, total_epochs=train_cfg['epochs'],
        steps_per_epoch=len(train_loader),
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    # ---- Training loop -----------------------------------------------------
    best_auc = 0.0
    patience_counter = 0
    patience = train_cfg.get('patience', 5)

    for epoch in range(1, train_cfg['epochs'] + 1):
        head.train()
        total_loss = 0.0
        n_samples = 0

        t0 = time.time()
        for volumes, labels in train_loader:
            volumes = volumes.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            with autocast():
                features = encode_fn(volumes)
                logits = head(features).squeeze(-1)
                loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item() * labels.size(0)
            n_samples += labels.size(0)

        elapsed = time.time() - t0
        train_loss = total_loss / max(n_samples, 1)
        val_loss, val_auc = evaluate_slice(encode_fn, head, val_loader, criterion, device)

        improved = val_auc > best_auc
        marker = ' *' if improved else ''
        print('Epoch %d/%d (%4.0fs) | Train Loss: %.4f | Val Loss: %.4f | Val AUC: %.4f%s'
              % (epoch, train_cfg['epochs'], elapsed, train_loss, val_loss, val_auc, marker))

        if improved:
            best_auc = val_auc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'head': head.state_dict(),
                'val_auc': val_auc,
            }, os.path.join(output_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping at epoch %d' % epoch)
                break

    # ---- Test evaluation ---------------------------------------------------
    test_dir = os.path.join(data_cfg['data_dir'], 'Test')
    test_auc = None
    test_loss = None
    if os.path.isdir(test_dir):
        best_ckpt = torch.load(os.path.join(output_dir, 'best_model.pt'), map_location=device)
        head.load_state_dict(best_ckpt['head'])

        test_dataset = OCTVolumeDataset(test_dir, num_slices=num_slices,
                                        slice_size=slice_size, return_label=True)
        test_loader = DataLoader(test_dataset, batch_size=data_cfg['batch_size'],
                                 shuffle=False, num_workers=data_cfg['num_workers'],
                                 pin_memory=True)
        print('  Test: %d volumes' % len(test_dataset))
        test_loss, test_auc = evaluate_slice(encode_fn, head, test_loader, criterion, device)
        print('TEST Loss: %.4f | TEST AUC: %.4f' % (test_loss, test_auc))
    else:
        print('No Test directory found, skipping test evaluation.')

    # ---- Save results ------------------------------------------------------
    results = {
        'mode': 'slice',
        'best_val_auc': best_auc,
        'test_auc': test_auc,
        'test_loss': test_loss,
        'config': config,
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('Results saved to %s' % output_dir)

    return results


# ---------------------------------------------------------------------------
# Combined model for DDP fine-tuning
# ---------------------------------------------------------------------------

class DownstreamModel(nn.Module):
    """End-to-end model: ViT encoder + AttentiveProbe + head.

    Wraps the full pipeline so DDP can sync gradients correctly.
    Encodes slices in chunks to fit memory while preserving gradients.
    """

    def __init__(self, encoder, probe, head, chunk_size=25):
        super(DownstreamModel, self).__init__()
        self.encoder = encoder
        self.probe = probe
        self.head = head
        self.chunk_size = chunk_size

    def forward(self, volumes):
        B, S, C, H, W = volumes.shape
        flat = volumes.reshape(B * S, C, H, W)
        flat = imagenet_normalize(flat)  # match pretraining distribution
        parts = []
        for i in range(0, flat.size(0), self.chunk_size):
            chunk = flat[i:i + self.chunk_size]
            out = self.encoder(chunk)          # (chunk, patches, D)
            parts.append(out.mean(dim=1))      # (chunk, D)
        features = torch.cat(parts, dim=0)     # (B*S, D)
        features = features.reshape(B, S, -1)  # (B, S, D)
        pooled = self.probe(features)          # (B, D)
        return self.head(pooled).squeeze(-1)   # (B,)


# ---------------------------------------------------------------------------
# Patch-level fine-tuning (encoder unfrozen, DDP)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_finetune(model, loader, criterion, device):
    """Evaluate fine-tune model on a data loader."""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_labels = []
    all_probs = []

    for volumes, labels in loader:
        volumes = volumes.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        with autocast():
            logits = model(volumes)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)
        total_loss += loss.item() * labels.size(0)
        n_samples += labels.size(0)
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    # Gather across ranks for full AUC
    if dist.is_initialized() and dist.get_world_size() > 1:
        gathered_labels = [None] * dist.get_world_size()
        gathered_probs = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_labels, all_labels)
        dist.all_gather_object(gathered_probs, all_probs)
        all_labels = np.concatenate(gathered_labels)
        all_probs = np.concatenate(gathered_probs)

        # Gather loss across ranks
        loss_tensor = torch.tensor([total_loss, float(n_samples)], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss = loss_tensor[0].item()
        n_samples = int(loss_tensor[1].item())

    avg_loss = total_loss / max(n_samples, 1)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) >= 2 else 0.5
    return avg_loss, auc


def run_patch_finetune(config, device, rank=0, world_size=1):
    """Fine-tune encoder + probe + head end-to-end with DDP.

    Protocol:
      - Encoder: very low LR (5e-6), unfrozen
      - Probe + head: normal LR
      - batch_size=1 per GPU, gradient accumulation, DDP
      - Early stop on val AUC, patience=5
    """
    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    log_cfg = config['logging']

    output_dir = log_cfg['output_dir']
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    is_main = (rank == 0)

    if is_main:
        print('=' * 70)
        print('Downstream Fine-tuning — Encoder + Probe + Head (DDP)')
        print('  World size: %d' % world_size)
        print('=' * 70)

    # ---- Build model -------------------------------------------------------
    vit_cfg = _VIT_CONFIGS[model_cfg['encoder_name']]
    encoder = VisionTransformer(
        img_size=model_cfg['crop_size'],
        patch_size=model_cfg['patch_size'],
        embed_dim=vit_cfg['embed_dim'],
        depth=vit_cfg['depth'],
        num_heads=vit_cfg['num_heads'],
    )

    ckpt_path = model_cfg['encoder_checkpoint']
    if is_main:
        print('Loading encoder from %s ...' % ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    encoder.load_state_dict(ckpt['target_encoder'])
    if is_main:
        print('  Loaded target_encoder weights (epoch %d)' % ckpt.get('epoch', -1))

    embed_dim = vit_cfg['embed_dim']
    num_slices = data_cfg['num_slices']

    probe = AttentiveProbe(
        num_slices=num_slices,
        embed_dim=embed_dim,
        num_heads=model_cfg.get('probe_num_heads', 12),
        depth=model_cfg.get('probe_depth', 2),
    )

    head_type = model_cfg.get('head_type', 'linear')
    if head_type == 'mlp':
        head = MLPHead(in_dim=embed_dim, dropout=train_cfg.get('dropout', 0.1))
    else:
        head = LinearHead(in_dim=embed_dim)

    chunk_size = data_cfg.get('encode_chunk_size', 25)
    model = DownstreamModel(encoder, probe, head, chunk_size).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    raw = model.module if hasattr(model, 'module') else model

    enc_params = sum(p.numel() for p in raw.encoder.parameters())
    probe_params = sum(p.numel() for p in raw.probe.parameters())
    head_params = sum(p.numel() for p in raw.head.parameters())
    if is_main:
        print('  Encoder:  %s params (trainable, lr=%.1e)'
              % (format(enc_params, ','), train_cfg.get('lr_encoder', 5e-6)))
        print('  Probe:    %s params (trainable, lr=%.1e)'
              % (format(probe_params, ','), train_cfg.get('lr_probe', 1e-4)))
        print('  Head:     %s params (trainable, lr=%.1e)'
              % (format(head_params, ','), train_cfg.get('lr_head', 1e-3)))

    # ---- Datasets ----------------------------------------------------------
    slice_size = data_cfg.get('slice_size', 256)
    batch_size = data_cfg.get('batch_size', 1)
    accum_steps = train_cfg.get('accum_steps', 4)

    train_dataset = OCTVolumeDataset(
        os.path.join(data_cfg['data_dir'], 'Training'),
        num_slices=num_slices, slice_size=slice_size, return_label=True,
    )
    val_dataset = OCTVolumeDataset(
        os.path.join(data_cfg['data_dir'], 'Validation'),
        num_slices=num_slices, slice_size=slice_size, return_label=True,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=data_cfg.get('num_workers', 2), pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, sampler=val_sampler,
        num_workers=data_cfg.get('num_workers', 2), pin_memory=True)

    eff_batch = batch_size * world_size * accum_steps
    if is_main:
        print('  Train: %d volumes  (bs=%d × %d GPUs × %d accum = %d eff)'
              % (len(train_dataset), batch_size, world_size, accum_steps, eff_batch))
        print('  Val:   %d volumes' % len(val_dataset))

    # ---- Optimizer ---------------------------------------------------------
    param_groups = [
        {'params': raw.encoder.parameters(), 'lr': train_cfg.get('lr_encoder', 5e-6)},
        {'params': raw.probe.parameters(), 'lr': train_cfg.get('lr_probe', 1e-4)},
        {'params': raw.head.parameters(), 'lr': train_cfg.get('lr_head', 1e-3)},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=train_cfg.get('weight_decay', 0.01))
    steps_per_epoch = len(train_loader) // accum_steps
    scheduler = cosine_schedule_with_warmup(
        optimizer, train_cfg.get('warmup_epochs', 3),
        train_cfg['epochs'], max(steps_per_epoch, 1),
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    # ---- CSV logger --------------------------------------------------------
    csv_file = None
    if is_main:
        csv_path = os.path.join(output_dir, 'train_log.csv')
        csv_file = open(csv_path, 'w')
        csv_file.write('epoch,train_loss,val_loss,val_auc,lr_enc,lr_probe,elapsed_s\n')
        csv_file.flush()

    # ---- Training loop -----------------------------------------------------
    if is_main:
        print('\n--- Training ---')
    best_auc = 0.0
    patience_counter = 0
    patience = train_cfg.get('patience', 5)
    epochs = train_cfg['epochs']

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        n_samples = 0
        optimizer.zero_grad(set_to_none=True)

        t0 = time.time()
        for step, (volumes, labels) in enumerate(train_loader):
            volumes = volumes.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            with autocast():
                logits = model(volumes)
                loss = criterion(logits, labels) / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            total_loss += loss.item() * accum_steps * labels.size(0)
            n_samples += labels.size(0)

        elapsed = time.time() - t0
        train_loss = total_loss / max(n_samples, 1)
        val_loss, val_auc = evaluate_finetune(model, val_loader, criterion, device)
        lr_enc = optimizer.param_groups[0]['lr']
        lr_probe = optimizer.param_groups[1]['lr']

        should_stop = False
        if is_main:
            improved = val_auc > best_auc
            marker = ' *' if improved else ''
            print('Epoch %2d/%d (%5.0fs) | Train: %.4f | Val: %.4f | AUC: %.4f | LR: %.1e/%.1e%s'
                  % (epoch, epochs, elapsed, train_loss, val_loss, val_auc,
                     lr_enc, lr_probe, marker))

            if csv_file:
                csv_file.write('%d,%.6f,%.6f,%.6f,%.8f,%.8f,%.1f\n'
                               % (epoch, train_loss, val_loss, val_auc, lr_enc, lr_probe, elapsed))
                csv_file.flush()

            if improved:
                best_auc = val_auc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'encoder': raw.encoder.state_dict(),
                    'probe': raw.probe.state_dict(),
                    'head': raw.head.state_dict(),
                    'val_auc': val_auc,
                }, os.path.join(output_dir, 'best_model.pt'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print('Early stopping at epoch %d (patience=%d)' % (epoch, patience))
                    should_stop = True

        # Broadcast early stop decision — ALL ranks must reach this
        if world_size > 1:
            stop_tensor = torch.tensor([should_stop], device=device)
            dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item():
                break
        elif should_stop:
            break

    if csv_file:
        csv_file.close()

    # ---- Tear down DDP before test eval (prevents NCCL timeout) ------------
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    # ---- Test evaluation (rank 0 only, no DDP) -----------------------------
    if is_main:
        print('\n--- Test Evaluation ---')
        best_path = os.path.join(output_dir, 'best_model.pt')
        if os.path.exists(best_path):
            best_ckpt = torch.load(best_path, map_location=device)
            raw.encoder.load_state_dict(best_ckpt['encoder'])
            raw.probe.load_state_dict(best_ckpt['probe'])
            raw.head.load_state_dict(best_ckpt['head'])
            best_epoch = best_ckpt['epoch']
        else:
            best_epoch = 0

        test_dataset = OCTVolumeDataset(
            os.path.join(data_cfg['data_dir'], 'Test'),
            num_slices=num_slices, slice_size=slice_size, return_label=True,
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=2, pin_memory=True)
        test_model = raw.to(device)
        test_loss, test_auc = evaluate_finetune(test_model, test_loader, criterion, device)
        print('Best epoch: %d  |  Val AUC: %.4f  |  TEST AUC: %.4f'
              % (best_epoch, best_auc, test_auc))

        results = {
            'mode': 'patch_finetune',
            'head_type': head_type,
            'num_slices': num_slices,
            'probe_depth': model_cfg.get('probe_depth', 2),
            'best_epoch': best_epoch,
            'best_val_auc': best_auc,
            'test_auc': test_auc,
            'test_loss': test_loss,
            'lr_encoder': train_cfg.get('lr_encoder', 5e-6),
            'accum_steps': accum_steps,
            'effective_batch': eff_batch,
            'config': config,
        }
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print('\nResults saved to %s' % output_dir)
        print('  best_val_auc = %.4f' % best_auc)
        print('  test_auc     = %.4f' % test_auc)
        print('  (SLIViT baseline: 0.869, frozen probe: 0.733)')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    seed = config.get('training', {}).get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    freeze_encoder = config.get('model', {}).get('freeze_encoder', True)

    if not freeze_encoder:
        # DDP mode for fine-tuning
        world_size, rank = init_distributed()
        device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
        if rank == 0:
            print('GPU: %s' % torch.cuda.get_device_name(0))
        run_patch_finetune(config, device, rank, world_size)
    else:
        # Single GPU for frozen probe
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print('GPU: %s' % torch.cuda.get_device_name(0))
        mode = config.get('mode', 'patch')
        if mode == 'patch':
            run_patch_downstream(config, device)
        elif mode == 'slice':
            run_slice_downstream(config, device)
        else:
            raise ValueError("Unknown mode: %s" % mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downstream glaucoma classification')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    args = parser.parse_args()
    main(args)
