"""
Downstream glaucoma classification using pretrained I-JEPA encoder.

Supports both patch-level and slice-level pretrained models:
  - Patch-level: each slice is encoded by the frozen ViT, then a trainable
    ViT integrator pools across slices followed by an MLP head.
  - Slice-level: slices are encoded by frozen ConvNeXt + frozen slice encoder,
    then mean-pooled and classified by a trainable MLP head.

Usage:
    # Patch-level pretrained -> ViT integrator + MLP
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
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import yaml

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.models.vision_transformer import (
    VisionTransformer, SliceEncoder, Block, VIT_EMBED_DIMS,
)
from src.models.feature_extractor import FrozenFeatureExtractor
from src.datasets.oct_volumes import OCTVolumeDataset
from src.helper import _VIT_CONFIGS


# ---------------------------------------------------------------------------
# ViT integrator for patch-level downstream
# ---------------------------------------------------------------------------

class ViTIntegrator(nn.Module):
    """Lightweight ViT that integrates per-slice CLS tokens across a volume.

    Input: (B, num_slices, embed_dim) -- one representation per slice.
    Output: (B, embed_dim) -- pooled volume representation.
    """

    def __init__(self, num_slices=32, embed_dim=768, depth=5, num_heads=12):
        super(ViTIntegrator, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_slices + 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4.0, qkv_bias=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # CLS token


# ---------------------------------------------------------------------------
# MLP classification head
# ---------------------------------------------------------------------------

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
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(encode_fn, head, loader, criterion, device, integrator=None):
    """Run evaluation and return (loss, AUC)."""
    if integrator is not None:
        integrator.eval()
    head.eval()

    total_loss = 0.0
    n_samples = 0
    all_labels = []
    all_probs = []

    for volumes, labels in loader:
        volumes = volumes.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        features = encode_fn(volumes)  # (B, dim) or (B, S, dim)
        if integrator is not None:
            features = integrator(features)
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


# ---------------------------------------------------------------------------
# Patch-level downstream
# ---------------------------------------------------------------------------

def run_patch_downstream(config, device):
    """Downstream evaluation using a patch-level I-JEPA pretrained encoder."""
    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    log_cfg = config['logging']

    output_dir = log_cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 70)
    print('Downstream Classification (patch-level pretrained)')
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

    # Load target_encoder weights from I-JEPA checkpoint
    ckpt_path = model_cfg['encoder_checkpoint']
    print('Loading encoder from %s ...' % ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt['target_encoder'])
    print('  Loaded target_encoder weights (epoch %d)' % ckpt.get('epoch', -1))

    if model_cfg.get('freeze_encoder', True):
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()

    embed_dim = vit_cfg['embed_dim']

    # ---- Integrator + head -------------------------------------------------
    integrator = ViTIntegrator(
        num_slices=data_cfg['num_slices'],
        embed_dim=embed_dim,
        depth=model_cfg.get('integrator_depth', 5),
        num_heads=model_cfg.get('integrator_heads', 12),
    ).to(device)

    head = MLPHead(in_dim=embed_dim).to(device)

    int_params = sum(p.numel() for p in integrator.parameters())
    head_params = sum(p.numel() for p in head.parameters())
    print('  Integrator params: %s' % format(int_params, ','))
    print('  Head params:       %s' % format(head_params, ','))

    # ---- Encode function ---------------------------------------------------
    @torch.no_grad()
    def encode_fn(volumes):
        """Encode each slice with frozen ViT, mean-pool patch tokens."""
        B, S, C, H, W = volumes.shape
        flat = volumes.reshape(B * S, C, H, W)
        features = encoder(flat)  # (B*S, num_patches, D)
        # Mean-pool over patch tokens as the slice representation
        cls_features = features.mean(dim=1)  # (B*S, D)
        return cls_features.reshape(B, S, -1)  # (B, S, D)

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
    param_groups = [
        {'params': integrator.parameters(), 'lr': train_cfg.get('lr_integrator', 1e-4)},
        {'params': head.parameters(), 'lr': train_cfg.get('lr_head', 1e-3)},
    ]
    # Optionally fine-tune encoder
    if not model_cfg.get('freeze_encoder', True):
        param_groups.append(
            {'params': encoder.parameters(), 'lr': train_cfg.get('lr_encoder', 1e-6)}
        )

    optimizer = torch.optim.AdamW(param_groups, weight_decay=train_cfg.get('weight_decay', 0.01))
    scheduler = cosine_schedule_with_warmup(
        optimizer, train_cfg.get('warmup_epochs', 3),
        train_cfg['epochs'], len(train_loader),
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    # ---- Training loop -----------------------------------------------------
    best_auc = 0.0
    patience_counter = 0
    patience = train_cfg.get('patience', 10)

    for epoch in range(1, train_cfg['epochs'] + 1):
        integrator.train()
        head.train()
        total_loss = 0.0
        n_samples = 0

        t0 = time.time()
        for volumes, labels in train_loader:
            volumes = volumes.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            with autocast():
                features = encode_fn(volumes)  # (B, S, D)
                pooled = integrator(features)  # (B, D)
                logits = head(pooled).squeeze(-1)
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
        val_loss, val_auc = evaluate(encode_fn, head, val_loader, criterion, device,
                                     integrator=integrator)

        improved = val_auc > best_auc
        marker = ' *' if improved else ''
        print('Epoch %d/%d (%4.0fs) | Train Loss: %.4f | Val Loss: %.4f | Val AUC: %.4f%s'
              % (epoch, train_cfg['epochs'], elapsed, train_loss, val_loss, val_auc, marker))

        if improved:
            best_auc = val_auc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'integrator': integrator.state_dict(),
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
        # Load best checkpoint
        best_ckpt = torch.load(os.path.join(output_dir, 'best_model.pt'), map_location=device)
        integrator.load_state_dict(best_ckpt['integrator'])
        head.load_state_dict(best_ckpt['head'])

        test_dataset = OCTVolumeDataset(test_dir, num_slices=num_slices,
                                        slice_size=slice_size, return_label=True)
        test_loader = DataLoader(test_dataset, batch_size=data_cfg['batch_size'],
                                 shuffle=False, num_workers=data_cfg['num_workers'],
                                 pin_memory=True)
        print('  Test: %d volumes' % len(test_dataset))
        test_loss, test_auc = evaluate(encode_fn, head, test_loader, criterion, device,
                                       integrator=integrator)
        print('TEST Loss: %.4f | TEST AUC: %.4f' % (test_loss, test_auc))
    else:
        print('No Test directory found, skipping test evaluation.')

    # ---- Save results ------------------------------------------------------
    results = {
        'mode': 'patch',
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
# Slice-level downstream
# ---------------------------------------------------------------------------

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
    patience = train_cfg.get('patience', 20)

    for epoch in range(1, train_cfg['epochs'] + 1):
        head.train()
        total_loss = 0.0
        n_samples = 0

        t0 = time.time()
        for volumes, labels in train_loader:
            volumes = volumes.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            with autocast():
                features = encode_fn(volumes)  # (B, D)
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
        val_loss, val_auc = evaluate(encode_fn, head, val_loader, criterion, device)

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
        test_loss, test_auc = evaluate(encode_fn, head, test_loader, criterion, device)
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
# Main
# ---------------------------------------------------------------------------

def main(args):
    # Seed
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    seed = config.get('training', {}).get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print('GPU: %s' % torch.cuda.get_device_name(0))

    mode = config.get('mode', 'patch')
    if mode == 'patch':
        run_patch_downstream(config, device)
    elif mode == 'slice':
        run_slice_downstream(config, device)
    else:
        raise ValueError("Unknown mode: %s (expected 'patch' or 'slice')" % mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downstream glaucoma classification')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    args = parser.parse_args()
    main(args)
