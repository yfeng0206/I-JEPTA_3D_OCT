"""
DINOv3 ViT-B/16 → AttentiveProbe → MLP/Linear → Glaucoma Classification

Ablation study: test whether DINOv3 (general-purpose SSL on 1.7B images)
produces better OCT features than our I-JEPA pretrained encoder.

Supports:
  - Frozen probe: precompute features with DataParallel on all GPUs, train probe on cached tensors
  - Unfrozen fine-tune: DDP with torchrun on all GPUs

Usage:
  # Frozen probe (single process, multi-GPU for encoding)
  python ablation/dinov3_probe/eval_dinov3.py --config <config.yaml>

  # Unfrozen fine-tune (DDP)
  torchrun --nproc_per_node=4 ablation/dinov3_probe/eval_dinov3.py --config <config.yaml>
"""
import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
import yaml

# Add project root to path for shared code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.eval_downstream import (
    AttentiveProbe, LinearHead, MLPHead,
    cosine_schedule_with_warmup, OCTVolumeDataset,
)


def load_dinov3_encoder(model_name='dinov3_vitb16', weights_path=None):
    """Load DINOv3 encoder via torch.hub or HuggingFace transformers."""

    # Method 1: HuggingFace transformers (preferred, simpler)
    try:
        from transformers import AutoModel
        hf_name = {
            'dinov3_vitb16': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
            'dinov3_vitl16': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
            'dinov3_vits16': 'facebook/dinov3-vits16-pretrain-lvd1689m',
        }.get(model_name, model_name)

        print(f'  Loading {hf_name} from HuggingFace...')
        model = AutoModel.from_pretrained(hf_name, trust_remote_code=True)
        embed_dim = model.config.hidden_size
        print(f'  Loaded: {sum(p.numel() for p in model.parameters()):,} params, embed_dim={embed_dim}')
        return model, embed_dim
    except Exception as e:
        print(f'  HuggingFace load failed: {e}')

    # Method 2: torch.hub with local weights
    if weights_path and os.path.exists(weights_path):
        print(f'  Loading {model_name} from local weights: {weights_path}')
        # Clone dinov3 repo if not present
        repo_dir = os.path.join(os.path.dirname(__file__), 'dinov3_repo')
        if not os.path.exists(repo_dir):
            os.system(f'git clone https://github.com/facebookresearch/dinov3.git {repo_dir}')
        model = torch.hub.load(repo_dir, model_name, source='local', weights=weights_path)
        embed_dim = model.embed_dim
        print(f'  Loaded: {sum(p.numel() for p in model.parameters()):,} params, embed_dim={embed_dim}')
        return model, embed_dim

    raise RuntimeError(
        f'Cannot load {model_name}. Either:\n'
        f'  1. Set HF_TOKEN and request access to the gated model on HuggingFace, or\n'
        f'  2. Download weights manually and pass --weights_path'
    )


class DINOv3Encoder(nn.Module):
    """Wrapper that extracts patch tokens from DINOv3 output."""

    def __init__(self, model, embed_dim):
        super().__init__()
        self.model = model
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) images
        Returns:
            (B, N_patches, embed_dim) patch tokens (no CLS)
        """
        outputs = self.model(x)
        # HuggingFace returns BaseModelOutput with last_hidden_state
        if hasattr(outputs, 'last_hidden_state'):
            tokens = outputs.last_hidden_state  # (B, 1+N_patches, D)
            return tokens[:, 1:, :]  # drop CLS, keep patch tokens
        # torch.hub returns tensor directly
        if isinstance(outputs, torch.Tensor):
            if outputs.dim() == 3 and outputs.size(1) > 1:
                return outputs[:, 1:, :]  # drop CLS
            return outputs
        raise ValueError(f'Unexpected output type: {type(outputs)}')


def precompute_features(encoder, data_dir, split, num_slices, slice_size,
                        device, chunk_size=50, cache_dir=None, num_gpus=1):
    """Encode all volumes with frozen DINOv3 and cache to disk."""
    cache_path = None
    if cache_dir:
        cache_path = os.path.join(cache_dir, f'{split}_s{num_slices}.pt')
        if os.path.exists(cache_path):
            print(f'  Loading cached {split} features from {cache_path}')
            data = torch.load(cache_path, map_location='cpu')
            return data['features'], data['labels']

    split_dir = os.path.join(data_dir, split)
    dataset = OCTVolumeDataset(
        split_dir, num_slices=num_slices, slice_size=slice_size, return_label=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=4, pin_memory=True)

    # Use DataParallel if multiple GPUs available (frozen probe only)
    encode_model = encoder
    if num_gpus > 1 and not isinstance(encoder, nn.parallel.DistributedDataParallel):
        print(f'  Using DataParallel across {num_gpus} GPUs for encoding')
        encode_model = nn.DataParallel(encoder)

    all_features = []
    all_labels = []

    encode_model.eval()
    t0 = time.time()
    with torch.no_grad():
        for i, (volume, label) in enumerate(loader):
            volume = volume.to(device)       # (1, S, 3, H, W)
            flat = volume.squeeze(0)          # (S, 3, H, W)

            parts = []
            for j in range(0, flat.size(0), chunk_size):
                chunk = flat[j:j + chunk_size]
                with autocast():
                    out = encode_model(chunk)      # (chunk, patches, D)
                parts.append(out.mean(dim=1).cpu())  # mean-pool patches → (chunk, D)

            all_features.append(torch.cat(parts, dim=0))  # (S, D)
            all_labels.append(label.squeeze())

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f'    {split}: {i + 1}/{len(dataset)} volumes ({elapsed:.0f}s)')

    features = torch.stack(all_features)     # (N, S, D)
    labels = torch.stack(all_labels).long()  # (N,)
    elapsed = time.time() - t0
    print(f'  {split}: {len(dataset)} volumes encoded in {elapsed:.0f}s ({len(dataset) / max(elapsed, 1):.1f} vol/s)')

    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save({'features': features, 'labels': labels}, cache_path)
        size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        print(f'  Cached to {cache_path} ({size_mb:.1f} MB)')

    return features, labels


def run_frozen_probe(config):
    """Frozen DINOv3 encoder → precompute features → train AttentiveProbe + Head."""
    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    output_dir = config['logging']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    print(f'GPUs available: {num_gpus}')

    print('=' * 70)
    print('DINOv3 Ablation — Frozen Probe')
    print('=' * 70)

    # Load encoder
    raw_model, embed_dim = load_dinov3_encoder(
        model_cfg.get('encoder_name', 'dinov3_vitb16'),
        model_cfg.get('weights_path'),
    )
    encoder = DINOv3Encoder(raw_model, embed_dim).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Precompute features
    print('\n--- Pre-computing features with frozen DINOv3 ---')
    num_slices = data_cfg['num_slices']
    slice_size = data_cfg.get('slice_size', 256)
    chunk_size = data_cfg.get('encode_chunk_size', 100)  # larger chunks since no grad
    cache_dir = os.path.join(output_dir, 'feature_cache')

    train_feats, train_labels = precompute_features(
        encoder, data_cfg['data_dir'], 'Training',
        num_slices, slice_size, device, chunk_size, cache_dir, num_gpus)
    val_feats, val_labels = precompute_features(
        encoder, data_cfg['data_dir'], 'Validation',
        num_slices, slice_size, device, chunk_size, cache_dir, num_gpus)
    test_feats, test_labels = precompute_features(
        encoder, data_cfg['data_dir'], 'Test',
        num_slices, slice_size, device, chunk_size, cache_dir, num_gpus)

    # Free encoder memory
    del encoder, raw_model
    torch.cuda.empty_cache()

    # Build probe + head
    probe_depth = model_cfg.get('probe_depth', 3)
    probe_heads = model_cfg.get('probe_num_heads', 12)
    head_type = model_cfg.get('head_type', 'mlp')

    probe = AttentiveProbe(embed_dim, probe_depth, probe_heads, num_slices).to(device)
    if head_type == 'mlp':
        head = MLPHead(embed_dim, dropout=train_cfg.get('dropout', 0.1)).to(device)
    else:
        head = LinearHead(embed_dim).to(device)

    print(f'\n  Probe: {sum(p.numel() for p in probe.parameters()):,} params (depth={probe_depth})')
    print(f'  Head:  {sum(p.numel() for p in head.parameters()):,} params ({head_type})')

    # Training
    train_ds = TensorDataset(train_feats, train_labels)
    val_ds = TensorDataset(val_feats, val_labels)
    bs = data_cfg.get('batch_size', 64)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    param_groups = [
        {'params': probe.parameters(), 'lr': train_cfg.get('lr_probe', 1e-4)},
        {'params': head.parameters(), 'lr': train_cfg.get('lr_head', 1e-3)},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=train_cfg.get('weight_decay', 0))
    scheduler = cosine_schedule_with_warmup(
        optimizer, train_cfg.get('warmup_epochs', 3),
        train_cfg['epochs'], len(train_loader),
    )
    scaler = GradScaler()
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0
    best_epoch = 0
    patience_counter = 0
    patience = train_cfg.get('patience', 20)
    log_rows = []

    print(f'\n--- Training ({train_cfg["epochs"]} epochs, patience={patience}) ---')

    for epoch in range(1, train_cfg['epochs'] + 1):
        t0 = time.time()

        # Train
        probe.train(); head.train()
        total_loss = 0; total_n = 0
        all_probs = []; all_targets = []
        for feats_b, labels_b in train_loader:
            feats_b, labels_b = feats_b.to(device), labels_b.to(device).float()
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                cls_out = probe(feats_b)
                logits = head(cls_out).squeeze(-1)
                loss = criterion(logits, labels_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item() * labels_b.size(0)
            total_n += labels_b.size(0)
            all_probs.append(torch.sigmoid(logits).detach().cpu())
            all_targets.append(labels_b.cpu())

        train_loss = total_loss / total_n
        from sklearn.metrics import roc_auc_score
        train_auc = roc_auc_score(
            torch.cat(all_targets).numpy(), torch.cat(all_probs).numpy())

        # Val
        probe.eval(); head.eval()
        val_loss_sum = 0; val_n = 0
        val_probs = []; val_targets = []
        with torch.no_grad():
            for feats_b, labels_b in val_loader:
                feats_b, labels_b = feats_b.to(device), labels_b.to(device).float()
                with autocast():
                    cls_out = probe(feats_b)
                    logits = head(cls_out).squeeze(-1)
                    loss = criterion(logits, labels_b)
                val_loss_sum += loss.item() * labels_b.size(0)
                val_n += labels_b.size(0)
                val_probs.append(torch.sigmoid(logits).cpu())
                val_targets.append(labels_b.cpu())

        val_loss = val_loss_sum / val_n
        val_auc = roc_auc_score(
            torch.cat(val_targets).numpy(), torch.cat(val_probs).numpy())

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        improved = val_auc > best_val_auc
        if improved:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'probe': probe.state_dict(),
                'head': head.state_dict(),
                'epoch': epoch,
                'val_auc': val_auc,
            }, os.path.join(output_dir, 'best_model.pt'))
        else:
            patience_counter += 1

        marker = ' *' if improved else ''
        print(f'Epoch {epoch:3d}/{train_cfg["epochs"]} ({elapsed:.1f}s) | '
              f'Train: {train_loss:.4f} (AUC {train_auc:.3f}) | '
              f'Val: {val_loss:.4f} | AUC: {val_auc:.4f} | '
              f'LR: {lr:.2e}{marker}')

        log_rows.append({
            'epoch': epoch, 'train_loss': f'{train_loss:.6f}',
            'train_auc': f'{train_auc:.6f}', 'val_loss': f'{val_loss:.6f}',
            'val_auc': f'{val_auc:.6f}', 'lr': f'{lr:.8f}',
            'elapsed_s': f'{elapsed:.1f}',
        })

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch} (patience={patience})')
            break

    # Save training log
    import csv
    with open(os.path.join(output_dir, 'train_log.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    # Test evaluation
    print(f'\n--- Test Evaluation ---')
    ckpt = torch.load(os.path.join(output_dir, 'best_model.pt'), map_location=device)
    probe.load_state_dict(ckpt['probe'])
    head.load_state_dict(ckpt['head'])
    probe.eval(); head.eval()

    test_ds = TensorDataset(test_feats, test_labels)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
    test_probs = []; test_targets = []; test_loss_sum = 0; test_n = 0
    with torch.no_grad():
        for feats_b, labels_b in test_loader:
            feats_b, labels_b = feats_b.to(device), labels_b.to(device).float()
            with autocast():
                cls_out = probe(feats_b)
                logits = head(cls_out).squeeze(-1)
                loss = criterion(logits, labels_b)
            test_loss_sum += loss.item() * labels_b.size(0)
            test_n += labels_b.size(0)
            test_probs.append(torch.sigmoid(logits).cpu())
            test_targets.append(labels_b.cpu())

    test_probs_cat = torch.cat(test_probs).numpy()
    test_targets_cat = torch.cat(test_targets).numpy()
    test_auc = roc_auc_score(test_targets_cat, test_probs_cat)
    test_loss = test_loss_sum / test_n

    preds = (test_probs_cat > 0.5).astype(int)
    sensitivity = (preds[test_targets_cat == 1] == 1).mean()
    specificity = (preds[test_targets_cat == 0] == 0).mean()

    print(f'Best epoch: {best_epoch}  |  Val AUC: {best_val_auc:.4f}  |  TEST AUC: {test_auc:.4f}')
    print(f'  Sensitivity: {sensitivity:.4f}  |  Specificity: {specificity:.4f}  (threshold=0.5)')

    # Save results
    import json
    results = {
        'mode': 'dinov3_frozen_probe',
        'encoder': model_cfg.get('encoder_name', 'dinov3_vitb16'),
        'head_type': head_type,
        'num_slices': num_slices,
        'probe_depth': probe_depth,
        'best_epoch': best_epoch,
        'best_val_auc': best_val_auc,
        'test_auc': test_auc,
        'test_loss': test_loss,
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'probe_params': sum(p.numel() for p in probe.parameters()),
        'head_params': sum(p.numel() for p in head.parameters()),
        'config': config,
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nResults saved to {output_dir}')
    print(f'  best_val_auc = {best_val_auc:.4f}')
    print(f'  test_auc     = {test_auc:.4f}')

    import numpy as np
    np.savez(os.path.join(output_dir, 'test_predictions.npz'),
             probs=test_probs_cat, labels=test_targets_cat)
    np.savez(os.path.join(output_dir, 'val_predictions.npz'),
             probs=torch.cat(val_probs).numpy(),
             labels=torch.cat(val_targets).numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config YAML')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f'Config: {args.config}')
    print(yaml.dump(config, default_flow_style=False))

    freeze = config['model'].get('freeze_encoder', True)
    if freeze:
        run_frozen_probe(config)
    else:
        # TODO: implement unfrozen fine-tuning with DDP
        # For now, can adapt from src/eval_downstream.py run_patch_finetune()
        raise NotImplementedError(
            'Unfrozen DINOv3 fine-tuning not yet implemented. '
            'Coming soon — will use same DDP pipeline as I-JEPA unfrozen runs.'
        )


if __name__ == '__main__':
    main()
