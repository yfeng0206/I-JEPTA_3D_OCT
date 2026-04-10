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
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import yaml

# Add project root to path for shared code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.eval_downstream import (
    AttentiveProbe, LinearHead, MLPHead, DownstreamModel,
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

    probe = AttentiveProbe(
        num_slices=num_slices,
        embed_dim=embed_dim,
        num_heads=probe_heads,
        depth=probe_depth,
    ).to(device)
    if head_type == 'mlp':
        head = MLPHead(embed_dim, dropout=train_cfg.get('dropout', 0.1)).to(device)
    else:
        head = LinearHead(embed_dim).to(device)

    print(f'\n  Probe: {sum(p.numel() for p in probe.parameters()):,} params (depth={probe_depth})')
    print(f'  Head:  {sum(p.numel() for p in head.parameters()):,} params ({head_type})')

    # Training
    train_ds = TensorDataset(train_feats, train_labels.float())
    val_ds = TensorDataset(val_feats, val_labels.float())
    bs = data_cfg.get('batch_size', 64)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, pin_memory=True)

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
            feats_b = feats_b.to(device, non_blocking=True)
            labels_b = labels_b.to(device, non_blocking=True)

            with autocast():
                cls_out = probe(feats_b)
                logits = head(cls_out).squeeze(-1)
                loss = criterion(logits, labels_b)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item() * labels_b.size(0)
            total_n += labels_b.size(0)
            all_probs.append(torch.sigmoid(logits).detach().cpu())
            all_targets.append(labels_b.cpu())

        train_loss = total_loss / total_n
        train_auc = roc_auc_score(
            torch.cat(all_targets).numpy(), torch.cat(all_probs).numpy())

        # Val
        probe.eval(); head.eval()
        val_loss_sum = 0; val_n = 0
        val_probs = []; val_targets = []
        with torch.no_grad():
            for feats_b, labels_b in val_loader:
                feats_b = feats_b.to(device, non_blocking=True)
                labels_b = labels_b.to(device, non_blocking=True)
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
    test_ds = TensorDataset(test_feats, test_labels.float())
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, pin_memory=True)
    test_probs = []; test_targets = []; test_loss_sum = 0; test_n = 0
    with torch.no_grad():
        for feats_b, labels_b in test_loader:
            feats_b = feats_b.to(device, non_blocking=True)
            labels_b = labels_b.to(device, non_blocking=True)
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

    np.savez(os.path.join(output_dir, 'test_predictions.npz'),
             probs=test_probs_cat, labels=test_targets_cat)
    np.savez(os.path.join(output_dir, 'val_predictions.npz'),
             probs=torch.cat(val_probs).numpy(),
             labels=torch.cat(val_targets).numpy())


def run_unfrozen_finetune(config):
    """Unfrozen DINOv3 + AttentiveProbe + Head, DDP fine-tuning.

    Uses the same DownstreamModel wrapper as I-JEPA unfrozen runs
    for identical training dynamics.
    """
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler

    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    output_dir = config['logging']['output_dir']

    # DDP setup
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group('nccl')
    else:
        rank, world_size, local_rank = 0, 1, 0

    is_main = (rank == 0)
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    if is_main:
        os.makedirs(output_dir, exist_ok=True)
        print(f'GPU: {torch.cuda.get_device_name(device)}')
        print('=' * 70)
        print('DINOv3 Ablation — Unfrozen Fine-tuning (DDP)')
        print(f'  World size: {world_size}')
        print('=' * 70)

    # Load encoder
    raw_model, embed_dim = load_dinov3_encoder(
        model_cfg.get('encoder_name', 'dinov3_vitb16'),
        model_cfg.get('weights_path'),
    )
    encoder = DINOv3Encoder(raw_model, embed_dim)

    num_slices = data_cfg['num_slices']
    probe_depth = model_cfg.get('probe_depth', 3)
    probe_heads = model_cfg.get('probe_num_heads', 12)
    head_type = model_cfg.get('head_type', 'mlp')

    probe = AttentiveProbe(
        num_slices=num_slices, embed_dim=embed_dim,
        num_heads=probe_heads, depth=probe_depth,
    )
    if head_type == 'mlp':
        head = MLPHead(in_dim=embed_dim, dropout=train_cfg.get('dropout', 0.1))
    else:
        head = LinearHead(in_dim=embed_dim)

    chunk_size = data_cfg.get('encode_chunk_size', 25)
    model = DownstreamModel(encoder, probe, head, chunk_size).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    raw = model.module if hasattr(model, 'module') else model

    if is_main:
        enc_params = sum(p.numel() for p in raw.encoder.parameters())
        probe_params = sum(p.numel() for p in raw.probe.parameters())
        head_params = sum(p.numel() for p in raw.head.parameters())
        print(f'  Encoder:  {enc_params:,} params (trainable, lr={train_cfg.get("lr_encoder", 5e-6):.1e})')
        print(f'  Probe:    {probe_params:,} params (trainable, lr={train_cfg.get("lr_probe", 1e-4):.1e})')
        print(f'  Head:     {head_params:,} params (trainable, lr={train_cfg.get("lr_head", 1e-3):.1e})')

    # Datasets
    slice_size = data_cfg.get('slice_size', 256)
    batch_size = data_cfg.get('batch_size', 1)
    accum_steps = train_cfg.get('accum_steps', 4)

    train_dataset = OCTVolumeDataset(
        os.path.join(data_cfg['data_dir'], 'Training'),
        num_slices=num_slices, slice_size=slice_size, return_label=True)
    val_dataset = OCTVolumeDataset(
        os.path.join(data_cfg['data_dir'], 'Validation'),
        num_slices=num_slices, slice_size=slice_size, return_label=True)
    test_dataset = OCTVolumeDataset(
        os.path.join(data_cfg['data_dir'], 'Test'),
        num_slices=num_slices, slice_size=slice_size, return_label=True)

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
        print(f'  Train: {len(train_dataset)} volumes  (bs={batch_size} x {world_size} GPUs x {accum_steps} accum = {eff_batch} eff)')
        print(f'  Val:   {len(val_dataset)} volumes')

    # Optimizer (same param groups as I-JEPA unfrozen)
    param_groups = [
        {'params': raw.encoder.parameters(), 'lr': train_cfg.get('lr_encoder', 5e-6)},
        {'params': raw.probe.parameters(), 'lr': train_cfg.get('lr_probe', 1e-4)},
        {'params': raw.head.parameters(), 'lr': train_cfg.get('lr_head', 1e-3)},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=train_cfg.get('weight_decay', 0.01))
    steps_per_epoch = len(train_loader) // accum_steps
    scheduler = cosine_schedule_with_warmup(
        optimizer, train_cfg.get('warmup_epochs', 3),
        train_cfg['epochs'], max(steps_per_epoch, 1))
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    # Training
    best_auc = 0.0
    patience_counter = 0
    patience = train_cfg.get('patience', 5)
    epochs = train_cfg['epochs']
    log_rows = []

    if is_main:
        print(f'\n--- Training ({epochs} epochs, patience={patience}) ---')

    for epoch in range(1, epochs + 1):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0; n_samples = 0
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()

        for itr, (volumes, labels) in enumerate(train_loader):
            volumes = volumes.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            with autocast():
                logits = model(volumes)
                loss = criterion(logits, labels) / accum_steps

            scaler.scale(loss).backward()
            total_loss += loss.item() * accum_steps * labels.size(0)
            n_samples += labels.size(0)

            if (itr + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        train_loss = total_loss / max(n_samples, 1)

        # Val
        model.eval()
        val_probs = []; val_labels_list = []
        with torch.no_grad():
            for volumes, labels in val_loader:
                volumes = volumes.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).float()
                with autocast():
                    logits = model(volumes)
                val_probs.append(torch.sigmoid(logits).cpu())
                val_labels_list.append(labels.cpu())

        # Gather across ranks
        if world_size > 1:
            all_probs = [None] * world_size
            all_labels = [None] * world_size
            dist.all_gather_object(all_probs, torch.cat(val_probs).numpy().tolist())
            dist.all_gather_object(all_labels, torch.cat(val_labels_list).numpy().tolist())
            if is_main:
                val_probs_np = np.array([p for sublist in all_probs for p in sublist])
                val_labels_np = np.array([l for sublist in all_labels for l in sublist])
                val_auc = roc_auc_score(val_labels_np, val_probs_np)
            else:
                val_auc = 0.0
        else:
            val_probs_np = torch.cat(val_probs).numpy()
            val_labels_np = torch.cat(val_labels_list).numpy()
            val_auc = roc_auc_score(val_labels_np, val_probs_np)

        elapsed = time.time() - t0
        lr_enc = optimizer.param_groups[0]['lr']
        lr_probe = optimizer.param_groups[1]['lr']

        # Early stopping (broadcast from rank 0)
        should_stop = False
        if is_main:
            improved = val_auc > best_auc
            if improved:
                best_auc = val_auc
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    'model': raw.state_dict(),
                    'epoch': epoch, 'val_auc': val_auc,
                }, os.path.join(output_dir, 'best_model.pt'))
            else:
                patience_counter += 1
            if patience_counter >= patience:
                should_stop = True

            marker = ' *' if improved else ''
            print(f'Epoch {epoch:3d}/{epochs} ({elapsed:.0f}s) | '
                  f'Train: {train_loss:.4f} | Val AUC: {val_auc:.4f} | '
                  f'LR: {lr_enc:.1e}/{lr_probe:.1e}{marker}')

            log_rows.append({
                'epoch': epoch, 'train_loss': f'{train_loss:.6f}',
                'val_loss': '0', 'val_auc': f'{val_auc:.6f}',
                'lr_enc': f'{lr_enc:.8f}', 'lr_probe': f'{lr_probe:.8f}',
                'elapsed_s': f'{elapsed:.1f}',
            })

        if world_size > 1:
            stop_tensor = torch.tensor([1 if should_stop else 0], device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = stop_tensor.item() == 1
        if should_stop:
            if is_main:
                print(f'Early stopping at epoch {epoch} (patience={patience})')
            break

    # Test evaluation (rank 0 only)
    if is_main:
        print(f'\n--- Test Evaluation ---')
        ckpt = torch.load(os.path.join(output_dir, 'best_model.pt'), map_location=device)
        raw.load_state_dict(ckpt['model'])
        model.eval()

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=2, pin_memory=True)
        test_probs = []; test_targets = []
        with torch.no_grad():
            for volumes, labels in test_loader:
                volumes = volumes.to(device, non_blocking=True)
                with autocast():
                    logits = model(volumes)
                test_probs.append(torch.sigmoid(logits).cpu())
                test_targets.append(labels)

        test_probs_np = torch.cat(test_probs).numpy()
        test_targets_np = torch.cat(test_targets).numpy()
        test_auc = roc_auc_score(test_targets_np, test_probs_np)

        preds = (test_probs_np > 0.5).astype(int)
        sensitivity = (preds[test_targets_np == 1] == 1).mean()
        specificity = (preds[test_targets_np == 0] == 0).mean()

        print(f'Best epoch: {best_epoch}  |  Val AUC: {best_auc:.4f}  |  TEST AUC: {test_auc:.4f}')

        # Save results
        results = {
            'mode': 'dinov3_unfrozen',
            'encoder': model_cfg.get('encoder_name', 'dinov3_vitb16'),
            'head_type': head_type, 'num_slices': num_slices,
            'probe_depth': probe_depth, 'best_epoch': best_epoch,
            'best_val_auc': best_auc, 'test_auc': test_auc,
            'lr_encoder': train_cfg.get('lr_encoder', 5e-6),
            'accum_steps': accum_steps, 'effective_batch': eff_batch,
            'sensitivity': float(sensitivity), 'specificity': float(specificity),
            'config': config,
        }
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        with open(os.path.join(output_dir, 'train_log.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)

        np.savez(os.path.join(output_dir, 'test_predictions.npz'),
                 probs=test_probs_np, labels=test_targets_np)

        print(f'\nResults saved to {output_dir}')
        print(f'  best_val_auc = {best_auc:.4f}')
        print(f'  test_auc     = {test_auc:.4f}')

    # DDP cleanup
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


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
        run_unfrozen_finetune(config)


if __name__ == '__main__':
    main()
