"""
Patch-level I-JEPA pretraining on OCT slices.

Each OCT volume is sliced into individual 2-D images (256x256), which are
treated as independent samples for standard I-JEPA training with 2-D block
masking on the 16x16 patch grid.

Usage:
    torchrun --nproc_per_node=4 train_patch.py --config configs/patch_vitb16_ep100.yaml

Compatible with PyTorch 1.13.1 and Python 3.8.
"""

import argparse
import copy
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

import yaml

# Ensure the project root is on the path so `src.*` imports work when
# invoked as ``python src/train_patch.py`` or via torchrun.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.helper import init_patch_model, init_opt, load_checkpoint, save_checkpoint
from src.masks.multiblock import MaskCollator
from src.masks.utils import apply_masks
from src.datasets.oct_slices import OCTSliceDataset
from src.transforms import make_transforms
from src.utils.distributed import init_distributed
from src.utils.logging import CSVLogger, AverageMeter, gpu_timer
from src.utils.tensors import repeat_interleave_batch


# ---------------------------------------------------------------------------
# Blob upload helper (best-effort, non-fatal)
# ---------------------------------------------------------------------------

import threading

_upload_threads = []  # Track background upload threads


def _upload_worker(local_path, blob_prefix, log_fn):
    """Background worker for blob upload."""
    try:
        from azure.identity import ManagedIdentityCredential
        from azure.storage.blob import ContainerClient
        account = os.environ.get('BLOB_ACCOUNT', 'STORAGE_ACCOUNT_REDACTED')
        container_name = os.environ.get(
            'BLOB_CONTAINER',
            'CONTAINER_REDACTED',
        )
        cred = ManagedIdentityCredential()
        container = ContainerClient(
            account_url='https://%s.blob.core.windows.net' % account,
            container_name=container_name,
            credential=cred,
        )
        fname = os.path.basename(local_path)
        blob_name = '%s/%s' % (blob_prefix, fname)
        size = os.path.getsize(local_path)
        log_fn('  Uploading %s (%s bytes) -> %s' % (fname, format(size, ','), blob_name))
        with open(local_path, 'rb') as f:
            container.upload_blob(blob_name, f, overwrite=True)
        log_fn('  Upload OK: %s' % fname)
    except Exception as e:
        log_fn('  Upload skipped: %s' % e)


def upload_to_blob(local_path, blob_prefix, log_fn=print, blocking=False):
    """Upload a file to Azure Blob Storage in a background thread.

    Non-blocking by default so DDP rank 0 is not held up while other
    ranks wait at the next collective.  Use blocking=True for final
    uploads where we need to ensure completion before exit.
    """
    if blocking:
        _upload_worker(local_path, blob_prefix, log_fn)
    else:
        t = threading.Thread(
            target=_upload_worker,
            args=(local_path, blob_prefix, log_fn),
            daemon=True,
        )
        t.start()
        _upload_threads.append(t)


# ---------------------------------------------------------------------------
# Momentum scheduler for EMA
# ---------------------------------------------------------------------------

def momentum_schedule(base_value, final_value, num_steps):
    """Yield a cosine schedule from base_value to final_value over num_steps."""
    for step in range(num_steps):
        progress = step / max(1, num_steps - 1)
        value = final_value - (final_value - base_value) * (
            math.cos(math.pi * progress) + 1.0
        ) / 2.0
        yield value


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # ---- Load config -------------------------------------------------------
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    data_cfg = config['data']
    mask_cfg = config['mask']
    meta_cfg = config['meta']
    opt_cfg = config['optimization']
    log_cfg = config['logging']

    # ---- Distributed setup -------------------------------------------------
    world_size, rank = init_distributed()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')

    is_main = (rank == 0)

    # ---- Output directory --------------------------------------------------
    output_dir = log_cfg['folder']
    write_tag = log_cfg['write_tag']
    # Blob prefix for periodic uploads (derive from output_dir basename)
    blob_prefix = 'ijepa-results/%s' % os.path.basename(output_dir)
    if is_main:
        os.makedirs(output_dir, exist_ok=True)

    # ---- Logging -----------------------------------------------------------
    csv_logger = None
    if is_main:
        csv_path = os.path.join(output_dir, '%s-log.csv' % write_tag)
        csv_logger = CSVLogger(
            csv_path,
            'epoch', 'iteration', 'loss', 'lr', 'wd', 'ema',
            'data_time_ms', 'forward_time_ms', 'backward_time_ms',
            'gpu_mem_mb',
        )

    def log(msg):
        if is_main:
            print(msg, flush=True)

    log('=' * 70)
    log('Patch-level I-JEPA Pretraining')
    log('  Config:     %s' % args.config)
    log('  World size: %d' % world_size)
    log('  Device:     %s' % device)
    if torch.cuda.is_available():
        log('  GPU:        %s' % torch.cuda.get_device_name(device))
        log('  GPU memory: %.1f GB' % (torch.cuda.get_device_properties(device).total_memory / 1e9))
    log('=' * 70)

    # ---- Model -------------------------------------------------------------
    encoder, predictor = init_patch_model(
        device=device,
        patch_size=mask_cfg['patch_size'],
        crop_size=data_cfg['crop_size'],
        model_name=meta_cfg['model_name'],
        pred_depth=meta_cfg['pred_depth'],
        pred_emb_dim=meta_cfg['pred_emb_dim'],
    )

    # Target encoder (EMA copy)
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    enc_params = sum(p.numel() for p in encoder.parameters())
    pred_params = sum(p.numel() for p in predictor.parameters())
    log('  Encoder params:   %s' % format(enc_params, ','))
    log('  Predictor params: %s' % format(pred_params, ','))

    # ---- Mask collator -----------------------------------------------------
    crop_size = data_cfg['crop_size']
    mask_collator = MaskCollator(
        input_size=(crop_size, crop_size),
        patch_size=mask_cfg['patch_size'],
        enc_mask_scale=tuple(mask_cfg['enc_mask_scale']),
        pred_mask_scale=tuple(mask_cfg['pred_mask_scale']),
        aspect_ratio=tuple(mask_cfg['aspect_ratio']),
        nenc=mask_cfg['num_enc_masks'],
        npred=mask_cfg['num_pred_masks'],
        min_keep=mask_cfg['min_keep'],
        allow_overlap=mask_cfg['allow_overlap'],
    )

    # ---- Transforms --------------------------------------------------------
    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=tuple(data_cfg.get('crop_scale', (0.3, 1.0))),
        gaussian_blur=data_cfg.get('use_gaussian_blur', False),
        horizontal_flip=data_cfg.get('use_horizontal_flip', False),
        color_distortion=data_cfg.get('use_color_distortion', False),
        color_jitter=data_cfg.get('color_jitter_strength', 0.0),
    )

    # ---- Dataset -----------------------------------------------------------
    data_dir = data_cfg['data_dir']
    num_slices = data_cfg.get('num_slices', 32)
    log('Loading dataset from %s ...' % data_dir)

    # Training set
    train_dir = os.path.join(data_dir, 'Training')
    if not os.path.isdir(train_dir):
        raise FileNotFoundError("No Training split found under %s" % data_dir)
    train_dataset = OCTSliceDataset(
        data_dir=train_dir, num_slices=num_slices,
        slice_size=crop_size, transform=transform,
    )
    log('  Training: %d slices (%d volumes)' % (len(train_dataset), len(train_dataset.file_paths)))

    # Validation set (for val loss tracking)
    val_dir = os.path.join(data_dir, 'Validation')
    val_dataset = None
    if os.path.isdir(val_dir):
        val_dataset = OCTSliceDataset(
            data_dir=val_dir, num_slices=num_slices,
            slice_size=crop_size, transform=transform,
        )
        log('  Validation: %d slices (%d volumes)' % (len(val_dataset), len(val_dataset.file_paths)))

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg['batch_size'],
        sampler=train_sampler,
        num_workers=data_cfg['num_workers'],
        pin_memory=data_cfg.get('pin_mem', True),
        drop_last=True,
        collate_fn=mask_collator,
    )

    val_loader = None
    if val_dataset is not None:
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=data_cfg['batch_size'],
            sampler=val_sampler,
            num_workers=data_cfg['num_workers'],
            pin_memory=data_cfg.get('pin_mem', True),
            drop_last=False,
            collate_fn=mask_collator,
        )

    accum_steps = opt_cfg.get('accum_steps', 1)
    iterations_per_epoch = len(train_loader) // accum_steps
    log('  Batches per epoch: %d (%d iters x %d accum)' % (len(train_loader), iterations_per_epoch, accum_steps))
    log('  Effective batch size: %d' % (data_cfg['batch_size'] * world_size * accum_steps))

    # ---- Optimizer ---------------------------------------------------------
    optimizer, scaler, lr_scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=opt_cfg['weight_decay'],
        final_wd=opt_cfg['final_weight_decay'],
        start_lr=opt_cfg['start_lr'],
        ref_lr=opt_cfg['lr'],
        final_lr=opt_cfg['final_lr'],
        iterations_per_epoch=iterations_per_epoch,
        warmup=opt_cfg['warmup'],
        num_epochs=opt_cfg['epochs'],
        ipe_scale=opt_cfg.get('ipe_scale', 1.0),
        use_bfloat16=meta_cfg.get('use_bfloat16', False),
    )

    # ---- DDP wrap ----------------------------------------------------------
    if world_size > 1:
        encoder = DDP(encoder, device_ids=[local_rank])
        predictor = DDP(predictor, device_ids=[local_rank])

    # ---- Load checkpoint ---------------------------------------------------
    start_epoch = 0
    if meta_cfg.get('load_checkpoint', False) and meta_cfg.get('read_checkpoint'):
        r_path = meta_cfg['read_checkpoint']
        enc_unwrap = encoder.module if hasattr(encoder, 'module') else encoder
        pred_unwrap = predictor.module if hasattr(predictor, 'module') else predictor
        enc_unwrap, pred_unwrap, target_encoder, optimizer, scaler, start_epoch = \
            load_checkpoint(device, r_path, enc_unwrap, pred_unwrap,
                            target_encoder, optimizer, scaler)

    # ---- Momentum schedule for EMA -----------------------------------------
    ema_start, ema_end = opt_cfg['ema']
    total_steps = opt_cfg['epochs'] * iterations_per_epoch
    mom_schedule = momentum_schedule(ema_start, ema_end, total_steps)

    # Fast-forward momentum schedule if resuming
    for _ in range(start_epoch * iterations_per_epoch):
        next(mom_schedule)

    # ---- Val loss evaluation function --------------------------------------
    @torch.no_grad()
    def evaluate_val():
        if val_loader is None:
            return None
        enc_unwrap = encoder.module if hasattr(encoder, 'module') else encoder
        pred_unwrap = predictor.module if hasattr(predictor, 'module') else predictor
        enc_unwrap.eval()
        pred_unwrap.eval()
        val_loss_meter = AverageMeter()
        for imgs_v, masks_enc_v, masks_pred_v in val_loader:
            imgs_v = imgs_v.to(device, non_blocking=True)
            masks_enc_v = [m.to(device, non_blocking=True) for m in masks_enc_v]
            masks_pred_v = [m.to(device, non_blocking=True) for m in masks_pred_v]
            B_v = imgs_v.size(0)
            h = target_encoder(imgs_v)
            h = F.layer_norm(h, (h.size(-1),))
            h = apply_masks(h, masks_pred_v)
            h = repeat_interleave_batch(h, B_v, repeat=len(masks_enc_v))
            z = enc_unwrap(imgs_v, masks_enc_v)
            z = pred_unwrap(z, masks_enc_v, masks_pred_v)
            loss = F.smooth_l1_loss(z, h)
            val_loss_meter.update(loss.item())
        enc_unwrap.train()
        pred_unwrap.train()
        return val_loss_meter.avg

    # ---- Training loop -----------------------------------------------------
    patience = opt_cfg.get('patience', 8)
    warmup_epochs = opt_cfg.get('warmup', 5)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    log('-' * 70)
    log('Starting training from epoch %d to %d (patience=%d, early-stop after epoch %d)'
        % (start_epoch + 1, opt_cfg['epochs'], patience, warmup_epochs))
    log('-' * 70)

    # Track last valid scheduler/EMA values for logging
    lr_val = opt_cfg.get('start_lr', 0.0001)
    wd_val = opt_cfg.get('weight_decay', 0.04)
    m = ema_start
    num_micro_batches = len(train_loader)

    for epoch in range(start_epoch, opt_cfg['epochs']):
        train_sampler.set_epoch(epoch)
        loss_meter = AverageMeter()

        t_epoch_start = time.time()

        for itr, (imgs, masks_enc, masks_pred) in enumerate(train_loader):
            t_data = time.time()

            # Move to device
            imgs = imgs.to(device, non_blocking=True)
            masks_enc = [m_t.to(device, non_blocking=True) for m_t in masks_enc]
            masks_pred = [m_t.to(device, non_blocking=True) for m_t in masks_pred]

            B = imgs.size(0)

            data_ms = (time.time() - t_data) * 1000.0

            def _forward_backward():
                # Only zero gradients at the start of an accumulation window
                if itr % accum_steps == 0:
                    optimizer.zero_grad(set_to_none=True)

                # Target path (no gradient)
                with torch.no_grad():
                    h = target_encoder(imgs)  # (B, N, D) full patch features
                    h = F.layer_norm(h, (h.size(-1),))
                    # Extract target features at predicted positions
                    h = apply_masks(h, masks_pred)
                    # Repeat for each encoder mask
                    h = repeat_interleave_batch(h, B, repeat=len(masks_enc))

                # Context path (with gradient)
                use_amp = (scaler is not None)
                if use_amp:
                    with autocast():
                        z = encoder(imgs, masks_enc)  # masked context tokens
                        z = predictor(z, masks_enc, masks_pred)  # predict targets
                        loss = F.smooth_l1_loss(z, h) / accum_steps
                else:
                    z = encoder(imgs, masks_enc)
                    z = predictor(z, masks_enc, masks_pred)
                    loss = F.smooth_l1_loss(z, h) / accum_steps

                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Step optimizer only at the end of accumulation window
                if (itr + 1) % accum_steps == 0 or (itr + 1) == len(train_loader):
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                return loss.item() * accum_steps  # report unscaled loss

            (loss_val, fwd_bwd_ms) = gpu_timer(_forward_backward)

            # Scheduler + EMA only on optimizer step iterations
            is_step = (itr + 1) % accum_steps == 0 or (itr + 1) == num_micro_batches
            if is_step:
                lr_val = lr_scheduler.step()
                wd_val = wd_scheduler.step()
                m = next(mom_schedule)
                enc_unwrap = encoder.module if hasattr(encoder, 'module') else encoder
                with torch.no_grad():
                    for p_online, p_target in zip(enc_unwrap.parameters(),
                                                  target_encoder.parameters()):
                        p_target.data.mul_(m).add_((1.0 - m) * p_online.detach().data)

            loss_meter.update(loss_val)

            # GPU memory
            gpu_mem_mb = 0.0
            if torch.cuda.is_available():
                gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

            # Log every 50 iterations
            if is_main and (itr + 1) % 50 == 0:
                log('  [Epoch %d/%d | Iter %d/%d] loss=%.4f  lr=%.2e  wd=%.4f  '
                    'ema=%.5f  gpu=%.0fMB'
                    % (epoch + 1, opt_cfg['epochs'], itr + 1, num_micro_batches,
                       loss_val, lr_val, wd_val, m, gpu_mem_mb))

            # CSV log
            if csv_logger is not None:
                csv_logger.log(
                    epoch + 1, itr + 1, loss_val, lr_val, wd_val, m,
                    data_ms, fwd_bwd_ms, 0.0, gpu_mem_mb,
                )

        # End-of-epoch diagnostic: prediction quality on one batch
        if is_main and val_loader is not None:
            enc_diag = encoder.module if hasattr(encoder, 'module') else encoder
            pred_diag = predictor.module if hasattr(predictor, 'module') else predictor
            enc_diag.eval()
            pred_diag.eval()
            with torch.no_grad():
                for diag_imgs, diag_menc, diag_mpred in val_loader:
                    diag_imgs = diag_imgs.to(device)
                    diag_menc = [m.to(device) for m in diag_menc]
                    diag_mpred = [m.to(device) for m in diag_mpred]
                    B_d = diag_imgs.size(0)

                    # Target
                    h_d = target_encoder(diag_imgs)
                    h_d = F.layer_norm(h_d, (h_d.size(-1),))
                    h_masked = apply_masks(h_d, diag_mpred)
                    h_masked = repeat_interleave_batch(h_masked, B_d, repeat=len(diag_menc))

                    # Prediction
                    z_d = enc_diag(diag_imgs, diag_menc)
                    z_d = pred_diag(z_d, diag_menc, diag_mpred)

                    # Cosine similarity
                    cos_sim = F.cosine_similarity(z_d, h_masked, dim=-1)
                    avg_cos = cos_sim.mean().item()
                    min_cos = cos_sim.min().item()
                    max_cos = cos_sim.max().item()

                    # L2 distance
                    l2_dist = (z_d - h_masked).norm(dim=-1).mean().item()

                    # Representation diversity: pairwise cosine across all patches (first sample)
                    num_patches = h_d.size(1)
                    all_reps = F.normalize(h_d[0], dim=-1)
                    pairwise = all_reps @ all_reps.T
                    mask_diag = ~torch.eye(num_patches, dtype=torch.bool, device=device)
                    avg_pairwise = pairwise[mask_diag].mean().item()

                    log('  [DIAG] Epoch %d: cos_sim=%.4f (min=%.4f max=%.4f) '
                        'l2_dist=%.4f  rep_diversity=%.4f (1.0=collapsed)'
                        % (epoch + 1, avg_cos, min_cos, max_cos, l2_dist, avg_pairwise))
                    log('  [DIAG] z shape=%s h shape=%s num_patches=%d'
                        % (str(z_d.shape), str(h_masked.shape), num_patches))
                    break
            enc_diag.train()
            pred_diag.train()

        # Train eval loss (20 batches, no grad)
        if is_main and val_loader is not None:
            enc_u = encoder.module if hasattr(encoder, 'module') else encoder
            pred_u = predictor.module if hasattr(predictor, 'module') else predictor
            enc_u.eval()
            pred_u.eval()
            train_eval_meter = AverageMeter()
            with torch.no_grad():
                for t_idx, (t_imgs, t_menc, t_mpred) in enumerate(train_loader):
                    if t_idx >= 20:
                        break
                    t_imgs = t_imgs.to(device)
                    t_menc = [m.to(device) for m in t_menc]
                    t_mpred = [m.to(device) for m in t_mpred]
                    B_t = t_imgs.size(0)
                    h_t = target_encoder(t_imgs)
                    h_t = F.layer_norm(h_t, (h_t.size(-1),))
                    h_t = apply_masks(h_t, t_mpred)
                    h_t = repeat_interleave_batch(h_t, B_t, repeat=len(t_menc))
                    z_t = enc_u(t_imgs, t_menc)
                    z_t = pred_u(z_t, t_menc, t_mpred)
                    train_eval_meter.update(F.smooth_l1_loss(z_t, h_t).item())
            if is_main:
                log('  train_eval_loss=%.6f' % train_eval_meter.avg)
            enc_u.train()
            pred_u.train()

        # End-of-epoch summary
        epoch_time = time.time() - t_epoch_start

        # Validation loss
        val_loss = evaluate_val()
        val_str = '  val_loss=%.4f' % val_loss if val_loss is not None else ''
        improved = ''

        if val_loss is not None:
            # Only track best / early-stop after warmup to avoid the
            # artificially-low epoch-1 loss (EMA target hasn't diverged yet)
            past_warmup = (epoch + 1) > warmup_epochs
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if past_warmup:
                    epochs_no_improve = 0
                improved = ' *'
                if is_main:
                    best_path = os.path.join(output_dir, '%s-best.pth.tar' % write_tag)
                    save_checkpoint(
                        best_path, encoder, predictor, target_encoder, optimizer,
                        scaler, epoch + 1, val_loss, data_cfg['batch_size'],
                        world_size, lr_val,
                    )
                    # Upload best checkpoint to blob immediately
                    upload_to_blob(best_path, blob_prefix, log)
            else:
                if past_warmup:
                    epochs_no_improve += 1

        log('Epoch %d/%d  (%.0fs)  train_loss=%.4f%s%s'
            % (epoch + 1, opt_cfg['epochs'], epoch_time, loss_meter.avg,
               val_str, improved))

        # Save latest checkpoint (main process only)
        if is_main:
            latest_path = os.path.join(output_dir, '%s-latest.pth.tar' % write_tag)
            save_checkpoint(
                latest_path, encoder, predictor, target_encoder, optimizer,
                scaler, epoch + 1, loss_meter.avg, data_cfg['batch_size'],
                world_size, lr_val,
            )
            if (epoch + 1) % 25 == 0:
                ep_path = os.path.join(output_dir, '%s-ep%d.pth.tar' % (write_tag, epoch + 1))
                save_checkpoint(
                    ep_path, encoder, predictor, target_encoder, optimizer,
                    scaler, epoch + 1, loss_meter.avg, data_cfg['batch_size'],
                    world_size, lr_val,
                )
            # Upload log CSV every 5 epochs
            if (epoch + 1) % 5 == 0:
                csv_file = os.path.join(output_dir, '%s-log.csv' % write_tag)
                if os.path.exists(csv_file):
                    upload_to_blob(csv_file, blob_prefix, log)

        # Early stopping (only active after warmup)
        if val_loss is not None and past_warmup and epochs_no_improve >= patience:
            log('Early stopping: val loss has not improved for %d epochs (best=%.4f)'
                % (patience, best_val_loss))
            break

    log('=' * 70)
    log('Training complete. Best val loss: %.4f' % best_val_loss)
    log('=' * 70)

    # Wait for any background uploads to finish, then do final uploads
    if is_main:
        for t in _upload_threads:
            t.join(timeout=300)  # Wait up to 5 min per thread
        for fname in ['%s-log.csv' % write_tag, 'config.yaml']:
            fpath = os.path.join(output_dir, fname)
            if os.path.exists(fpath):
                upload_to_blob(fpath, blob_prefix, log, blocking=True)

    # Clean DDP shutdown so torchrun exits cleanly
    import torch.distributed as dist
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Patch-level I-JEPA pretraining')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    args = parser.parse_args()
    main(args)
