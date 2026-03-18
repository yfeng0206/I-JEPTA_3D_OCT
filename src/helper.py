"""
Model initialization, optimizer setup, and checkpoint management for I-JEPA.

Provides factory functions for both patch-level and slice-level I-JEPA
encoders/predictors, as well as the frozen ConvNeXt feature extractor
used in the slice-level approach.  Delegates to the model definitions in
``src.models.vision_transformer`` and ``src.models.feature_extractor``.

Compatible with PyTorch 1.13.1 and Python 3.8.
"""

import copy
import os

import torch
import torch.nn as nn

from src.models.vision_transformer import (
    VisionTransformer,
    VisionTransformerPredictor,
    SliceEncoder,
    SlicePredictor,
    VIT_EMBED_DIMS,
    vit_base,
    vit_predictor,
    slice_encoder,
    slice_predictor,
)
from src.models.feature_extractor import FrozenFeatureExtractor
from src.utils.tensors import trunc_normal_
from src.utils.schedulers import WarmupCosineSchedule, CosineWDSchedule


# ---------------------------------------------------------------------------
# ViT model configs (mirrors VIT_EMBED_DIMS for convenience)
# ---------------------------------------------------------------------------

_VIT_CONFIGS = {
    'vit_tiny':  dict(embed_dim=192,  depth=12, num_heads=3),
    'vit_small': dict(embed_dim=384,  depth=12, num_heads=6),
    'vit_base':  dict(embed_dim=768,  depth=12, num_heads=12),
    'vit_large': dict(embed_dim=1024, depth=24, num_heads=16),
    'vit_huge':  dict(embed_dim=1280, depth=32, num_heads=16),
}


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

def init_patch_model(device, patch_size=16, crop_size=256, model_name='vit_base',
                     pred_depth=6, pred_emb_dim=384):
    """Initialize encoder + predictor for patch-level I-JEPA.

    Args:
        device: Target torch device.
        patch_size: Patch size for the ViT.
        crop_size: Input image spatial resolution.
        model_name: One of 'vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'vit_huge'.
        pred_depth: Number of transformer blocks in the predictor.
        pred_emb_dim: Hidden dimension of the predictor.

    Returns:
        (encoder, predictor) tuple on *device*.
    """
    cfg = _VIT_CONFIGS[model_name]

    encoder = VisionTransformer(
        img_size=crop_size,
        patch_size=patch_size,
        embed_dim=cfg['embed_dim'],
        depth=cfg['depth'],
        num_heads=cfg['num_heads'],
    )

    num_patches = encoder.patch_embed.num_patches

    predictor = VisionTransformerPredictor(
        num_patches=num_patches,
        embed_dim=cfg['embed_dim'],
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=cfg['num_heads'],
    )

    encoder = encoder.to(device)
    predictor = predictor.to(device)
    return encoder, predictor


def init_slice_model(device, num_slices=32, embed_dim=768, enc_depth=6,
                     pred_depth=6, pred_emb_dim=384, num_heads=12):
    """Initialize encoder + predictor for slice-level I-JEPA.

    Args:
        device: Target torch device.
        num_slices: Number of slice tokens.
        embed_dim: Embedding dimension.
        enc_depth: Number of transformer blocks in the encoder.
        pred_depth: Number of transformer blocks in the predictor.
        pred_emb_dim: Hidden dimension of the predictor.
        num_heads: Number of attention heads.

    Returns:
        (encoder, predictor) tuple on *device*.
    """
    encoder = SliceEncoder(
        num_slices=num_slices,
        embed_dim=embed_dim,
        depth=enc_depth,
        num_heads=num_heads,
    )

    predictor = SlicePredictor(
        num_slices=num_slices,
        embed_dim=embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=num_heads,
    )

    encoder = encoder.to(device)
    predictor = predictor.to(device)
    return encoder, predictor


def init_feature_extractor(device, checkpoint_path=None):
    """Initialize a frozen ConvNeXt feature extractor for the slice-level approach.

    Args:
        device: Target torch device.
        checkpoint_path: Optional path to SLIViT pretrained ConvNeXt weights.

    Returns:
        FrozenFeatureExtractor on *device*.
    """
    fe = FrozenFeatureExtractor(checkpoint_path=checkpoint_path)
    fe = fe.to(device)
    return fe


def init_opt(encoder, predictor, wd, final_wd, start_lr, ref_lr, final_lr,
             iterations_per_epoch, warmup, num_epochs, ipe_scale=1.0,
             use_bfloat16=False):
    """Initialize AdamW optimizer with warmup cosine LR and cosine WD schedules.

    Creates four parameter groups:
      - encoder params with weight decay
      - predictor params with weight decay
      - encoder params without weight decay (bias, LayerNorm)
      - predictor params without weight decay (bias, LayerNorm)

    Args:
        encoder: The encoder model.
        predictor: The predictor model.
        wd: Reference weight decay.
        final_wd: Final weight decay at end of schedule.
        start_lr: Learning rate at iteration 0.
        ref_lr: Peak learning rate.
        final_lr: Minimum learning rate.
        iterations_per_epoch: Number of training iterations per epoch.
        warmup: Number of warmup epochs.
        num_epochs: Total number of training epochs.
        ipe_scale: Scale factor for iterations per epoch.
        use_bfloat16: Whether to use bfloat16 mixed precision.

    Returns:
        (optimizer, scaler, lr_scheduler, wd_scheduler)
    """
    # Separate parameters that should and should not get weight decay
    enc_wd_params, enc_no_wd_params = _split_wd_params(encoder)
    pred_wd_params, pred_no_wd_params = _split_wd_params(predictor)

    param_groups = [
        {'params': enc_wd_params, 'weight_decay': wd},
        {'params': pred_wd_params, 'weight_decay': wd},
        {'params': enc_no_wd_params, 'weight_decay': 0.0},
        {'params': pred_no_wd_params, 'weight_decay': 0.0},
    ]

    optimizer = torch.optim.AdamW(param_groups)

    ipe = int(iterations_per_epoch * ipe_scale)
    T_max = int(num_epochs * ipe)
    warmup_steps = int(warmup * ipe)

    lr_scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup_steps,
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=T_max,
    )

    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=T_max,
    )

    # GradScaler for mixed precision; disabled if using bfloat16 or CPU
    if use_bfloat16:
        scaler = None
    else:
        scaler = torch.cuda.amp.GradScaler()

    return optimizer, scaler, lr_scheduler, wd_scheduler


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def load_checkpoint(device, r_path, encoder, predictor, target_encoder, opt, scaler):
    """Load a training checkpoint.

    Args:
        device: Device to map tensors to.
        r_path: Path to the checkpoint file.
        encoder: Encoder model (state_dict will be loaded in-place).
        predictor: Predictor model (state_dict will be loaded in-place).
        target_encoder: Target (EMA) encoder model.
        opt: Optimizer.
        scaler: GradScaler (may be None).

    Returns:
        (encoder, predictor, target_encoder, opt, scaler, start_epoch)
    """
    checkpoint = torch.load(r_path, map_location=device)

    encoder.load_state_dict(checkpoint['encoder'])
    predictor.load_state_dict(checkpoint['predictor'])
    target_encoder.load_state_dict(checkpoint['target_encoder'])
    opt.load_state_dict(checkpoint['opt'])

    if scaler is not None and 'scaler' in checkpoint and checkpoint['scaler'] is not None:
        scaler.load_state_dict(checkpoint['scaler'])

    start_epoch = checkpoint.get('epoch', 0)
    print("[Checkpoint] Loaded from %s  (epoch %d)" % (r_path, start_epoch))
    return encoder, predictor, target_encoder, opt, scaler, start_epoch


def save_checkpoint(path, encoder, predictor, target_encoder, optimizer,
                    scaler, epoch, loss, batch_size, world_size, lr):
    """Save a training checkpoint.

    Args:
        path: File path to write.
        encoder: Encoder model (or DDP-wrapped).
        predictor: Predictor model (or DDP-wrapped).
        target_encoder: Target (EMA) encoder.
        optimizer: Optimizer.
        scaler: GradScaler (may be None).
        epoch: Current epoch number.
        loss: Last training loss value.
        batch_size: Per-GPU batch size.
        world_size: Number of distributed processes.
        lr: Current learning rate.
    """
    enc_state = encoder.module.state_dict() if hasattr(encoder, 'module') else encoder.state_dict()
    pred_state = predictor.module.state_dict() if hasattr(predictor, 'module') else predictor.state_dict()
    te_state = target_encoder.module.state_dict() if hasattr(target_encoder, 'module') else target_encoder.state_dict()

    state = {
        'encoder': enc_state,
        'predictor': pred_state,
        'target_encoder': te_state,
        'opt': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'epoch': epoch,
        'loss': loss,
        'batch_size': batch_size,
        'world_size': world_size,
        'lr': lr,
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_wd_params(model):
    """Split model parameters into those that should receive weight decay and those that should not.

    Bias parameters and LayerNorm parameters are excluded from weight decay.

    Returns:
        (wd_params, no_wd_params): Two lists of parameter tensors.
    """
    wd_params = []
    no_wd_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or 'bias' in name or 'norm' in name.lower():
            no_wd_params.append(param)
        else:
            wd_params.append(param)
    return wd_params, no_wd_params
