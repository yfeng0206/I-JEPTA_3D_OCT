"""
Download ImageNet-pretrained ViT-B/16 and convert to our I-JEPA format.

Downloads weights from timm (or HuggingFace) and saves only the
transferable parameters (patch_embed + transformer blocks + final norm).
Skips cls_token, pos_embed (ours is sinusoidal), and classification head.

Usage:
    pip install timm
    python scripts/download_imagenet_vit.py [--output checkpoints/vit_b16_imagenet.pth]
    python scripts/download_imagenet_vit.py --source dino  # use DINO self-supervised weights

The output checkpoint can be passed to train_patch.py via:
    meta.pretrained_encoder: checkpoints/vit_b16_imagenet.pth
"""

import argparse
import os
import sys

import torch


def download_timm_vit(model_name='vit_base_patch16_224'):
    """Download ViT-B/16 from timm with ImageNet-1K pretrained weights."""
    try:
        import timm
    except ImportError:
        print('Installing timm...')
        os.system('%s -m pip install timm' % sys.executable)
        import timm

    print('Downloading %s from timm...' % model_name)
    model = timm.create_model(model_name, pretrained=True)
    return model.state_dict()


def download_dino_vit():
    """Download DINO ViT-B/16 (self-supervised, from Meta)."""
    print('Downloading DINO ViT-B/16 from torch hub...')
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    return model.state_dict()


def download_mae_vit():
    """Download MAE ViT-B/16 (self-supervised, from Meta)."""
    print('Downloading MAE ViT-B/16...')
    url = 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth'
    ckpt = torch.hub.load_state_dict_from_url(url, map_location='cpu')
    return ckpt['model']


def convert_to_ijepa_format(src_state_dict, source='timm'):
    """Extract transferable weights, mapping to our ViT key names.

    Our ViT (src/models/vision_transformer.py) uses these keys:
        patch_embed.proj.weight/bias
        blocks.{i}.norm1.weight/bias
        blocks.{i}.attn.qkv.weight/bias
        blocks.{i}.attn.proj.weight/bias
        blocks.{i}.norm2.weight/bias
        blocks.{i}.mlp.fc1.weight/bias
        blocks.{i}.mlp.fc2.weight/bias
        norm.weight/bias

    We skip: cls_token, pos_embed, head.*, pre_logits.*
    """
    # Keys to skip (not transferable or not in our architecture)
    skip_prefixes = ('cls_token', 'pos_embed', 'head.', 'pre_logits.',
                     'fc_norm.', 'norm_pre.', 'dist_token')

    transferred = {}
    skipped = []

    for key, value in src_state_dict.items():
        if any(key.startswith(p) for p in skip_prefixes):
            skipped.append(key)
            continue

        # MAE uses different key prefix
        if source == 'mae':
            # MAE keys might have different structure, handle if needed
            pass

        # DINO keys match timm format
        transferred[key] = value

    return transferred, skipped


def main():
    parser = argparse.ArgumentParser(
        description='Download ImageNet ViT-B/16 for I-JEPA encoder init')
    parser.add_argument('--source', type=str, default='timm',
                        choices=['timm', 'dino', 'mae'],
                        help='Weight source: timm (supervised), dino (self-supervised), mae')
    parser.add_argument('--output', type=str,
                        default='checkpoints/vit_b16_imagenet.pth',
                        help='Output path for converted checkpoint')
    args = parser.parse_args()

    # Download
    if args.source == 'timm':
        src_sd = download_timm_vit()
    elif args.source == 'dino':
        src_sd = download_dino_vit()
    elif args.source == 'mae':
        src_sd = download_mae_vit()
    else:
        raise ValueError('Unknown source: %s' % args.source)

    print('  Source state dict: %d keys' % len(src_sd))

    # Convert
    transferred, skipped = convert_to_ijepa_format(src_sd, args.source)
    print('  Transferred: %d keys' % len(transferred))
    print('  Skipped: %s' % ', '.join(skipped))

    # Verify shapes match ViT-B/16
    expected_shapes = {
        'patch_embed.proj.weight': (768, 3, 16, 16),
        'patch_embed.proj.bias': (768,),
        'blocks.0.attn.qkv.weight': (2304, 768),
        'blocks.11.mlp.fc2.bias': (768,),
        'norm.weight': (768,),
    }
    for key, expected in expected_shapes.items():
        if key in transferred:
            actual = tuple(transferred[key].shape)
            status = 'OK' if actual == expected else 'MISMATCH (got %s)' % str(actual)
            print('  %s: %s -> %s' % (key, str(expected), status))

    # Count params
    total_params = sum(v.numel() for v in transferred.values())
    print('  Total params: %s' % format(total_params, ','))

    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save({
        'encoder': transferred,
        'source': args.source,
        'num_keys': len(transferred),
        'num_params': total_params,
    }, args.output)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print('\nSaved to %s (%.1f MB)' % (args.output, size_mb))
    print('Use with train_patch.py: meta.pretrained_encoder: %s' % args.output)


if __name__ == '__main__':
    main()
