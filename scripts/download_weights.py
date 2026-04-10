"""Download pretrained model weights from HuggingFace Hub.

Usage:
    python scripts/download_weights.py --all          # Download everything
    python scripts/download_weights.py --encoder      # Just the best encoder (ep32)
    python scripts/download_weights.py --list         # List available weights

Requires: huggingface_hub
    pip install huggingface_hub
"""
import argparse
import os
import sys

REPO_ID = "yfeng0206/ijepa-oct-glaucoma"

WEIGHTS = {
    "jepa_patch-imagenet-init-ep32-best.pth.tar": {
        "desc": "Best encoder: ViT-B/16, ImageNet-init -> I-JEPA on 600K OCT slices, epoch 32",
        "size": "1.5 GB",
        "auc": "0.829 (fine-tuned)",
    },
    "jepa_patch-run3-ep11.pth.tar": {
        "desc": "Random-init encoder: ViT-B/16, I-JEPA on 600K OCT slices, epoch 11",
        "size": "1.5 GB",
        "auc": "0.819 val (fine-tuned)",
    },
    "vit_b16_imagenet_timm.pth": {
        "desc": "ImageNet supervised ViT-B/16 (timm). Base initialization for I-JEPA pretraining.",
        "size": "327 MB",
        "auc": "N/A (base init)",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Download model weights from HuggingFace")
    parser.add_argument("--all", action="store_true", help="Download all weights")
    parser.add_argument("--encoder", action="store_true", help="Download best encoder only")
    parser.add_argument("--list", action="store_true", help="List available weights")
    parser.add_argument("--output-dir", default="checkpoints", help="Output directory")
    args = parser.parse_args()

    if args.list or not (args.all or args.encoder):
        print(f"Available weights from {REPO_ID}:\n")
        for fname, info in WEIGHTS.items():
            print(f"  {fname}")
            print(f"    {info['desc']}")
            print(f"    Size: {info['size']}  |  AUC: {info['auc']}")
            print()
        if not (args.all or args.encoder):
            print("Use --encoder for best encoder, --all for everything")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.encoder:
        files = ["jepa_patch-imagenet-init-ep32-best.pth.tar"]
    else:
        files = list(WEIGHTS.keys())

    for fname in files:
        info = WEIGHTS[fname]
        local_path = os.path.join(args.output_dir, fname)
        if os.path.exists(local_path):
            print(f"Already exists: {local_path}")
            continue
        print(f"Downloading {fname} ({info['size']})...")
        path = hf_hub_download(REPO_ID, fname, local_dir=args.output_dir)
        print(f"  Saved to {path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
