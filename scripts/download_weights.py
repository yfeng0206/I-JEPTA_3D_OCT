"""Download pretrained I-JEPA and downstream model weights from Azure Blob.

Usage:
    python scripts/download_weights.py --all          # Download everything
    python scripts/download_weights.py --encoder      # Just the best encoder (ep32)
    python scripts/download_weights.py --downstream   # Just downstream model weights

Requires: azure-identity, azure-storage-blob
    pip install azure-identity azure-storage-blob
"""
import argparse
import os
import sys

BLOB_ACCOUNT = "STORAGE_ACCOUNT_REDACTED"
BLOB_CONTAINER = "CONTAINER_REDACTED"

# Best pretrained encoder: ImageNet-init I-JEPA, epoch 32
ENCODER_WEIGHTS = {
    "ijepa-results/patch_vit_base_ps16_ep100_bs64_lr0.00025_20260402_001335/jepa_patch-best.pth.tar": {
        "local": "weights/encoder/jepa_patch-ep32-best.pth.tar",
        "desc": "I-JEPA ViT-B/16 encoder (ImageNet-init, ep32 best, 1.5 GB)",
        "size_mb": 1437,
    },
}

# Best downstream models (encoder + probe + head)
DOWNSTREAM_WEIGHTS = {
    "ijepa-downstream/downstream_patch_s32_ep25_bs1_mlp_20260407_042850/best_model.pt": {
        "local": "weights/downstream/unfrozen_imagenet_d2_s32_best.pt",
        "desc": "Unfrozen d=2 s32 (0.828 test AUC, 401 MB)",
        "size_mb": 382,
    },
    "ijepa-downstream/downstream_patch_s64_ep25_bs1_mlp_20260407_144520/best_model.pt": {
        "local": "weights/downstream/unfrozen_imagenet_d2_s64_best.pt",
        "desc": "Unfrozen d=2 s64 (0.829 test AUC, 401 MB)",
        "size_mb": 382,
    },
    "ijepa-downstream/downstream_patch_s32_ep25_bs1_mlp_20260408_060928/best_model.pt": {
        "local": "weights/downstream/unfrozen_imagenet_d3_s32_best.pt",
        "desc": "Unfrozen d=3 s32 (0.829 test AUC, 401 MB)",
        "size_mb": 382,
    },
}


def download(blob_name, local_path, desc, container):
    if os.path.exists(local_path):
        print(f"  Already exists: {local_path}")
        return
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"  Downloading: {desc}")
    print(f"    {blob_name} -> {local_path}")
    data = container.download_blob(blob_name).readall()
    with open(local_path, "wb") as f:
        f.write(data)
    size_mb = len(data) / (1024 * 1024)
    print(f"    Done ({size_mb:.0f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Download model weights from Azure Blob")
    parser.add_argument("--all", action="store_true", help="Download all weights")
    parser.add_argument("--encoder", action="store_true", help="Download best encoder only")
    parser.add_argument("--downstream", action="store_true", help="Download downstream models only")
    args = parser.parse_args()

    if not (args.all or args.encoder or args.downstream):
        parser.print_help()
        sys.exit(1)

    try:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import ContainerClient
    except ImportError:
        print("Install dependencies: pip install azure-identity azure-storage-blob")
        sys.exit(1)

    cred = DefaultAzureCredential()
    container = ContainerClient(
        account_url=f"https://{BLOB_ACCOUNT}.blob.core.windows.net",
        container_name=BLOB_CONTAINER,
        credential=cred,
    )

    if args.all or args.encoder:
        print("\n=== Encoder Weights ===")
        for blob_name, info in ENCODER_WEIGHTS.items():
            download(blob_name, info["local"], info["desc"], container)

    if args.all or args.downstream:
        print("\n=== Downstream Model Weights ===")
        for blob_name, info in DOWNSTREAM_WEIGHTS.items():
            download(blob_name, info["local"], info["desc"], container)

    print("\nDone!")


if __name__ == "__main__":
    main()
