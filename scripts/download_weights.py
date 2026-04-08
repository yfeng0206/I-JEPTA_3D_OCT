"""Download pretrained model weights.

Weights will be hosted on HuggingFace Hub (coming soon).
For now, contact the authors for access.

Available weights:
- Encoder: I-JEPA ViT-B/16 (ImageNet-init, ep32 best) — 1.5 GB
- Downstream: Unfrozen d=2 s32 (0.828 test AUC) — 401 MB
- Downstream: Unfrozen d=2 s64 (0.829 test AUC) — 401 MB
- Downstream: Unfrozen d=3 s32 (0.829 test AUC) — 401 MB
"""
# TODO: Add HuggingFace Hub download once weights are uploaded
# from huggingface_hub import hf_hub_download
# hf_hub_download(repo_id="yfeng0206/ijepa-oct-glaucoma", filename="weights/encoder/jepa_patch-ep32-best.pth.tar")
