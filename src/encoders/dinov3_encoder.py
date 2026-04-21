"""DINOv3 ViT-B/16 (Meta general-purpose vision FM).

Public HuggingFace model: facebook/dinov3-vitb16-pretrain-lvd1689m

Input:  (S, 3, 256, 256) RGB in [0, 1]
Output: (S, 768) per-slice CLS-token pooled output

Loaded via transformers.AutoModel (trust_remote_code=False, standard ViT).
"""

from . import register
from .base import EncoderAdapter

import torch
from torch.cuda.amp import autocast


@register
class DINOv3Encoder(EncoderAdapter):
    name = 'dinov3'
    embed_dim = 768
    input_layout = 'per_slice_2d'
    input_size = 256

    HF_MODEL_ID = 'facebook/dinov3-vitb16-pretrain-lvd1689m'

    def _load_model(self):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(
            self.kwargs.get('hf_model_id', self.HF_MODEL_ID),
            torch_dtype=torch.float16,
        ).to(self.device).eval()

    @torch.no_grad()
    def encode_volume(self, volume, chunk_size=16):
        """volume: (S, 3, H, W) in [0, 1]. Returns (S, 768) fp16 CPU."""
        S = volume.size(0)
        parts = []
        for j in range(0, S, chunk_size):
            chunk = volume[j:j + chunk_size].to(self.device)
            chunk = self._resize_if_needed(chunk)
            chunk = self._imagenet_normalize(chunk)
            with autocast():
                out = self.model(pixel_values=chunk)
            # DINOv3 AutoModel returns BaseModelOutputWithPooling; CLS is last_hidden_state[:, 0]
            # or pooler_output if available. Prefer CLS for foundation-model consistency.
            if hasattr(out, 'pooler_output') and out.pooler_output is not None:
                feats = out.pooler_output                # (chunk, D)
            else:
                feats = out.last_hidden_state[:, 0]      # CLS token (chunk, D)
            parts.append(feats.to(torch.float16).cpu())
        return torch.cat(parts, dim=0)                   # (S, 768)
