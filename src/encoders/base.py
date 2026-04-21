"""Abstract base for encoder adapters.

Subclasses must set class attributes (`name`, `embed_dim`, `input_layout`,
`input_size`) and implement `_load_model`, `encode_volume`, and `cleanup`.
"""

import gc
import torch
import torch.nn.functional as F


class EncoderAdapter:
    name: str = ''
    embed_dim: int = 0
    input_layout: str = 'per_slice_2d'     # or 'volume_3d'
    input_size: int = 256                   # resize target (H = W)
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std  = (0.229, 0.224, 0.225)

    def __init__(self, device='cuda', weights_path=None, **kwargs):
        self.device = device
        self.weights_path = weights_path
        self.kwargs = kwargs
        self.model = None
        self._load_model()

    # ---- subclass hooks -----------------------------------------------------
    def _load_model(self):
        raise NotImplementedError

    @torch.no_grad()
    def encode_volume(self, volume):
        """volume: (S, 3, H, W) float32 in [0, 1]. Returns (S, D) or (D,)."""
        raise NotImplementedError

    # ---- shared helpers -----------------------------------------------------
    def _resize_if_needed(self, x):
        """x: (B, 3, H, W). Resize to self.input_size if needed."""
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(
                x, size=(self.input_size, self.input_size),
                mode='bilinear', align_corners=False,
            )
        return x

    def _imagenet_normalize(self, x):
        mean = torch.tensor(self.imagenet_mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor(self.imagenet_std,  dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def cleanup(self):
        del self.model
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __repr__(self):
        return (f'<{self.__class__.__name__} name={self.name} '
                f'embed_dim={self.embed_dim} layout={self.input_layout}>')
