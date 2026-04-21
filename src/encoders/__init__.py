"""Foundation-model encoder adapters for baseline comparisons.

Each adapter presents a unified interface so the downstream probe code can
treat any backbone (DINOv3, OCTCube, our own I-JEPA) the same way:

    adapter = build_adapter(name, device=..., weights_path=...)
    features = adapter.encode_volume(volume)   # (S, D) or (D,)
    ...
    adapter.cleanup()                           # free GPU + RAM

Layouts:
  'per_slice_2d'  — returns (S, D). Pair with CrossAttnPool / MeanPool / d=1.
  'volume_3d'     — returns (D,).   Pair with LinearHead only.
"""

from .base import EncoderAdapter

ADAPTER_REGISTRY = {}


def register(cls):
    ADAPTER_REGISTRY[cls.name] = cls
    return cls


def build_adapter(name, **kwargs):
    if name not in ADAPTER_REGISTRY:
        raise ValueError(
            f'unknown encoder adapter {name!r}. Known: {list(ADAPTER_REGISTRY)}'
        )
    return ADAPTER_REGISTRY[name](**kwargs)


# Import side-effect registers each adapter.
from . import dinov3_encoder      # noqa: E402, F401
from . import octcube_encoder     # noqa: E402, F401
