"""Minimal data augmentation for I-JEPA OCT pretraining.

I-JEPA relies on very light augmentation -- typically just a random
resized crop and normalisation.  This module exposes a single factory
function ``make_transforms`` whose defaults match the original I-JEPA
paper.

Compatible with PyTorch 1.13.1 / Python 3.8.
"""

from typing import List, Tuple

import torchvision.transforms as T


# ImageNet statistics (used as a reasonable default for transfer learning).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_transforms(
    crop_size=256,                          # type: int
    crop_scale=(0.3, 1.0),                  # type: Tuple[float, float]
    gaussian_blur=False,                    # type: bool
    horizontal_flip=False,                  # type: bool
    color_distortion=False,                 # type: bool
    color_jitter=0.0,                       # type: float
    normalize_mean=IMAGENET_MEAN,           # type: Tuple[float, ...]
    normalize_std=IMAGENET_STD,             # type: Tuple[float, ...]
):
    """Build a ``torchvision.transforms.Compose`` pipeline.

    The pipeline expects a PIL Image as input and returns a
    ``(3, crop_size, crop_size)`` float tensor normalised with the
    given channel statistics.

    Args:
        crop_size: Output spatial size after random-resized crop.
        crop_scale: (min, max) area fraction for ``RandomResizedCrop``.
        gaussian_blur: Apply Gaussian blur (kernel 23, sigma [0.1, 2.0]).
        horizontal_flip: Apply random horizontal flip (p=0.5).
        color_distortion: Enable color jitter augmentation.
        color_jitter: Strength of color jitter (brightness, contrast,
            saturation all set to this value; hue set to 0).
        normalize_mean: Per-channel mean for ``Normalize``.
        normalize_std: Per-channel std for ``Normalize``.

    Returns:
        ``torchvision.transforms.Compose`` instance.
    """
    ops = []  # type: List[object]

    # -- Spatial ----------------------------------------------------------
    ops.append(
        T.RandomResizedCrop(
            crop_size,
            scale=crop_scale,
            interpolation=T.InterpolationMode.BICUBIC
            if hasattr(T.InterpolationMode, "BICUBIC")
            else 3,  # PIL.Image.BICUBIC for older torchvision
        )
    )

    if horizontal_flip:
        ops.append(T.RandomHorizontalFlip(p=0.5))

    # -- Color ------------------------------------------------------------
    if color_distortion and color_jitter > 0.0:
        ops.append(
            T.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=0.0,
            )
        )

    # -- Blur -------------------------------------------------------------
    if gaussian_blur:
        # GaussianBlur was added in torchvision 0.8.
        ops.append(T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)))

    # -- Tensor conversion & normalisation --------------------------------
    ops.append(T.ToTensor())  # PIL -> (C, H, W) float in [0, 1]
    ops.append(T.Normalize(mean=normalize_mean, std=normalize_std))

    return T.Compose(ops)
