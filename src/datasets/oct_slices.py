"""Dataset for patch-level I-JEPA: loads individual 2-D OCT B-scans.

Each FairVision ``.npz`` volume has key ``oct_bscans`` with shape
(200, 200, 200) (uint8) and a ``glaucoma`` label.  This dataset
uniformly samples ``num_slices`` axial slices from each volume and
exposes every slice as an independent sample.

Total dataset size: ``num_volumes * num_slices``.

Intended for self-supervised pretraining -- no labels are returned.
"""

import glob
import os
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class OCTSliceDataset(Dataset):
    """Individual-slice dataset for patch-level I-JEPA pretraining.

    Args:
        data_dir: Path to the split directory (e.g. ``Training/`` or
            ``Validation/``) that contains ``.npz`` files.
        num_slices: Number of slices to uniformly sample from each volume.
        slice_size: Target spatial resolution (square) of each slice.
        transform: Optional ``torchvision.transforms`` applied to the
            PIL image *before* final tensor conversion.
    """

    def __init__(
        self,
        data_dir,          # type: str
        num_slices=32,     # type: int
        slice_size=256,    # type: int
        transform=None,    # type: Optional[object]
    ):
        super(OCTSliceDataset, self).__init__()

        self.data_dir = data_dir
        self.num_slices = num_slices
        self.slice_size = slice_size
        self.transform = transform

        # Discover all .npz files.
        pattern = os.path.join(data_dir, "*.npz")
        self.file_paths = sorted(glob.glob(pattern))
        if len(self.file_paths) == 0:
            raise RuntimeError(
                "No .npz files found in {!r}. Check the data_dir path.".format(
                    data_dir
                )
            )

        # Pre-compute the slice indices (integers into the depth axis).
        # np.linspace gives evenly-spaced indices across the 200-slice volume.
        self.slice_indices = np.linspace(
            0, 199, num=num_slices, dtype=np.int64
        )

    # ------------------------------------------------------------------

    def __len__(self):
        # type: () -> int
        return len(self.file_paths) * self.num_slices

    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        # type: (int) -> torch.Tensor
        """Return a single slice as a (3, slice_size, slice_size) tensor."""
        file_idx = idx // self.num_slices
        slice_within = idx % self.num_slices

        # Load the volume (lazy-load; only the needed array is read).
        data = np.load(self.file_paths[file_idx], allow_pickle=True)
        volume = data["oct_bscans"]  # (200, 200, 200) uint8

        # Select the slice along the first axis.
        depth_idx = int(self.slice_indices[slice_within])
        slice_2d = volume[depth_idx]  # (200, 200) uint8

        # Resize to target resolution using PIL (bilinear).
        pil_img = Image.fromarray(slice_2d, mode="L")
        pil_img = pil_img.resize(
            (self.slice_size, self.slice_size), Image.BILINEAR
        )

        # Convert to 3-channel RGB (duplicate the grayscale channel).
        pil_img = pil_img.convert("RGB")

        # Apply user-supplied transforms (e.g. RandomResizedCrop, etc.).
        if self.transform is not None:
            pil_img = self.transform(pil_img)
            # If the transform already produces a tensor, return as-is.
            if isinstance(pil_img, torch.Tensor):
                return pil_img

        # Fallback: manual conversion to tensor normalised to [0, 1].
        arr = np.array(pil_img, dtype=np.float32) / 255.0  # (H, W, 3)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)    # (3, H, W)
        return tensor
