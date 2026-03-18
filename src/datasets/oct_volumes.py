"""Dataset for slice-level I-JEPA and downstream classification.

Each FairVision ``.npz`` volume has key ``oct_bscans`` with shape
(200, 200, 200) (uint8) and a ``glaucoma`` label (int).

This dataset returns a full volume represented as a stack of uniformly
sampled 2-D slices, each resized to ``slice_size x slice_size`` and
converted to 3-channel RGB.

Modes:
    * **Pretraining** (``return_label=False``): returns a single tensor
      of shape ``(num_slices, 3, slice_size, slice_size)``.
    * **Downstream** (``return_label=True``): returns
      ``(slices_tensor, label)`` where ``label`` is the integer
      glaucoma label.
"""

import glob
import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class OCTVolumeDataset(Dataset):
    """Volume-level dataset for slice-level I-JEPA.

    Args:
        data_dir: Path to the split directory containing ``.npz`` files.
        num_slices: Number of slices to uniformly sample from each volume.
        slice_size: Target spatial resolution (square) per slice.
        return_label: If ``True``, also return the ``glaucoma`` label.
        transform: Optional ``torchvision.transforms`` applied to each
            individual slice PIL image before stacking.
    """

    def __init__(
        self,
        data_dir,              # type: str
        num_slices=32,         # type: int
        slice_size=256,        # type: int
        return_label=False,    # type: bool
        transform=None,        # type: Optional[object]
    ):
        super(OCTVolumeDataset, self).__init__()

        self.data_dir = data_dir
        self.num_slices = num_slices
        self.slice_size = slice_size
        self.return_label = return_label
        self.transform = transform

        # Discover .npz files.
        pattern = os.path.join(data_dir, "*.npz")
        self.file_paths = sorted(glob.glob(pattern))
        if len(self.file_paths) == 0:
            raise RuntimeError(
                "No .npz files found in {!r}. Check the data_dir path.".format(
                    data_dir
                )
            )

        # Pre-compute uniform slice indices into the 200-depth volume.
        self.slice_indices = np.linspace(
            0, 199, num=num_slices, dtype=np.int64
        )

    # ------------------------------------------------------------------

    def __len__(self):
        # type: () -> int
        return len(self.file_paths)

    # ------------------------------------------------------------------

    def _process_slice(self, slice_2d):
        # type: (np.ndarray) -> torch.Tensor
        """Resize, convert to RGB, apply transform, return (3, H, W) tensor."""
        pil_img = Image.fromarray(slice_2d, mode="L")
        pil_img = pil_img.resize(
            (self.slice_size, self.slice_size), Image.BILINEAR
        )
        pil_img = pil_img.convert("RGB")

        if self.transform is not None:
            pil_img = self.transform(pil_img)
            if isinstance(pil_img, torch.Tensor):
                return pil_img

        # Manual conversion: normalise to [0, 1].
        arr = np.array(pil_img, dtype=np.float32) / 255.0  # (H, W, 3)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)    # (3, H, W)
        return tensor

    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        # type: (int) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]
        """Return a volume as a stacked slice tensor.

        Returns:
            If ``return_label`` is False:
                slices_tensor of shape ``(num_slices, 3, slice_size, slice_size)``
            If ``return_label`` is True:
                ``(slices_tensor, label)``
        """
        data = np.load(self.file_paths[idx], allow_pickle=True)
        volume = data["oct_bscans"]  # (200, 200, 200) uint8

        slices = []
        for s_idx in self.slice_indices:
            slice_2d = volume[int(s_idx)]  # (200, 200)
            slices.append(self._process_slice(slice_2d))

        slices_tensor = torch.stack(slices, dim=0)  # (num_slices, 3, H, W)

        if self.return_label:
            label = int(data["glaucoma"].item())
            return slices_tensor, label

        return slices_tensor
