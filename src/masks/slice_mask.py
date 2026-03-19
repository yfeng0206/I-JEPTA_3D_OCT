"""1-D slice-level masking collator for slice-level I-JEPA.

Operates on a sequence of ``num_slices`` slice tokens (default 32) and
produces contiguous runs of masked / visible slices analogous to the 2-D
block masks used in patch-level I-JEPA.

The output format matches the original I-JEPA MaskCollator:
- masks_enc: list of 1 tensor of shape (B, num_context) — context indices
- masks_pred: list of npred tensors, each (B, min_seg_len) — target indices
"""

import random
from typing import List, Tuple

import torch


class SliceMaskCollator:
    """Generate context and target masks over a 1-D sequence of slices.

    Args:
        num_slices: Total number of slice tokens in each volume.
        enc_mask_scale: (min, max) fraction of slices to keep as context.
        pred_mask_scale: (min, max) fraction of slices per target segment.
        npred: Number of target segments per volume.
        min_keep: Minimum number of context slices to retain.
    """

    def __init__(
        self,
        num_slices=32,
        enc_mask_scale=(0.75, 0.9),
        pred_mask_scale=(0.1, 0.2),
        npred=4,
        min_keep=10,
    ):
        self.num_slices = num_slices
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.npred = npred
        self.min_keep = min_keep

    def _sample_segment_length(self, scale):
        lo = max(1, int(self.num_slices * scale[0]))
        hi = max(lo, int(self.num_slices * scale[1]))
        return random.randint(lo, hi)

    @staticmethod
    def _sample_segment(length, total):
        length = min(length, total)
        start = random.randint(0, total - length)
        return list(range(start, start + length))

    def __call__(self, batch):
        if isinstance(batch[0], (tuple, list)):
            tensors = [b[0] for b in batch]
        else:
            tensors = batch

        B = len(tensors)
        collated_batch = torch.stack(tensors, dim=0)

        # Collect per-sample masks
        all_enc_indices = []    # list of B lists of context indices
        all_pred_indices = []   # list of B lists of npred target segments

        for _ in range(B):
            # Sample npred target segments
            target_set = set()
            segments = []
            for _ in range(self.npred):
                seg_len = self._sample_segment_length(self.pred_mask_scale)
                seg = self._sample_segment(seg_len, self.num_slices)
                segments.append(sorted(seg))
                target_set.update(seg)

            # Context = complement of all targets
            context = sorted(i for i in range(self.num_slices) if i not in target_set)

            # Ensure min_keep context slices
            if len(context) < self.min_keep:
                deficit = self.min_keep - len(context)
                extra = random.sample(sorted(target_set), min(deficit, len(target_set)))
                context = sorted(set(context) | set(extra))
                extra_set = set(extra)
                segments = [
                    [idx for idx in seg if idx not in extra_set]
                    for seg in segments
                ]
                # Ensure no segment is empty
                remaining = sorted(target_set - extra_set)
                segments = [
                    seg if len(seg) > 0 else ([random.choice(remaining)] if remaining else [0])
                    for seg in segments
                ]

            all_enc_indices.append(context)
            all_pred_indices.append(segments)

        # Build encoder masks: truncate to min context length across batch
        min_enc = min(len(ctx) for ctx in all_enc_indices)
        enc_mask = torch.stack([
            torch.tensor(ctx[:min_enc], dtype=torch.long)
            for ctx in all_enc_indices
        ], dim=0)  # (B, min_enc)

        # Build predictor masks: find the GLOBAL minimum segment length
        # across ALL blocks and ALL samples (like original I-JEPA), so
        # apply_masks can torch.cat the results along dim=0.
        global_min_seg = min(
            len(all_pred_indices[b][p])
            for b in range(B)
            for p in range(self.npred)
        )
        global_min_seg = max(global_min_seg, 1)

        pred_masks = []
        for p in range(self.npred):
            pred_masks.append(torch.stack([
                torch.tensor(all_pred_indices[b][p][:global_min_seg], dtype=torch.long)
                for b in range(B)
            ], dim=0))  # (B, global_min_seg)

        return collated_batch, [enc_mask], pred_masks
