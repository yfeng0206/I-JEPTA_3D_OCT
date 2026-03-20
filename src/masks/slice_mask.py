"""1-D slice-level masking collator for slice-level I-JEPA.

Matches the original I-JEPA MaskCollator design:
- Context is a sampled contiguous BLOCK (not the full complement of targets)
- Target positions within the context block are removed
- Slices outside both context and target are ignored
- All pred masks are truncated to the same global minimum length
"""

import random
from typing import List, Tuple

import torch


class SliceMaskCollator:
    """Generate context and target masks over a 1-D sequence of slices.

    Args:
        num_slices: Total number of slice tokens in each volume.
        enc_mask_scale: (min, max) fraction of slices for the context block.
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

        all_enc_indices = []
        all_pred_indices = []

        for _ in range(B):
            # 1. Sample target segments first
            target_set = set()
            segments = []
            for _ in range(self.npred):
                seg_len = self._sample_segment_length(self.pred_mask_scale)
                seg = self._sample_segment(seg_len, self.num_slices)
                segments.append(sorted(seg))
                target_set.update(seg)

            # 2. Sample a contiguous CONTEXT BLOCK (like original I-JEPA)
            ctx_len = self._sample_segment_length(self.enc_mask_scale)
            ctx_block = set(self._sample_segment(ctx_len, self.num_slices))

            # 3. Remove target positions from context block
            context = sorted(ctx_block - target_set)

            # 4. Ensure minimum context slices
            attempts = 0
            while len(context) < self.min_keep and attempts < 20:
                ctx_len = self._sample_segment_length(self.enc_mask_scale)
                ctx_block = set(self._sample_segment(ctx_len, self.num_slices))
                context = sorted(ctx_block - target_set)
                attempts += 1

            # If still too few, fall back to full complement
            if len(context) < self.min_keep:
                context = sorted(
                    i for i in range(self.num_slices) if i not in target_set
                )

            all_enc_indices.append(context)
            all_pred_indices.append(segments)

        # Build encoder masks: truncate to min context length across batch
        min_enc = min(len(ctx) for ctx in all_enc_indices)
        enc_mask = torch.stack([
            torch.tensor(ctx[:min_enc], dtype=torch.long)
            for ctx in all_enc_indices
        ], dim=0)

        # Build predictor masks: global minimum segment length
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
            ], dim=0))

        return collated_batch, [enc_mask], pred_masks
