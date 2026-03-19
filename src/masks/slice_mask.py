"""1-D slice-level masking collator for slice-level I-JEPA.

Operates on a sequence of ``num_slices`` slice tokens (default 32) and
produces contiguous runs of masked / visible slices analogous to the 2-D
block masks used in patch-level I-JEPA.
"""

import random
from typing import List, Tuple

import torch


class SliceMaskCollator:
    """Generate context and target masks over a 1-D sequence of slices.

    Target masks are contiguous segments of ~3--6 slices.
    The context mask is the complement: all slices *not* covered by any
    target, guaranteeing at least ``min_keep`` context tokens.

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_segment_length(self, scale):
        # type: (Tuple[float, float],) -> int
        """Return a random contiguous-segment length in [lo, hi] slices."""
        lo = max(1, int(self.num_slices * scale[0]))
        hi = max(lo, int(self.num_slices * scale[1]))
        return random.randint(lo, hi)

    @staticmethod
    def _sample_segment(length, total):
        # type: (int, int) -> List[int]
        """Return sorted indices for a contiguous segment of *length* within [0, total)."""
        length = min(length, total)
        start = random.randint(0, total - length)
        return list(range(start, start + length))

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------

    def __call__(self, batch):
        """Collate a batch and produce encoder / predictor masks.

        Args:
            batch: list of tensors.  Each element is either
                * a single tensor (pretraining) -- e.g. (num_slices, C, H, W)
                * a tuple ``(tensor, label)`` (downstream)
                Only the first element is stacked; labels are handled by
                the caller.

        Returns:
            collated_batch: stacked batch tensor.
            collated_masks_enc: list with one (B, num_keep) tensor.
            collated_masks_pred: list of ``npred`` (B, seg_len) tensors.
        """
        # Unpack: allow both raw tensors and (tensor, label) tuples.
        if isinstance(batch[0], (tuple, list)):
            tensors = [b[0] for b in batch]
        else:
            tensors = batch

        B = len(tensors)
        collated_batch = torch.stack(tensors, dim=0)

        all_enc = []   # one list for the single context mask
        all_pred = [[] for _ in range(self.npred)]

        for _ in range(B):
            # --- Target segments ---
            target_set = set()  # type: set
            per_sample_pred = []
            for p in range(self.npred):
                seg_len = self._sample_segment_length(self.pred_mask_scale)
                seg = self._sample_segment(seg_len, self.num_slices)
                per_sample_pred.append(seg)
                target_set.update(seg)

            # --- Context: complement of all target slices ---
            context = sorted(
                i for i in range(self.num_slices) if i not in target_set
            )

            # If we removed too many slices, randomly add some back until
            # we reach min_keep.
            if len(context) < self.min_keep:
                deficit = self.min_keep - len(context)
                extra = random.sample(sorted(target_set), min(deficit, len(target_set)))
                context = sorted(set(context) | set(extra))
                # Also remove those indices from every target segment that
                # contains them so target and context stay disjoint.
                extra_set = set(extra)
                per_sample_pred = [
                    [idx for idx in seg if idx not in extra_set]
                    for seg in per_sample_pred
                ]
                # Ensure no target segment is empty.
                per_sample_pred = [
                    seg if len(seg) > 0 else [random.choice(sorted(target_set - extra_set))]
                    if len(target_set - extra_set) > 0 else [0]
                    for seg in per_sample_pred
                ]

            all_enc.append(torch.tensor(context, dtype=torch.long))

            for p in range(self.npred):
                all_pred[p].append(
                    torch.tensor(sorted(per_sample_pred[p]), dtype=torch.long)
                )

        collated_masks_enc = self._truncate_and_stack([all_enc])
        collated_masks_pred = self._truncate_and_stack(all_pred)

        return collated_batch, collated_masks_enc, collated_masks_pred

    # ------------------------------------------------------------------

    @staticmethod
    def _truncate_and_stack(mask_groups):
        # type: (List[List[torch.Tensor]]) -> List[torch.Tensor]
        """Truncate each group to the minimum length and stack into (B, min_len).

        This matches the original I-JEPA approach: all samples in a batch
        must have the same number of mask indices per target block.
        """
        result = []
        for group in mask_groups:
            min_len = min(t.numel() for t in group)
            truncated = [t[:min_len] for t in group]
            result.append(torch.stack(truncated, dim=0))
        return result
