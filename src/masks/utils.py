"""Mask utilities for I-JEPA."""

import torch


def apply_masks(x, masks):
    """Apply binary masks to select token subsets from a sequence.

    Args:
        x: Tensor of shape (B, N, D) — full sequence of tokens.
        masks: List of index tensors, each of shape (B, num_keep) containing
            the integer indices of tokens to keep.  When multiple masks are
            provided the results are concatenated along the batch dimension
            (each mask yields one copy of the batch).

    Returns:
        Tensor of shape (B_total, num_keep, D) where B_total = B * len(masks).
    """
    all_x = []
    for m in masks:
        # m: (B, num_keep) — integer indices into the N dimension
        mask_keep = m.unsqueeze(-1).expand(-1, -1, x.size(-1))  # (B, num_keep, D)
        all_x.append(torch.gather(x, dim=1, index=mask_keep))
    return torch.cat(all_x, dim=0)
