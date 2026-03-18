"""Tensor utilities for I-JEPA."""

import math

import torch


def trunc_normal_(tensor, mean=0., std=1.):
    """Fill tensor with values drawn from a truncated normal distribution.

    Values are clamped to lie within [mean - 2*std, mean + 2*std] and then
    the distribution is re-normalized so that the resulting samples truly
    approximate a truncated normal.

    Compatible with PyTorch 1.13.1 / Python 3.8.
    """
    with torch.no_grad():
        # Compute truncation bounds in standard-normal space
        a, b = -2.0, 2.0  # number of std devs for clamping

        # Use inverse CDF (Phi^-1) method for proper truncated normal.
        # For the standard normal CDF we use math.erf.
        def _norm_cdf(x):
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

        l = _norm_cdf(a)  # noqa: E741
        u = _norm_cdf(b)

        # Draw uniform samples in [l, u], then apply inverse CDF
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(mean - 2 * std, mean + 2 * std)

    return tensor


def repeat_interleave_batch(x, B, repeat):
    """Repeat each element in a batch ``repeat`` times.

    Given a tensor ``x`` of shape (B, ...), returns a tensor of shape
    (B * repeat, ...) where each sample is repeated ``repeat`` consecutive
    times.  This is useful for multi-mask handling where every image needs
    to be paired with several different masks.

    Args:
        x: Input tensor of shape (B, ...).
        B: Batch size (leading dimension of *x*).
        repeat: Number of times to repeat each sample.

    Returns:
        Tensor of shape (B * repeat, ...).
    """
    N = x.shape[0] // B
    x = x.unsqueeze(1).repeat(1, repeat, *([1] * (x.dim() - 1)))
    x = x.reshape(B * repeat * N, *x.shape[2:])
    return x
