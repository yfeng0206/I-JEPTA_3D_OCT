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
    """Repeat batches block-wise, matching the original I-JEPA implementation.

    Input ``x`` has shape ``(N * B, ...)``, interpreted as ``N`` groups of
    ``B`` samples each.  For each group, the ``B`` samples are emitted
    ``repeat`` times before moving to the next group.

    Example with N=2, B=3, repeat=2:
        Input groups:   [g0_s0, g0_s1, g0_s2,  g1_s0, g1_s1, g1_s2]
        Output:         [g0_s0, g0_s1, g0_s2,  g0_s0, g0_s1, g0_s2,
                         g1_s0, g1_s1, g1_s2,  g1_s0, g1_s1, g1_s2]

    Args:
        x: Input tensor of shape (N * B, ...).
        B: Batch size per group.
        repeat: Number of times to repeat each group.

    Returns:
        Tensor of shape (N * B * repeat, ...).
    """
    N = len(x) // B
    x = torch.cat([
        torch.cat([x[i * B:(i + 1) * B] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x
