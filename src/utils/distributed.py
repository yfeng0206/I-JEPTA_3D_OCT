"""Distributed training utilities for I-JEPA.

Supports initialization via SLURM environment variables as well as
``torchrun`` (which sets LOCAL_RANK / RANK / WORLD_SIZE).
"""

import os

import torch
import torch.distributed as dist


def init_distributed():
    """Initialize torch.distributed using NCCL backend.

    Detection order:
        1. SLURM (``SLURM_PROCID``, ``SLURM_LOCALID``, ``SLURM_NTASKS``).
        2. torchrun / ``torch.distributed.launch`` (``RANK``, ``LOCAL_RANK``,
           ``WORLD_SIZE``).

    Returns:
        (world_size, rank) after the process group has been initialized.
    """
    # -- Already initialized? -------------------------------------------------
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        return world_size, rank

    # -- Detect launch method --------------------------------------------------
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
    elif 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
    else:
        # Fall back to single-process (non-distributed) mode.
        return 1, 0

    # -- Init process group ----------------------------------------------------
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )
    return world_size, rank


class AllReduce(torch.autograd.Function):
    """Autograd-compatible all-reduce that averages a scalar across processes.

    Usage::

        loss = AllReduce.apply(loss)

    In the forward pass the input tensor is all-reduced (averaged).  The
    backward pass simply passes gradients through (identity).
    """

    @staticmethod
    def forward(ctx, x):
        if dist.is_initialized() and dist.get_world_size() > 1:
            x_reduced = x.clone()
            dist.all_reduce(x_reduced, op=dist.ReduceOp.SUM)
            x_reduced.div_(dist.get_world_size())
            return x_reduced
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
