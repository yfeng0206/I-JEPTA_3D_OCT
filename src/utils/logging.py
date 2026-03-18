"""Logging utilities for I-JEPA training."""

import csv
import time

import torch


class CSVLogger:
    """Simple CSV logger that appends one row per call.

    Args:
        log_file: Path to the CSV file.  Created (with header) if it does not
            exist; rows are appended otherwise.
        *columns: Column names written as the header row.
    """

    def __init__(self, log_file, *columns):
        self.log_file = log_file
        self.columns = columns
        # Write header if the file is new
        try:
            with open(log_file, 'r'):
                pass
        except FileNotFoundError:
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(columns)

    def log(self, *values):
        """Append a single row of values (must match column order)."""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(values)


class AverageMeter:
    """Tracks a running average and current value.

    Usage::

        meter = AverageMeter()
        for batch in loader:
            loss = compute_loss(batch)
            meter.update(loss.item())
        print(meter.avg)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        """Record a new value.

        Args:
            val: The value to record.
            n: Number of items this value represents (for weighted average).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def gpu_timer(fn):
    """Time a callable using CUDA events for accurate GPU timing.

    Falls back to wall-clock time when CUDA is not available.

    Args:
        fn: A zero-argument callable to time.

    Returns:
        (result, elapsed_ms): The return value of *fn* and elapsed time in
        milliseconds.
    """
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    else:
        t0 = time.time()
        result = fn()
        elapsed_ms = (time.time() - t0) * 1000.0

    return result, elapsed_ms


def grad_logger(named_params):
    """Log gradient statistics (mean abs value and max abs value) per parameter.

    Args:
        named_params: Iterable of (name, parameter) pairs, e.g. from
            ``model.named_parameters()``.

    Returns:
        list of dicts: Each dict has keys ``'name'``, ``'grad_mean'``, and
        ``'grad_max'``.  Parameters without gradients are skipped.
    """
    stats = []
    for name, param in named_params:
        if param.grad is not None:
            grad = param.grad.data
            stats.append({
                'name': name,
                'grad_mean': grad.abs().mean().item(),
                'grad_max': grad.abs().max().item(),
            })
    return stats
