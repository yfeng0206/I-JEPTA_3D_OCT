"""Learning-rate and weight-decay schedulers for I-JEPA.

Both schedulers follow the convention used in the original I-JEPA codebase:
calling ``step()`` updates the relevant optimizer param-group entries and
returns the new value so it can be logged easily.
"""

import math


class WarmupCosineSchedule:
    """Cosine-annealing learning-rate schedule with linear warmup.

    During the first ``warmup_steps`` steps the LR is linearly ramped from
    ``start_lr`` to ``ref_lr``.  After that it is cosine-annealed from
    ``ref_lr`` down to ``final_lr`` over the remaining steps up to ``T_max``.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of warmup iterations.
        start_lr: Learning rate at iteration 0.
        ref_lr: Peak learning rate (reached at end of warmup).
        final_lr: Minimum learning rate at end of schedule.
        T_max: Total number of iterations (warmup + cosine).
    """

    def __init__(self, optimizer, warmup_steps, start_lr, ref_lr, final_lr, T_max):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.T_max = T_max
        self._step = 0

    def step(self):
        """Advance the schedule by one step.

        Returns:
            float: The new learning rate.
        """
        if self._step < self.warmup_steps:
            # Linear warmup
            progress = self._step / max(1, self.warmup_steps)
            lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # Cosine annealing
            progress = (self._step - self.warmup_steps) / max(
                1, self.T_max - self.warmup_steps
            )
            progress = min(progress, 1.0)
            lr = self.final_lr + 0.5 * (self.ref_lr - self.final_lr) * (
                1.0 + math.cos(math.pi * progress)
            )

        self._step += 1

        for group in self.optimizer.param_groups:
            group['lr'] = lr

        return lr


class CosineWDSchedule:
    """Cosine weight-decay schedule.

    Anneals weight decay from ``ref_wd`` to ``final_wd`` over ``T_max``
    iterations using a cosine curve.

    Args:
        optimizer: Wrapped optimizer.
        ref_wd: Starting (reference) weight decay.
        final_wd: Final weight decay at end of schedule.
        T_max: Total number of iterations.
    """

    def __init__(self, optimizer, ref_wd, final_wd, T_max):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0
        # Only schedule groups that were initialized with non-zero weight decay
        self._wd_group_indices = [
            i for i, g in enumerate(optimizer.param_groups)
            if g.get('weight_decay', 0.0) > 0.0
        ]

    def step(self):
        """Advance the schedule by one step.

        Returns:
            float: The new weight-decay value.
        """
        progress = self._step / max(1, self.T_max)
        progress = min(progress, 1.0)
        wd = self.final_wd + 0.5 * (self.ref_wd - self.final_wd) * (
            1.0 + math.cos(math.pi * progress)
        )

        self._step += 1

        for idx in self._wd_group_indices:
            self.optimizer.param_groups[idx]['weight_decay'] = wd

        return wd
