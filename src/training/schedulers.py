"""
Learning Rate Schedulers for ECLIPSE.

Provides:
- WarmupCosineScheduler: Cosine annealing with warmup
- LinearWarmupScheduler: Linear warmup then constant
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import List


class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine annealing schedule with linear warmup.

    Learning rate increases linearly during warmup, then decreases
    following a cosine curve.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        """
        Initialize scheduler.

        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr: Minimum learning rate
            last_epoch: Last epoch (for resuming)
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get learning rates for all param groups."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


class LinearWarmupScheduler(_LRScheduler):
    """
    Linear warmup followed by constant learning rate.

    Useful for fine-tuning or when cosine decay isn't needed.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        """
        Initialize scheduler.

        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            last_epoch: Last epoch (for resuming)
        """
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get learning rates for all param groups."""
        if self.last_epoch < self.warmup_steps:
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        return self.base_lrs


class CyclicWarmupScheduler(_LRScheduler):
    """
    Cyclic learning rate with warmup at the start of each cycle.

    Good for training dynamics models where different phases
    may benefit from learning rate restarts.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        cycle_length: int,
        warmup_fraction: float = 0.1,
        min_lr_factor: float = 0.1,
        last_epoch: int = -1,
    ):
        """
        Initialize scheduler.

        Args:
            optimizer: Optimizer to schedule
            cycle_length: Steps per cycle
            warmup_fraction: Fraction of cycle for warmup
            min_lr_factor: Minimum LR as fraction of base
            last_epoch: Last epoch (for resuming)
        """
        self.cycle_length = cycle_length
        self.warmup_steps = int(cycle_length * warmup_fraction)
        self.min_lr_factor = min_lr_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get learning rates for all param groups."""
        cycle_position = self.last_epoch % self.cycle_length

        if cycle_position < self.warmup_steps:
            # Warmup phase
            factor = cycle_position / max(1, self.warmup_steps)
        else:
            # Cosine decay within cycle
            progress = (cycle_position - self.warmup_steps) / max(
                1, self.cycle_length - self.warmup_steps
            )
            factor = self.min_lr_factor + (1 - self.min_lr_factor) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        return [base_lr * factor for base_lr in self.base_lrs]


def get_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    **kwargs
) -> _LRScheduler:
    """
    Get scheduler by name.

    Args:
        name: Scheduler name
        optimizer: Optimizer to schedule
        **kwargs: Scheduler-specific arguments

    Returns:
        Learning rate scheduler
    """
    schedulers = {
        "warmup_cosine": WarmupCosineScheduler,
        "linear_warmup": LinearWarmupScheduler,
        "cyclic_warmup": CyclicWarmupScheduler,
    }

    if name not in schedulers:
        raise ValueError(f"Unknown scheduler: {name}. Available: {list(schedulers.keys())}")

    return schedulers[name](optimizer, **kwargs)
