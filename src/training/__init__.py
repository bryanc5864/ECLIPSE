"""
Training Infrastructure for ECLIPSE.

Provides training loops, loss functions, and utilities for:
- Module 1 (ecDNA-Former): Formation prediction training
- Module 2 (CircularODE): Dynamics modeling training
- Module 3 (VulnCausal): Causal inference training
- Full ECLIPSE: End-to-end training
"""

from .trainer import (
    ECDNAFormerTrainer,
    CircularODETrainer,
    VulnCausalTrainer,
    ECLIPSETrainer,
)
from .losses import (
    FocalLoss,
    PhysicsInformedLoss,
    CausalLoss,
    MultiTaskLoss,
)
from .schedulers import (
    WarmupCosineScheduler,
    LinearWarmupScheduler,
)

__all__ = [
    "ECDNAFormerTrainer",
    "CircularODETrainer",
    "VulnCausalTrainer",
    "ECLIPSETrainer",
    "FocalLoss",
    "PhysicsInformedLoss",
    "CausalLoss",
    "MultiTaskLoss",
    "WarmupCosineScheduler",
    "LinearWarmupScheduler",
]
