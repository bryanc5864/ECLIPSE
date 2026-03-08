"""
CircularODE: Physics-Informed Neural SDE for ecDNA Dynamics.

Models ecDNA copy number evolution as a stochastic differential equation
with biologically-motivated constraints:
- Binomial segregation (random inheritance)
- Oncogene-driven fitness advantage
- Treatment-induced selection
"""

from .model import CircularODE
from .dynamics import DriftNetwork, DiffusionNetwork, SegregationPhysics
from .treatment import TreatmentEncoder

__all__ = [
    "CircularODE",
    "DriftNetwork",
    "DiffusionNetwork",
    "SegregationPhysics",
    "TreatmentEncoder",
]
