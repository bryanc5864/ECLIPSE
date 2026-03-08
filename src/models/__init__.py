"""
ECLIPSE Model Architectures.

Module 1: ecDNA-Former
    - Predicts ecDNA formation from genomic context
    - Uses topological deep learning over chromatin structure

Module 2: CircularODE
    - Models ecDNA copy number dynamics
    - Physics-informed neural stochastic differential equation

Module 3: VulnCausal
    - Discovers causal therapeutic vulnerabilities
    - Uses invariant prediction and causal inference
"""

from .ecdna_former import ECDNAFormer
from .circular_ode import CircularODE
from .vuln_causal import VulnCausal
from .eclipse import ECLIPSE

__all__ = [
    "ECDNAFormer",
    "CircularODE",
    "VulnCausal",
    "ECLIPSE",
]
