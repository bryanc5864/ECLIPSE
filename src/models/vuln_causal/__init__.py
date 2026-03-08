"""
VulnCausal: Causal Inference for Therapeutic Vulnerability Discovery.

Discovers causal synthetic lethal interactions with ecDNA using:
- Causal representation learning (disentanglement)
- Invariant Risk Minimization (IRM)
- Neural causal discovery (NOTEARS)
- Do-calculus for intervention estimation
"""

from .model import VulnCausal
from .causal_encoder import CausalRepresentationLearner
from .invariant_predictor import InvariantRiskMinimization
from .causal_graph import NeuralCausalDiscovery
from .intervention import DoCalculusNetwork

__all__ = [
    "VulnCausal",
    "CausalRepresentationLearner",
    "InvariantRiskMinimization",
    "NeuralCausalDiscovery",
    "DoCalculusNetwork",
]
