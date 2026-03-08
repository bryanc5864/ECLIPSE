"""
ecDNA-Former: Topological Deep Learning for ecDNA Formation Prediction.

Components:
- SequenceEncoder: DNA language model for sequence context
- TopologyEncoder: Hierarchical graph transformer for Hi-C
- FragileSiteEncoder: Attention over chromosomal fragile sites
- CrossModalFusion: Bottleneck cross-attention fusion
- PredictionHeads: Formation probability + oncogene prediction
"""

from .model import ECDNAFormer
from .sequence_encoder import SequenceEncoder
from .topology_encoder import TopologyEncoder, HierarchicalGraphTransformer
from .fragile_site_encoder import FragileSiteEncoder
from .fusion import CrossModalFusion
from .heads import FormationHead, OncogeneHead

__all__ = [
    "ECDNAFormer",
    "SequenceEncoder",
    "TopologyEncoder",
    "HierarchicalGraphTransformer",
    "FragileSiteEncoder",
    "CrossModalFusion",
    "FormationHead",
    "OncogeneHead",
]
