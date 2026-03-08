"""
Utilities for ECLIPSE.

Provides:
- Genomic utilities (coordinate handling, sequence processing)
- Graph utilities (Hi-C graph construction)
- Evaluation metrics (AUROC, calibration, etc.)
- Visualization utilities
"""

from .genomics import (
    GenomicCoordinates,
    SequenceProcessor,
    parse_bed_file,
    liftover_coordinates,
)
from .graphs import (
    build_hic_graph,
    compute_graph_features,
    normalize_adjacency,
)
from .metrics import (
    compute_auroc,
    compute_auprc,
    compute_calibration_error,
    compute_f1_multilabel,
    EvaluationMetrics,
)

__all__ = [
    "GenomicCoordinates",
    "SequenceProcessor",
    "parse_bed_file",
    "liftover_coordinates",
    "build_hic_graph",
    "compute_graph_features",
    "normalize_adjacency",
    "compute_auroc",
    "compute_auprc",
    "compute_calibration_error",
    "compute_f1_multilabel",
    "EvaluationMetrics",
]
