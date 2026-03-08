"""
ECLIPSE: Extrachromosomal Circular DNA Learning for Integrated
Prediction of Synthetic-lethality and Expression

A computational framework for:
- Module 1 (ecDNA-Former): Predicting ecDNA formation from genomic context
- Module 2 (CircularODE): Modeling ecDNA evolutionary dynamics
- Module 3 (VulnCausal): Discovering therapeutic vulnerabilities
"""

__version__ = "0.1.0"
__author__ = "ECLIPSE Team"

from . import data
from . import models
from . import training
from . import utils
