"""
Data loading and processing modules for ECLIPSE.

Handles:
- AmpliconRepository (ecDNA annotations)
- CytoCellDB (cell line ecDNA status)
- DepMap (CRISPR screens, expression, drug sensitivity)
- 4D Nucleome (Hi-C chromatin contact maps)
- Supplementary data (fragile sites, COSMIC genes)
"""

from .download import DataDownloader
from .loaders import (
    AmpliconRepositoryLoader,
    CytoCellDBLoader,
    DepMapLoader,
    HiCLoader,
    FragileSiteLoader,
)
from .processing import (
    DataProcessor,
    FeatureExtractor,
    SplitGenerator,
)
from .datasets import (
    ECDNADataset,
    DynamicsDataset,
    VulnerabilityDataset,
)
from torch.utils.data import DataLoader


def create_dataloader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
):
    """
    Create a DataLoader with sensible defaults.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


__all__ = [
    "DataDownloader",
    "AmpliconRepositoryLoader",
    "CytoCellDBLoader",
    "DepMapLoader",
    "HiCLoader",
    "FragileSiteLoader",
    "DataProcessor",
    "FeatureExtractor",
    "SplitGenerator",
    "ECDNADataset",
    "DynamicsDataset",
    "VulnerabilityDataset",
    "create_dataloader",
]
