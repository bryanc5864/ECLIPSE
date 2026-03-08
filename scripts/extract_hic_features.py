#!/usr/bin/env python3
"""
Extract Hi-C topology features for ecDNA prediction.
Fast version - uses precomputed TAD boundaries and simple contact metrics.
"""

import numpy as np
import pandas as pd
import cooler
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Oncogene locations (hg38) - midpoint positions
ONCOGENE_LOCI = {
    'MYC': ('chr8', 127739192),
    'MYCN': ('chr2', 15943778),
    'EGFR': ('chr7', 55115322),
    'ERBB2': ('chr17', 39709170),
    'CDK4': ('chr12', 57750087),
    'CDK6': ('chr7', 92350071),
    'MDM2': ('chr12', 68829429),
    'MDM4': ('chr1', 204506377),
    'CCND1': ('chr11', 69647815),
    'CCNE1': ('chr19', 29818154),
    'FGFR1': ('chr8', 38439986),
    'FGFR2': ('chr10', 121538395),
    'MET': ('chr7', 116735286),
    'PDGFRA': ('chr4', 54263671),
    'TERT': ('chr5', 1274107),
    'AR': ('chrX', 67637325),
    'BRAF': ('chr7', 140822128),
    'KRAS': ('chr12', 25228087),
    'PIK3CA': ('chr3', 179194103),
    'KIT': ('chr4', 54699316),
}

# Known TAD boundaries near oncogenes (from literature/ENCODE)
# These are approximate positions where TAD boundaries occur
TAD_BOUNDARIES = {
    'chr8_MYC': [127200000, 128300000],  # MYC is in a TAD
    'chr7_EGFR': [54500000, 55800000],
    'chr12_CDK4_MDM2': [57200000, 58000000, 68500000, 69500000],
    'chr17_ERBB2': [39200000, 40200000],
}


def load_hic(hic_file: str, resolution: int = 50000):
    """Load Hi-C data."""
    logger.info(f"Loading Hi-C from {hic_file}...")

    available = cooler.fileops.list_coolers(hic_file)

    # Find closest resolution
    for res_path in available:
        res = res_path.split('/')[-1]
        if res.isdigit() and int(res) >= resolution:
            clr = cooler.Cooler(f"{hic_file}::{res_path}")
            logger.info(f"Using resolution: {clr.binsize}bp")
            return clr

    # Default to first available
    clr = cooler.Cooler(f"{hic_file}::{available[0]}")
    return clr


def get_local_contact_sum(clr, chrom, pos, window=500000):
    """Get sum of contacts in local window - proxy for accessibility."""
    try:
        resolution = clr.binsize
        bin_idx = pos // resolution
        window_bins = window // resolution

        start = max(0, bin_idx - window_bins) * resolution
        end = (bin_idx + window_bins) * resolution

        region = f"{chrom}:{start}-{end}"
        matrix = clr.matrix(balance=False).fetch(region)

        # Sum of contacts (log scale)
        total = np.nansum(matrix)
        return np.log1p(total)
    except:
        return 0.0


def get_long_range_contacts(clr, chrom, pos, min_dist=1000000):
    """Get fraction of long-range contacts (>1Mb) - indicates looping."""
    try:
        resolution = clr.binsize
        center_bin = pos // resolution

        # Get row of contacts
        matrix = clr.matrix(balance=False).fetch(chrom)
        local_bin = center_bin - clr.offset(chrom)

        if local_bin < 0 or local_bin >= matrix.shape[0]:
            return 0.0

        row = matrix[local_bin, :]

        # Count long-range vs short-range
        min_dist_bins = min_dist // resolution

        short_range = np.nansum(row[max(0, local_bin-min_dist_bins):local_bin+min_dist_bins])
        long_range = np.nansum(row) - short_range

        if short_range + long_range > 0:
            return long_range / (short_range + long_range)
        return 0.0
    except:
        return 0.0


def extract_hic_features(hic_file: str):
    """Extract Hi-C features for all oncogenes."""

    clr = load_hic(hic_file)
    features = {}

    # 1. Local contact density for each oncogene
    logger.info("Computing local contact density...")
    contact_densities = []
    for gene, (chrom, pos) in ONCOGENE_LOCI.items():
        density = get_local_contact_sum(clr, chrom, pos)
        features[f'hic_density_{gene}'] = density
        contact_densities.append(density)

    # 2. Long-range contact fraction
    logger.info("Computing long-range contact fractions...")
    long_range_fracs = []
    for gene, (chrom, pos) in ONCOGENE_LOCI.items():
        frac = get_long_range_contacts(clr, chrom, pos)
        features[f'hic_longrange_{gene}'] = frac
        long_range_fracs.append(frac)

    # 3. Summary statistics
    features['hic_density_mean'] = np.mean(contact_densities)
    features['hic_density_std'] = np.std(contact_densities)
    features['hic_density_max'] = np.max(contact_densities)
    features['hic_longrange_mean'] = np.mean(long_range_fracs)
    features['hic_longrange_std'] = np.std(long_range_fracs)

    # 4. Relative features (normalized by genome average)
    genome_mean = features['hic_density_mean']
    if genome_mean > 0:
        for gene in ONCOGENE_LOCI:
            features[f'hic_density_rel_{gene}'] = features[f'hic_density_{gene}'] / genome_mean

    return features


def main():
    data_dir = Path("data")
    hic_file = data_dir / "hic" / "GM12878.mcool"

    if not hic_file.exists():
        logger.error(f"Hi-C file not found: {hic_file}")
        return None

    features = extract_hic_features(str(hic_file))

    logger.info(f"\n=== Hi-C Features ({len(features)}) ===")
    for name, value in sorted(features.items())[:20]:
        logger.info(f"  {name}: {value:.4f}")
    logger.info(f"  ... and {len(features) - 20} more")

    # Save
    output_file = data_dir / "features" / "hic_features.npz"
    np.savez(output_file, **features)
    logger.info(f"\nSaved to {output_file}")

    return features


if __name__ == "__main__":
    main()
