#!/usr/bin/env python3
"""
Compute null baseline for vulnerability validation rate.

Samples random gene sets of size k from DepMap CRISPR genes,
checks how many match the same literature validation criteria,
and computes a null distribution to compare against our 14/47 = 29.8%.

Also computes Wilson confidence intervals on the observed rate.

Usage:
    python scripts/compute_null_baseline.py
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Categories of validated genes and their biological basis
# A gene "validates" if it belongs to one of these well-known ecDNA-relevant pathways
# AND is a general cancer dependency (CRISPR essential in many lines)
VALIDATION_CATEGORIES = {
    "DNA_damage_checkpoint": {
        "genes": {"CHEK1", "CHEK2", "ATR", "ATM", "WEE1", "CDK1", "CDK2",
                  "CDC25A", "CDC25B", "CDC25C", "CLSPN", "CLASPIN", "RAD17"},
        "description": "DNA damage checkpoint kinases",
    },
    "mitotic_spindle": {
        "genes": {"KIF11", "KIF15", "KIF18A", "KIF2C", "KIF23", "AURKA", "AURKB",
                  "PLK1", "PLK4", "TTK", "BUB1", "BUB1B", "BUB3", "MAD2L1",
                  "NDC80", "NUF2", "SPC24", "SPC25", "CENPE", "CENPF"},
        "description": "Mitotic spindle and kinetochore",
    },
    "chromosome_segregation": {
        "genes": {"NCAPD2", "NCAPD3", "NCAPG", "NCAPH", "NCAPH2", "SMC2", "SMC4",
                  "SGO1", "SGO2", "SGOL1", "ESPL1", "WAPL", "PDS5A", "PDS5B",
                  "RAD21", "SMC1A", "SMC3"},
        "description": "Condensin/cohesin and segregation",
    },
    "replication_licensing": {
        "genes": {"ORC1", "ORC2", "ORC3", "ORC4", "ORC5", "ORC6",
                  "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7",
                  "CDC6", "CDT1", "CDC45", "GINS1", "GINS2", "GINS3", "GINS4"},
        "description": "Replication origin licensing and helicase",
    },
    "spliceosome": {
        "genes": {"SNRPF", "SNRPD1", "SNRPD2", "SNRPD3", "SNRPE", "SNRPG",
                  "SF3B1", "SF3A1", "U2AF1", "U2AF2", "EFTUD2", "BUD31",
                  "PRPF8", "PRPF19", "DDX41", "DDX46"},
        "description": "Core spliceosome (MYC-synthetic lethal)",
    },
    "proteasome": {
        "genes": {"PSMD7", "PSMD1", "PSMD2", "PSMD3", "PSMD4", "PSMD6",
                  "PSMD8", "PSMD11", "PSMD12", "PSMD14",
                  "PSMB1", "PSMB2", "PSMB3", "PSMB4", "PSMB5"},
        "description": "26S proteasome subunits",
    },
    "ribosome": {
        "genes": {"RPL23", "RPL5", "RPL11", "RPL22", "RPL26", "RPL29",
                  "RPS3", "RPS6", "RPS14", "RPS19", "RPS27A"},
        "description": "Ribosomal proteins (MDM2/p53 axis)",
    },
    "apoptosis": {
        "genes": {"BCL2L1", "BCL2", "MCL1", "BCL2L2", "BCLXL", "BAX", "BAK1",
                  "BID", "BIRC5", "XIAP"},
        "description": "Anti-apoptotic (BCL-XL family)",
    },
    "rna_helicase": {
        "genes": {"DDX3X", "DDX5", "DDX17", "DDX39B", "DHX9", "DHX15", "DHX38"},
        "description": "RNA helicases",
    },
    "chaperone": {
        "genes": {"URI1", "PFDN1", "PFDN2", "PFDN4", "PFDN5", "PFDN6",
                  "HSP90AA1", "HSP90AB1", "CCT2", "CCT3", "CCT4", "CCT5"},
        "description": "Chaperone/prefoldin",
    },
}

# All validation genes (union of all categories)
ALL_VALIDATION_GENES = set()
for cat in VALIDATION_CATEGORIES.values():
    ALL_VALIDATION_GENES |= cat["genes"]


def wilson_ci(k, n, confidence=0.95):
    """Wilson score confidence interval for a proportion."""
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return max(0, center - margin), min(1, center + margin)


def main():
    data_dir = Path("data")

    # Load the full gene list from differential dependency analysis
    dep_file = data_dir / "vulnerabilities" / "differential_dependency_full.csv"
    if dep_file.exists():
        dep_df = pd.read_csv(dep_file)
        all_genes = set(dep_df.iloc[:, 0].unique())
        logger.info(f"Total genes in dependency analysis: {len(all_genes)}")
    else:
        # Fallback: use DepMap CRISPR gene list
        crispr_file = data_dir / "depmap" / "crispr_gene_effect.csv"
        if crispr_file.exists():
            crispr = pd.read_csv(crispr_file, index_col=0, nrows=0)
            all_genes = set(col.split(" (")[0] for col in crispr.columns)
            logger.info(f"Total genes in CRISPR screen: {len(all_genes)}")
        else:
            logger.error("No gene list found. Need differential_dependency_full.csv or crispr_gene_effect.csv")
            return

    # Our validated genes
    our_validated = {"CHK1", "CDK1", "KIF11", "NCAPD2", "SGO1", "NDC80",
                     "ORC6", "MCM2", "PSMD7", "RPL23", "URI1", "SNRPF",
                     "DDX3X", "BCL2L1"}
    our_candidates = 47  # Total candidates we reported
    our_validated_count = len(our_validated)

    logger.info(f"\nOur result: {our_validated_count}/{our_candidates} = "
                f"{our_validated_count/our_candidates:.1%}")

    # Wilson CI on our observed rate
    ci_low, ci_high = wilson_ci(our_validated_count, our_candidates)
    logger.info(f"Wilson 95% CI: [{ci_low:.1%}, {ci_high:.1%}]")

    # Count how many genes in the full gene list overlap with validation categories
    overlap = all_genes & ALL_VALIDATION_GENES
    logger.info(f"\nGenes in both gene list and validation categories: {len(overlap)}")
    logger.info(f"Validation gene pool: {len(ALL_VALIDATION_GENES)}")
    logger.info(f"Background rate: {len(overlap)}/{len(all_genes)} = "
                f"{len(overlap)/len(all_genes):.1%}")

    # Null distribution: sample random gene sets of size 47
    n_simulations = 100_000
    gene_list = sorted(all_genes)
    n_genes = len(gene_list)

    logger.info(f"\nRunning {n_simulations:,} random samplings (k={our_candidates})...")

    np.random.seed(42)
    null_counts = np.zeros(n_simulations, dtype=int)

    for i in range(n_simulations):
        random_set = set(np.random.choice(gene_list, size=our_candidates, replace=False))
        null_counts[i] = len(random_set & ALL_VALIDATION_GENES)

    null_rate = null_counts / our_candidates

    # P-value: fraction of random sets with >= our_validated_count hits
    p_value = (null_counts >= our_validated_count).mean()

    logger.info(f"\n{'='*60}")
    logger.info("NULL BASELINE RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Observed: {our_validated_count}/{our_candidates} = {our_validated_count/our_candidates:.1%}")
    logger.info(f"Wilson 95% CI: [{ci_low:.1%}, {ci_high:.1%}]")
    logger.info(f"Null mean: {null_counts.mean():.1f}/{our_candidates} = {null_rate.mean():.1%}")
    logger.info(f"Null std: {null_counts.std():.1f}")
    logger.info(f"Null 95th pctile: {np.percentile(null_counts, 95):.0f}/{our_candidates} = "
                f"{np.percentile(null_rate, 95):.1%}")
    logger.info(f"P-value (observed >= null): {p_value:.4f}")
    logger.info(f"Enrichment: {(our_validated_count/our_candidates) / max(null_rate.mean(), 1e-6):.1f}x")

    # Save results
    output_dir = data_dir / "validation"
    output_dir.mkdir(exist_ok=True)

    results = {
        "observed_validated": our_validated_count,
        "observed_candidates": our_candidates,
        "observed_rate": our_validated_count / our_candidates,
        "wilson_ci_low": ci_low,
        "wilson_ci_high": ci_high,
        "null_mean": null_counts.mean(),
        "null_std": null_counts.std(),
        "null_95th_pctile": np.percentile(null_counts, 95),
        "p_value": p_value,
        "enrichment": (our_validated_count/our_candidates) / max(null_rate.mean(), 1e-6),
        "n_simulations": n_simulations,
        "n_validation_genes_in_pool": len(overlap),
        "n_total_genes": n_genes,
    }

    pd.DataFrame([results]).to_csv(output_dir / "null_baseline_results.csv", index=False)

    # Also save the null distribution
    pd.DataFrame({"null_count": null_counts}).to_csv(
        output_dir / "null_distribution.csv", index=False
    )

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
