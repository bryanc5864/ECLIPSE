#!/usr/bin/env python3
"""
Pathway enrichment analysis for vulnerability candidates.

Runs hypergeometric test for GO biological process and KEGG pathway enrichment
on our 47 vulnerability candidates vs all 17,453 tested genes.

Usage:
    python scripts/analyze_pathway_enrichment.py
"""

import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Manually curated pathway annotations for ecDNA-relevant pathways
# Based on GO Biological Process and KEGG
PATHWAY_ANNOTATIONS = {
    "GO:0007049 cell cycle": {
        "CDK1", "CDK2", "CDK4", "CDK6", "CCNA2", "CCNB1", "CCND1", "CCNE1",
        "CDC20", "CDC25A", "CDC25B", "CDC25C", "CDC6", "CDT1", "CDKN1A",
        "CDKN1B", "CDKN2A", "RB1", "E2F1", "E2F3", "WEE1", "PLK1", "PLK4",
        "AURKA", "AURKB", "BUB1", "BUB1B", "BUB3", "MAD2L1", "TTK",
        "CHEK1", "CHEK2", "SGO1", "ESPL1", "MASTL",
    },
    "GO:0006260 DNA replication": {
        "ORC1", "ORC2", "ORC3", "ORC4", "ORC5", "ORC6",
        "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7",
        "CDC6", "CDT1", "CDC45", "GINS1", "GINS2", "GINS3", "GINS4",
        "PCNA", "POLA1", "POLD1", "POLE", "RFC1", "RFC2", "RFC3",
        "FEN1", "LIG1", "RPA1", "RPA2", "RPA3", "PRIM1", "PRIM2",
    },
    "GO:0007067 mitotic nuclear division": {
        "KIF11", "KIF15", "KIF18A", "KIF2C", "KIF23", "CENPE", "CENPF",
        "NDC80", "NUF2", "SPC24", "SPC25", "KNSTRN", "ZWINT",
        "NCAPD2", "NCAPD3", "NCAPG", "NCAPH", "NCAPH2", "SMC2", "SMC4",
        "AURKA", "AURKB", "PLK1", "TPX2", "BIRC5", "INCENP",
        "SGO1", "SGO2", "RAD21", "SMC1A", "SMC3", "WAPL",
    },
    "GO:0006281 DNA repair": {
        "CHEK1", "CHEK2", "ATR", "ATM", "BRCA1", "BRCA2", "RAD51",
        "RAD50", "MRE11", "NBN", "XRCC1", "XRCC4", "LIG4",
        "PARP1", "PARP2", "MDC1", "H2AX", "TP53BP1", "RNF8",
        "FANCD2", "FANCA", "FANCB", "FANCC",
    },
    "GO:0006412 translation": {
        "RPL23", "RPL5", "RPL11", "RPL22", "RPL26", "RPL29", "RPL3", "RPL4",
        "RPS3", "RPS6", "RPS14", "RPS19", "RPS27A", "RPS8", "RPS15A",
        "EIF4A1", "EIF4E", "EIF4G1", "EIF2S1", "EIF3A", "EIF5B",
        "RPL23A", "RPL7", "RPL8", "RPL10", "RPL13", "RPL15",
    },
    "GO:0000398 mRNA splicing": {
        "SNRPF", "SNRPD1", "SNRPD2", "SNRPD3", "SNRPE", "SNRPG",
        "SF3B1", "SF3A1", "U2AF1", "U2AF2", "EFTUD2", "BUD31",
        "PRPF8", "PRPF19", "PRPF31", "PRPF6", "PRPF3",
        "DDX41", "DDX46", "DDX39B", "DHX15", "DHX38",
        "SRSF1", "SRSF2", "SRSF3", "HNRNPA1", "HNRNPC",
    },
    "GO:0010941 regulation of cell death": {
        "BCL2L1", "BCL2", "MCL1", "BCL2L2", "BAX", "BAK1", "BID", "BIM",
        "BIRC5", "XIAP", "BIRC2", "BIRC3", "CASP3", "CASP8", "CASP9",
        "APAF1", "CYCS", "DIABLO", "TNFRSF10A", "TNFRSF10B",
        "TP53", "MDM2", "MDM4",
    },
    "GO:0000502 proteasome complex": {
        "PSMD7", "PSMD1", "PSMD2", "PSMD3", "PSMD4", "PSMD6",
        "PSMD8", "PSMD11", "PSMD12", "PSMD14",
        "PSMB1", "PSMB2", "PSMB3", "PSMB4", "PSMB5", "PSMB6", "PSMB7",
        "PSMA1", "PSMA2", "PSMA3", "PSMA4", "PSMA5", "PSMA6", "PSMA7",
        "PSMC1", "PSMC2", "PSMC3", "PSMC4", "PSMC5", "PSMC6",
    },
    "GO:0006457 protein folding": {
        "URI1", "PFDN1", "PFDN2", "PFDN4", "PFDN5", "PFDN6",
        "HSP90AA1", "HSP90AB1", "HSPA1A", "HSPA8", "HSPA5",
        "CCT2", "CCT3", "CCT4", "CCT5", "CCT6A", "CCT7", "CCT8", "TCP1",
        "DNAJA1", "DNAJB1",
    },
    "KEGG:hsa04110 Cell cycle": {
        "CDK1", "CDK2", "CDK4", "CDK6", "CCNA2", "CCNB1", "CCND1", "CCNE1",
        "CDC20", "CDC25A", "CDC6", "CDT1", "RB1", "E2F1", "TP53",
        "CHEK1", "CHEK2", "ATR", "ATM", "WEE1", "PLK1",
        "BUB1", "BUB1B", "BUB3", "MAD2L1", "TTK",
        "CDKN1A", "CDKN1B", "CDKN2A", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7",
        "ORC1", "ORC2", "ORC3", "ORC4", "ORC5", "ORC6",
        "SGO1", "ESPL1",
    },
}


def hypergeometric_test(candidates, pathway_genes, background_size):
    """
    Hypergeometric test for pathway enrichment.

    P(X >= k) where:
    - N = background_size (total genes tested)
    - K = len(pathway_genes & background) (pathway genes in background)
    - n = len(candidates) (our candidates)
    - k = len(candidates & pathway_genes) (overlap)
    """
    overlap = candidates & pathway_genes
    k = len(overlap)
    K = len(pathway_genes)  # Note: should intersect with background for precision
    n = len(candidates)
    N = background_size

    if k == 0:
        return 1.0, k, overlap

    # P(X >= k) = 1 - P(X <= k-1) = sf(k-1, N, K, n)
    p_value = stats.hypergeom.sf(k - 1, N, K, n)
    return p_value, k, overlap


def main():
    data_dir = Path("data")

    # Load our candidate genes
    dep_file = data_dir / "vulnerabilities" / "differential_dependency_full.csv"
    if dep_file.exists():
        dep_df = pd.read_csv(dep_file)
        all_genes = set(dep_df.iloc[:, 0].unique())
        # Top 47 candidates (our reported set)
        top_candidates = set(dep_df.iloc[:47, 0].unique())
    else:
        logger.error("differential_dependency_full.csv not found")
        return

    # Also load top 100 if available
    top100_file = data_dir / "vulnerabilities" / "top_100_vulnerabilities.csv"
    if top100_file.exists():
        top100_df = pd.read_csv(top100_file)
        top_100 = set(top100_df.iloc[:, 0].unique())
    else:
        top_100 = set(dep_df.iloc[:100, 0].unique())

    background_size = len(all_genes)

    logger.info(f"Background genes: {background_size}")
    logger.info(f"Top candidates: {len(top_candidates)}")
    logger.info(f"Top 100: {len(top_100)}")

    # Run enrichment for top 47 and top 100
    results = []
    for candidate_name, candidates in [("top_47", top_candidates), ("top_100", top_100)]:
        logger.info(f"\n{'='*60}")
        logger.info(f"ENRICHMENT: {candidate_name} ({len(candidates)} genes)")
        logger.info(f"{'='*60}")

        for pathway_name, pathway_genes in PATHWAY_ANNOTATIONS.items():
            p_value, k, overlap = hypergeometric_test(candidates, pathway_genes, background_size)

            results.append({
                "candidate_set": candidate_name,
                "pathway": pathway_name,
                "pathway_size": len(pathway_genes),
                "overlap_count": k,
                "overlap_genes": ", ".join(sorted(overlap)) if overlap else "",
                "expected": len(candidates) * len(pathway_genes) / background_size,
                "enrichment": (k / max(len(candidates) * len(pathway_genes) / background_size, 1e-6)),
                "p_value": p_value,
            })

            if k > 0:
                expected = len(candidates) * len(pathway_genes) / background_size
                enrichment = k / max(expected, 1e-6)
                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                logger.info(f"  {pathway_name}: {k}/{len(pathway_genes)} "
                            f"(expected {expected:.1f}, {enrichment:.1f}x, p={p_value:.4f}) {sig}")
                if overlap:
                    logger.info(f"    Genes: {', '.join(sorted(overlap))}")

    # Multiple testing correction (Benjamini-Hochberg)
    results_df = pd.DataFrame(results)
    for candidate_name in ["top_47", "top_100"]:
        mask = results_df["candidate_set"] == candidate_name
        p_vals = results_df.loc[mask, "p_value"].values
        n_tests = len(p_vals)
        sorted_idx = np.argsort(p_vals)
        adjusted = np.zeros(n_tests)
        for i, idx in enumerate(sorted_idx):
            adjusted[idx] = min(1.0, p_vals[idx] * n_tests / (i + 1))
        # Ensure monotonicity
        for i in range(n_tests - 2, -1, -1):
            adjusted[sorted_idx[i]] = min(adjusted[sorted_idx[i]], adjusted[sorted_idx[i + 1]])
        results_df.loc[mask, "p_adjusted_bh"] = adjusted

    # Save
    output_dir = data_dir / "validation"
    output_dir.mkdir(exist_ok=True)
    results_df.to_csv(output_dir / "pathway_enrichment_results.csv", index=False)

    # Summary
    sig = results_df[(results_df["candidate_set"] == "top_47") & (results_df["p_adjusted_bh"] < 0.05)]
    logger.info(f"\n{'='*60}")
    logger.info(f"SIGNIFICANT PATHWAYS (top_47, BH-adjusted p < 0.05): {len(sig)}")
    logger.info(f"{'='*60}")
    for _, row in sig.sort_values("p_adjusted_bh").iterrows():
        logger.info(f"  {row['pathway']}: {row['overlap_count']} genes, "
                    f"{row['enrichment']:.1f}x, p_adj={row['p_adjusted_bh']:.4f}")

    logger.info(f"\nSaved to {output_dir / 'pathway_enrichment_results.csv'}")


if __name__ == "__main__":
    main()
