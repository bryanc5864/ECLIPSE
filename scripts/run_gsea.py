#!/usr/bin/env python3
"""
Gene Set Enrichment Analysis (GSEA) for vulnerability candidates (Module 3).

Implements standard GSEA (Subramanian et al. 2005) using ranked gene lists
from differential dependency analysis. Compares pathway-level enrichment
with the existing hypergeometric test results.

Pathway-level FDR can be significant even when no individual gene survives
gene-level FDR, because GSEA detects coordinated shifts across gene sets.

Usage:
    python scripts/run_gsea.py --n-permutations 10000
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_pathway_enrichment import PATHWAY_ANNOTATIONS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_enrichment_score(ranked_genes, gene_set, weights=None):
    """
    Compute GSEA enrichment score (running-sum KS statistic).

    Args:
        ranked_genes: List of gene names, ranked by metric (most extreme first).
        gene_set: Set of genes in the pathway.
        weights: Optional array of absolute metric values for weighted scoring.

    Returns:
        (es, running_scores): Enrichment score and running sum array.
    """
    n = len(ranked_genes)
    hits = np.array([1 if g in gene_set else 0 for g in ranked_genes])

    n_hit = hits.sum()
    n_miss = n - n_hit

    if n_hit == 0 or n_miss == 0:
        return 0.0, np.zeros(n)

    # Weighted scoring (Subramanian et al. 2005)
    if weights is not None:
        hit_weights = hits * np.abs(weights)
        hit_norm = hit_weights.sum()
    else:
        hit_weights = hits.astype(float)
        hit_norm = n_hit

    miss_norm = n_miss

    # Running sum
    running = np.zeros(n)
    for i in range(n):
        if hits[i]:
            running[i] = (running[i - 1] if i > 0 else 0) + hit_weights[i] / hit_norm
        else:
            running[i] = (running[i - 1] if i > 0 else 0) - 1.0 / miss_norm

    # ES = max deviation from zero
    es_pos = running.max()
    es_neg = running.min()
    es = es_pos if abs(es_pos) >= abs(es_neg) else es_neg

    return es, running


def gsea_permutation_test(ranked_genes, gene_set, weights, observed_es,
                          n_permutations=10000, seed=42):
    """
    Permutation test for GSEA enrichment score.

    Permutes gene labels (not the ranking) to generate a null distribution.
    """
    rng = np.random.RandomState(seed)
    n = len(ranked_genes)

    null_es = np.zeros(n_permutations)
    gene_array = np.array(ranked_genes)

    for i in range(n_permutations):
        perm_genes = rng.permutation(gene_array)
        es, _ = compute_enrichment_score(perm_genes, gene_set, weights)
        null_es[i] = es

    # Two-sided p-value
    if observed_es >= 0:
        p_value = (null_es >= observed_es).mean()
    else:
        p_value = (null_es <= observed_es).mean()

    # Use minimum p-value floor to avoid p=0
    p_value = max(p_value, 1.0 / (n_permutations + 1))

    return p_value, null_es


def bh_correction(p_values):
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    adjusted = np.zeros(n)

    for i, idx in enumerate(sorted_idx):
        adjusted[idx] = min(1.0, p_values[idx] * n / (i + 1))

    # Ensure monotonicity
    for i in range(n - 2, -1, -1):
        adjusted[sorted_idx[i]] = min(adjusted[sorted_idx[i]], adjusted[sorted_idx[i + 1]])

    return adjusted


def main():
    parser = argparse.ArgumentParser(description="Gene Set Enrichment Analysis")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load differential dependency data
    dep_file = data_dir / "vulnerabilities" / "differential_dependency_full.csv"
    if not dep_file.exists():
        logger.error(f"Differential dependency file not found: {dep_file}")
        sys.exit(1)

    dep_df = pd.read_csv(dep_file)
    logger.info(f"Loaded {len(dep_df)} genes from differential dependency analysis")

    # Determine gene and effect size columns
    gene_col = dep_df.columns[0]
    # Look for effect size column
    effect_cols = [c for c in dep_df.columns if 'effect' in c.lower() or 'cohen' in c.lower()
                   or 'diff' in c.lower() or 'stat' in c.lower()]
    if effect_cols:
        effect_col = effect_cols[0]
    else:
        # Fallback: use second column
        effect_col = dep_df.columns[1]
    logger.info(f"Using gene column: '{gene_col}', effect column: '{effect_col}'")

    # Clean gene names (remove parenthetical ENTREZ IDs if present)
    dep_df['gene_clean'] = dep_df[gene_col].astype(str).str.split(r' \(', regex=True).str[0]

    # Rank genes by effect size (most negative = most dependent in ecDNA+)
    dep_df = dep_df.sort_values(effect_col, ascending=True)
    ranked_genes = dep_df['gene_clean'].tolist()
    effect_sizes = dep_df[effect_col].values.astype(float)

    all_genes = set(ranked_genes)
    logger.info(f"Ranked {len(ranked_genes)} genes by {effect_col}")
    logger.info(f"Effect size range: [{effect_sizes.min():.4f}, {effect_sizes.max():.4f}]")

    # Run GSEA for each pathway
    logger.info(f"\n{'='*60}")
    logger.info(f"GSEA ({args.n_permutations} permutations)")
    logger.info(f"{'='*60}")

    results = []
    for pathway_name, pathway_genes in PATHWAY_ANNOTATIONS.items():
        # Intersect pathway genes with our tested genes
        overlap_genes = pathway_genes & all_genes
        if len(overlap_genes) < 2:
            logger.info(f"  {pathway_name}: <2 genes overlap, skipping")
            continue

        # Compute observed enrichment score
        es, running = compute_enrichment_score(ranked_genes, overlap_genes, effect_sizes)

        # Permutation test
        p_value, null_es = gsea_permutation_test(
            ranked_genes, overlap_genes, effect_sizes, es,
            n_permutations=args.n_permutations, seed=args.seed,
        )

        # Normalized enrichment score (NES)
        if es >= 0:
            pos_null = null_es[null_es >= 0]
            nes = es / max(pos_null.mean(), 1e-8) if len(pos_null) > 0 else 0.0
        else:
            neg_null = null_es[null_es < 0]
            nes = -es / max(abs(neg_null.mean()), 1e-8) if len(neg_null) > 0 else 0.0

        # Find leading edge genes (those before the max running-sum point)
        peak_idx = np.argmax(np.abs(running))
        leading_edge = [g for g in ranked_genes[:peak_idx + 1] if g in overlap_genes]

        results.append({
            "pathway": pathway_name,
            "pathway_size": len(pathway_genes),
            "genes_in_data": len(overlap_genes),
            "enrichment_score": es,
            "normalized_es": nes,
            "p_value": p_value,
            "leading_edge_count": len(leading_edge),
            "leading_edge_genes": ", ".join(leading_edge[:20]),
        })

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        logger.info(f"  {pathway_name}: ES={es:.3f}, NES={nes:.3f}, p={p_value:.4f} {sig}")

    # BH correction
    gsea_df = pd.DataFrame(results)
    gsea_df["fdr"] = bh_correction(gsea_df["p_value"].values)

    # Sort by FDR
    gsea_df = gsea_df.sort_values("fdr")

    # Save GSEA results
    output_dir = data_dir / "validation"
    output_dir.mkdir(exist_ok=True)
    gsea_df.to_csv(output_dir / "gsea_results.csv", index=False)

    # ---------- Compare with hypergeometric results ----------
    logger.info(f"\n{'='*60}")
    logger.info("GSEA vs Hypergeometric Comparison")
    logger.info(f"{'='*60}")

    hyper_file = output_dir / "pathway_enrichment_results.csv"
    if hyper_file.exists():
        hyper_df = pd.read_csv(hyper_file)
        # Use top_47 results
        hyper_47 = hyper_df[hyper_df["candidate_set"] == "top_47"].copy()

        comparison_rows = []
        for _, gsea_row in gsea_df.iterrows():
            pathway = gsea_row["pathway"]
            hyper_match = hyper_47[hyper_47["pathway"] == pathway]

            comparison_rows.append({
                "pathway": pathway,
                "gsea_es": gsea_row["enrichment_score"],
                "gsea_nes": gsea_row["normalized_es"],
                "gsea_p": gsea_row["p_value"],
                "gsea_fdr": gsea_row["fdr"],
                "hyper_overlap": hyper_match["overlap_count"].values[0] if len(hyper_match) else 0,
                "hyper_enrichment": hyper_match["enrichment"].values[0] if len(hyper_match) else 0,
                "hyper_p": hyper_match["p_value"].values[0] if len(hyper_match) else 1.0,
                "hyper_fdr": hyper_match["p_adjusted_bh"].values[0] if len(hyper_match) else 1.0,
            })

        comp_df = pd.DataFrame(comparison_rows)
        comp_df.to_csv(output_dir / "gsea_vs_hypergeometric.csv", index=False)

        logger.info(f"\n{'Pathway':<45s} {'GSEA FDR':>10s} {'Hyper FDR':>10s}")
        logger.info("-" * 70)
        for _, row in comp_df.iterrows():
            gsea_sig = "*" if row["gsea_fdr"] < 0.05 else " "
            hyper_sig = "*" if row["hyper_fdr"] < 0.05 else " "
            logger.info(f"  {row['pathway']:<43s} {row['gsea_fdr']:>8.4f}{gsea_sig} "
                        f"{row['hyper_fdr']:>8.4f}{hyper_sig}")
    else:
        logger.info("Hypergeometric results not found; skipping comparison")

    # ---------- Summary ----------
    logger.info(f"\n{'='*60}")
    logger.info("GSEA SUMMARY")
    logger.info(f"{'='*60}")

    sig_gsea = gsea_df[gsea_df["fdr"] < 0.05]
    logger.info(f"Pathways significant at FDR < 0.05: {len(sig_gsea)}/{len(gsea_df)}")
    for _, row in sig_gsea.iterrows():
        logger.info(f"  {row['pathway']}: NES={row['normalized_es']:.3f}, FDR={row['fdr']:.4f}")
        if row['leading_edge_genes']:
            logger.info(f"    Leading edge: {row['leading_edge_genes']}")

    logger.info(f"\nResults saved to:")
    logger.info(f"  {output_dir / 'gsea_results.csv'}")
    if hyper_file.exists():
        logger.info(f"  {output_dir / 'gsea_vs_hypergeometric.csv'}")


if __name__ == "__main__":
    main()
