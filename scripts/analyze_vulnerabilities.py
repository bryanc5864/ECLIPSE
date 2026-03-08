#!/usr/bin/env python3
"""
Module 3: VulnCausal - Differential Dependency Analysis

Find genes that selectively kill ecDNA+ cancer cells using CRISPR data.
This is the baseline analysis before building the full causal model.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_crispr_data(data_dir: Path):
    """Load CRISPR gene effect scores from DepMap."""
    logger.info("Loading CRISPR data...")
    crispr = pd.read_csv(data_dir / "depmap" / "crispr.csv", index_col=0)
    logger.info(f"  CRISPR shape: {crispr.shape}")
    logger.info(f"  Cell lines: {crispr.shape[0]}, Genes: {crispr.shape[1]}")
    return crispr


def load_ecdna_labels(data_dir: Path):
    """Load ecDNA labels from CytoCellDB."""
    logger.info("Loading ecDNA labels...")
    cyto = pd.read_excel(data_dir / "cytocell_db" / "CytoCellDB_Supp_File1.xlsx")

    # Create mapping from DepMap_ID to ecDNA status
    labels = cyto[['DepMap_ID', 'ECDNA', 'lineage', 'primary_disease']].copy()
    labels = labels.dropna(subset=['DepMap_ID'])
    labels['is_ecdna'] = (labels['ECDNA'] == 'Y').astype(int)
    labels = labels.set_index('DepMap_ID')

    logger.info(f"  Total labeled: {len(labels)}")
    logger.info(f"  ecDNA+: {labels['is_ecdna'].sum()}")
    logger.info(f"  ecDNA-: {(labels['is_ecdna'] == 0).sum()}")

    return labels


def differential_dependency_analysis(crispr: pd.DataFrame, labels: pd.DataFrame):
    """
    Find genes where ecDNA+ cells are more dependent (more negative CRISPR score).

    More negative CRISPR score = gene is more essential for survival.
    """
    logger.info("Running differential dependency analysis...")

    # Find common samples
    common = crispr.index.intersection(labels.index)
    logger.info(f"  Common samples: {len(common)}")

    crispr_aligned = crispr.loc[common]
    labels_aligned = labels.loc[common]

    # Split by ecDNA status
    ecdna_pos_idx = labels_aligned[labels_aligned['is_ecdna'] == 1].index
    ecdna_neg_idx = labels_aligned[labels_aligned['is_ecdna'] == 0].index

    logger.info(f"  ecDNA+ samples with CRISPR: {len(ecdna_pos_idx)}")
    logger.info(f"  ecDNA- samples with CRISPR: {len(ecdna_neg_idx)}")

    ecdna_pos = crispr_aligned.loc[ecdna_pos_idx]
    ecdna_neg = crispr_aligned.loc[ecdna_neg_idx]

    # Test each gene
    results = []
    n_genes = len(crispr_aligned.columns)

    for i, gene in enumerate(crispr_aligned.columns):
        if i % 2000 == 0:
            logger.info(f"  Processing gene {i}/{n_genes}...")

        pos_scores = ecdna_pos[gene].dropna()
        neg_scores = ecdna_neg[gene].dropna()

        if len(pos_scores) < 10 or len(neg_scores) < 10:
            continue

        # Mann-Whitney U test (one-sided: ecDNA+ more dependent = more negative)
        try:
            stat, pval = stats.mannwhitneyu(pos_scores, neg_scores, alternative='less')
        except:
            continue

        # Effect size (negative = ecDNA-specific vulnerability)
        effect_size = pos_scores.mean() - neg_scores.mean()

        # Cohen's d
        pooled_std = np.sqrt((pos_scores.std()**2 + neg_scores.std()**2) / 2)
        cohens_d = effect_size / pooled_std if pooled_std > 0 else 0

        # Parse gene name
        gene_name = gene.split(' (')[0] if ' (' in gene else gene

        results.append({
            'gene': gene_name,
            'gene_full': gene,
            'effect_size': effect_size,
            'cohens_d': cohens_d,
            'pvalue': pval,
            'ecdna_pos_mean': pos_scores.mean(),
            'ecdna_neg_mean': neg_scores.mean(),
            'ecdna_pos_std': pos_scores.std(),
            'ecdna_neg_std': neg_scores.std(),
            'n_pos': len(pos_scores),
            'n_neg': len(neg_scores),
        })

    results_df = pd.DataFrame(results)

    # Multiple testing correction
    results_df['fdr'] = stats.false_discovery_control(results_df['pvalue'], method='bh')

    # Sort by effect size (most negative = most ecDNA-specific)
    results_df = results_df.sort_values('effect_size')

    return results_df


def categorize_vulnerabilities(results_df: pd.DataFrame):
    """Categorize vulnerabilities by biological function."""

    # Known categories of potential ecDNA vulnerabilities
    categories = {
        'DNA_replication': ['POLA1', 'POLA2', 'POLB', 'POLD1', 'POLD2', 'POLE', 'POLE2',
                           'MCM2', 'MCM3', 'MCM4', 'MCM5', 'MCM6', 'MCM7', 'MCM8', 'MCM10',
                           'ORC1', 'ORC2', 'ORC3', 'ORC4', 'ORC5', 'ORC6',
                           'CDC6', 'CDC45', 'CDT1', 'GINS1', 'GINS2', 'GINS3', 'GINS4',
                           'PCNA', 'RFC1', 'RFC2', 'RFC3', 'RFC4', 'RFC5'],
        'DNA_repair': ['BRCA1', 'BRCA2', 'RAD51', 'RAD52', 'PARP1', 'PARP2',
                       'ATM', 'ATR', 'CHEK1', 'CHEK2', 'TP53BP1', 'XRCC1', 'XRCC2',
                       'FANCA', 'FANCD2', 'FANCM', 'BLM', 'WRN', 'RECQL4'],
        'Topoisomerase': ['TOP1', 'TOP2A', 'TOP2B', 'TOP3A', 'TOP3B'],
        'Chromatin': ['SMARCA4', 'SMARCA2', 'ARID1A', 'ARID1B', 'ATRX',
                      'KMT2A', 'KMT2D', 'KDM6A', 'EZH2', 'SUZ12',
                      'HDAC1', 'HDAC2', 'HDAC3', 'BRD4', 'BRD2'],
        'Cell_cycle': ['CDK1', 'CDK2', 'CDK4', 'CDK6', 'CCNA2', 'CCNB1', 'CCND1', 'CCNE1',
                       'PLK1', 'PLK4', 'AURKA', 'AURKB', 'BUB1', 'MAD2L1'],
        'Transcription': ['MYC', 'MYCN', 'MAX', 'MXD1', 'EP300', 'CREBBP',
                          'MED1', 'MED12', 'CDK7', 'CDK8', 'CDK9'],
        'Mitosis': ['KIF11', 'CENPE', 'CENPF', 'NDC80', 'NUF2', 'SPC24', 'SPC25',
                    'BUB1B', 'BUBR1', 'TTK', 'SGOL1'],
    }

    def get_category(gene):
        for cat, genes in categories.items():
            if gene in genes:
                return cat
        return 'Other'

    results_df['category'] = results_df['gene'].apply(get_category)

    return results_df


def main():
    data_dir = Path("data")
    output_dir = data_dir / "vulnerabilities"
    output_dir.mkdir(exist_ok=True)

    # Load data
    crispr = load_crispr_data(data_dir)
    labels = load_ecdna_labels(data_dir)

    # Run analysis
    results = differential_dependency_analysis(crispr, labels)

    # Categorize
    results = categorize_vulnerabilities(results)

    # Save full results
    results.to_csv(output_dir / "differential_dependency_full.csv", index=False)
    logger.info(f"\nSaved full results to {output_dir / 'differential_dependency_full.csv'}")

    # Print summary
    logger.info("\n" + "="*70)
    logger.info("TOP 30 ecDNA-SPECIFIC VULNERABILITIES")
    logger.info("(More negative effect = more essential in ecDNA+ cells)")
    logger.info("="*70)

    top_hits = results.head(30)
    print(f"\n{'Gene':<12} {'Effect':<10} {'Cohen d':<10} {'FDR':<12} {'Category':<20}")
    print("-" * 70)
    for _, row in top_hits.iterrows():
        print(f"{row['gene']:<12} {row['effect_size']:<10.4f} {row['cohens_d']:<10.3f} "
              f"{row['fdr']:<12.2e} {row['category']:<20}")

    # Significant hits
    sig_hits = results[results['fdr'] < 0.05]
    logger.info(f"\n\nSignificant hits (FDR < 0.05): {len(sig_hits)}")

    # By category
    logger.info("\n\nTop hits by category:")
    for cat in results['category'].unique():
        cat_hits = results[(results['category'] == cat) & (results['fdr'] < 0.1)]
        if len(cat_hits) > 0:
            logger.info(f"\n{cat}:")
            for _, row in cat_hits.head(5).iterrows():
                logger.info(f"  {row['gene']}: effect={row['effect_size']:.4f}, FDR={row['fdr']:.2e}")

    # Save top hits
    top_100 = results.head(100)
    top_100.to_csv(output_dir / "top_100_vulnerabilities.csv", index=False)
    logger.info(f"\nSaved top 100 to {output_dir / 'top_100_vulnerabilities.csv'}")

    # Summary statistics
    logger.info("\n\n=== SUMMARY ===")
    logger.info(f"Total genes tested: {len(results)}")
    logger.info(f"Significant (FDR < 0.05): {len(results[results['fdr'] < 0.05])}")
    logger.info(f"Significant (FDR < 0.10): {len(results[results['fdr'] < 0.10])}")
    logger.info(f"Large effect (|d| > 0.3): {len(results[abs(results['cohens_d']) > 0.3])}")

    return results


if __name__ == "__main__":
    results = main()
