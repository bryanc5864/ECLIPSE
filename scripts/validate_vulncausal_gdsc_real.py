#!/usr/bin/env python3
"""
Validate VulnCausal hits against REAL GDSC2 drug sensitivity data.

Cross-references CytoCellDB ecDNA annotations with GDSC2 IC50 values
to test whether ecDNA+ cell lines are more sensitive to drugs targeting
our identified vulnerability genes.

Data sources:
- GDSC2: https://www.cancerrxgene.org/ (release 8.5)
- CytoCellDB: Fessler et al. NAR Cancer 2024
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load and merge GDSC2 and CytoCellDB data."""

    print("Loading GDSC2 drug sensitivity data...")
    gdsc = pd.read_excel('data/gdsc/GDSC2_fitted_dose_response.xlsx')
    print(f"  {len(gdsc)} dose-response measurements")
    print(f"  {gdsc['CELL_LINE_NAME'].nunique()} cell lines, {gdsc['DRUG_NAME'].nunique()} drugs")

    print("\nLoading CytoCellDB ecDNA annotations...")
    cyto = pd.read_excel('data/cytocell_db/CytoCellDB_Supp_File1.xlsx')
    cyto['is_ecdna'] = cyto['ECDNA'] == 'Y'
    print(f"  {len(cyto)} cell lines ({cyto['is_ecdna'].sum()} ecDNA+, {(~cyto['is_ecdna']).sum()} ecDNA-)")

    # Merge via COSMIC ID
    cyto_map = cyto[['COSMICID', 'is_ecdna', 'stripped_cell_line_name']].dropna(subset=['COSMICID'])
    cyto_map['COSMICID'] = cyto_map['COSMICID'].astype(int)
    cyto_map = cyto_map.drop_duplicates(subset='COSMICID')

    merged = gdsc.merge(cyto_map, left_on='COSMIC_ID', right_on='COSMICID', how='inner')
    print(f"\nMerged dataset: {len(merged)} measurements")
    print(f"  Cell lines with ecDNA status: {merged['COSMIC_ID'].nunique()}")
    print(f"  ecDNA+ lines: {merged[merged['is_ecdna']]['COSMIC_ID'].nunique()}")
    print(f"  ecDNA- lines: {merged[~merged['is_ecdna']]['COSMIC_ID'].nunique()}")

    return merged


# Drugs in GDSC2 that target our vulnerability pathways
DRUG_QUERIES = [
    # CHK1 pathway (our top validated hit)
    {'gene': 'CHK1', 'drug_names': ['AZD7762', 'MK-8776'], 'putative_targets': ['CHEK1', 'CHEK2', 'CHK1']},
    # CDK inhibitors (CDK1 hit)
    {'gene': 'CDK1', 'drug_names': ['Dinaciclib', 'AT-7519', 'AZD5438', 'RO-3306', 'Palbociclib', 'Ribociclib', 'Abemaciclib'], 'putative_targets': ['CDK1', 'CDK2', 'CDK4', 'CDK6', 'CDK9']},
    # Proteasome inhibitors (PSMD7 hit)
    {'gene': 'PSMD7', 'drug_names': ['Bortezomib', 'MG-132'], 'putative_targets': ['PSMB5', 'proteasome']},
    # BCL-XL/apoptosis (BCL2L1 hit)
    {'gene': 'BCL2L1', 'drug_names': ['Navitoclax', 'ABT-737', 'Venetoclax'], 'putative_targets': ['BCL2', 'BCL-XL', 'BCL2L1']},
    # DNA replication (ORC6/MCM2 hits)
    {'gene': 'ORC6/MCM2', 'drug_names': ['Gemcitabine', 'Cytarabine', '5-Fluorouracil'], 'putative_targets': ['DNA replication', 'antimetabolite']},
    # Mitosis/kinesin (KIF11 hit)
    {'gene': 'KIF11', 'drug_names': ['Ispinesib', 'GSK461364', 'BI-2536', 'Volasertib'], 'putative_targets': ['KIF11', 'PLK1', 'KSP']},
    # Spliceosome (SNRPF hit)
    {'gene': 'SNRPF', 'drug_names': ['Pladienolide B', 'E7107'], 'putative_targets': ['SF3B1', 'spliceosome']},
]


def find_drugs_in_gdsc(merged, query):
    """Find matching drugs in GDSC2 for a given query."""
    # Try matching by drug name first
    drug_mask = merged['DRUG_NAME'].str.upper().isin([d.upper() for d in query['drug_names']])

    # Also try matching by putative target
    for target in query['putative_targets']:
        target_mask = merged['PUTATIVE_TARGET'].str.contains(target, case=False, na=False)
        drug_mask = drug_mask | target_mask

    return merged[drug_mask]


def analyze_drug(drug_data, drug_name, gene_target):
    """Analyze ecDNA+ vs ecDNA- sensitivity for a specific drug."""
    ecdna_pos = drug_data[drug_data['is_ecdna']]['LN_IC50']
    ecdna_neg = drug_data[~drug_data['is_ecdna']]['LN_IC50']

    if len(ecdna_pos) < 3 or len(ecdna_neg) < 3:
        return None

    # Mann-Whitney U test (one-sided: ecDNA+ more sensitive = lower IC50)
    stat, pval = stats.mannwhitneyu(ecdna_pos, ecdna_neg, alternative='less')

    # Effect size
    mean_diff = ecdna_neg.mean() - ecdna_pos.mean()
    selectivity_ratio = np.exp(mean_diff)

    # Cohen's d
    pooled_std = np.sqrt((ecdna_pos.std()**2 + ecdna_neg.std()**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

    # Also do two-sided t-test
    t_stat, t_pval = stats.ttest_ind(ecdna_pos, ecdna_neg)

    return {
        'gene_target': gene_target,
        'drug': drug_name,
        'n_ecdna_pos': len(ecdna_pos),
        'n_ecdna_neg': len(ecdna_neg),
        'mean_ln_ic50_pos': ecdna_pos.mean(),
        'mean_ln_ic50_neg': ecdna_neg.mean(),
        'mean_ic50_pos_uM': np.exp(ecdna_pos.mean()),
        'mean_ic50_neg_uM': np.exp(ecdna_neg.mean()),
        'selectivity_ratio': selectivity_ratio,
        'cohens_d': cohens_d,
        'mw_pvalue': pval,
        'ttest_pvalue': t_pval,
    }


def main():
    print("=" * 80)
    print("MODULE 3 VALIDATION: VulnCausal vs REAL GDSC2 Drug Sensitivity")
    print("=" * 80)
    print()

    merged = load_data()

    # List all available drugs
    all_drugs = sorted(merged['DRUG_NAME'].unique())
    print(f"\nAvailable drugs in merged dataset: {len(all_drugs)}")

    results = []

    for query in DRUG_QUERIES:
        gene = query['gene']
        print(f"\n{'='*70}")
        print(f"Target gene: {gene}")
        print(f"Searching for drugs: {', '.join(query['drug_names'])}")
        print(f"Target keywords: {', '.join(query['putative_targets'])}")
        print(f"{'='*70}")

        # Find matching drugs
        drug_subset = find_drugs_in_gdsc(merged, query)

        if len(drug_subset) == 0:
            print("  No matching drugs found in GDSC2")
            continue

        matching_drugs = drug_subset['DRUG_NAME'].unique()
        print(f"  Found {len(matching_drugs)} matching drugs: {', '.join(matching_drugs)}")

        for drug_name in matching_drugs:
            drug_data = drug_subset[drug_subset['DRUG_NAME'] == drug_name]

            result = analyze_drug(drug_data, drug_name, gene)
            if result is None:
                print(f"\n  {drug_name}: insufficient ecDNA+ data (skipping)")
                continue

            results.append(result)

            sig = '***' if result['mw_pvalue'] < 0.001 else ('**' if result['mw_pvalue'] < 0.01 else ('*' if result['mw_pvalue'] < 0.05 else ''))

            print(f"\n  {drug_name}:")
            print(f"    ecDNA+ (n={result['n_ecdna_pos']}): mean IC50 = {result['mean_ic50_pos_uM']:.3f} uM")
            print(f"    ecDNA- (n={result['n_ecdna_neg']}): mean IC50 = {result['mean_ic50_neg_uM']:.3f} uM")
            print(f"    Selectivity: {result['selectivity_ratio']:.2f}x")
            print(f"    Cohen's d: {result['cohens_d']:.3f}")
            print(f"    P-value (MW): {result['mw_pvalue']:.4f} {sig}")

    # Summary
    if results:
        df_results = pd.DataFrame(results)

        print("\n" + "=" * 80)
        print("SUMMARY: ALL DRUGS TESTED")
        print("=" * 80)

        print(f"\n{'Gene':<10} {'Drug':<20} {'n+':<4} {'n-':<5} {'Selectivity':>12} {'Cohen d':>8} {'P-value':>10} {'Sig':>5}")
        print("-" * 80)
        for _, row in df_results.sort_values('mw_pvalue').iterrows():
            sig = '***' if row['mw_pvalue'] < 0.001 else ('**' if row['mw_pvalue'] < 0.01 else ('*' if row['mw_pvalue'] < 0.05 else ''))
            print(f"{row['gene_target']:<10} {row['drug']:<20} {row['n_ecdna_pos']:<4} {row['n_ecdna_neg']:<5} {row['selectivity_ratio']:>10.2f}x {row['cohens_d']:>8.3f} {row['mw_pvalue']:>10.4f} {sig:>5}")

        n_sig = sum(1 for r in results if r['mw_pvalue'] < 0.05)
        n_total = len(results)
        print(f"\nSignificant (p<0.05): {n_sig}/{n_total}")
        print(f"Mean selectivity ratio: {df_results['selectivity_ratio'].mean():.2f}x")

        # By gene target
        print("\n" + "=" * 80)
        print("SUMMARY BY GENE TARGET")
        print("=" * 80)
        for gene in df_results['gene_target'].unique():
            gene_df = df_results[df_results['gene_target'] == gene]
            best = gene_df.loc[gene_df['mw_pvalue'].idxmin()]
            print(f"\n{gene}:")
            print(f"  Best drug: {best['drug']}")
            print(f"  Selectivity: {best['selectivity_ratio']:.2f}x (p={best['mw_pvalue']:.4f})")

        # Bonferroni correction
        print("\n" + "=" * 80)
        print("MULTIPLE TESTING CORRECTION")
        print("=" * 80)
        bonferroni_threshold = 0.05 / n_total
        n_bonferroni = sum(1 for r in results if r['mw_pvalue'] < bonferroni_threshold)
        print(f"Bonferroni threshold: {bonferroni_threshold:.4f}")
        print(f"Significant after correction: {n_bonferroni}/{n_total}")

        # FDR (Benjamini-Hochberg)
        pvals = sorted([r['mw_pvalue'] for r in results])
        n = len(pvals)
        fdr_sig = 0
        for i, p in enumerate(pvals):
            if p <= (i + 1) / n * 0.05:
                fdr_sig = i + 1
        print(f"FDR significant (BH, q<0.05): {fdr_sig}/{n_total}")

        # Save
        output_dir = Path("data/validation")
        output_dir.mkdir(exist_ok=True, parents=True)
        df_results.to_csv(output_dir / "vulncausal_gdsc_real_validation.csv", index=False)
        print(f"\nResults saved to {output_dir / 'vulncausal_gdsc_real_validation.csv'}")

    else:
        print("\nNo drug results found!")

    return results


if __name__ == "__main__":
    main()
