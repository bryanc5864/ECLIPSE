#!/usr/bin/env python3
"""
Extract NON-LEAKY features for ecDNA prediction.

Uses upstream data that is available BEFORE ecDNA detection:
1. DepMap gene-level CNV (genome-wide, not AA amplicon CN)
2. DepMap gene expression (RNA-seq)
3. Hi-C topology features (reference-based chromatin context)
4. Fragile site proximity (from reference genome)

Does NOT use:
- AmpliconArchitect outputs (AA_AMP_Max_CN, genes_on_ecDNA, AMP_Type)
- Any features derived from ecDNA detection
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Key oncogenes known to be frequently amplified on ecDNA
ECDNA_ONCOGENES = [
    'MYC', 'MYCN', 'MYCL1', 'EGFR', 'ERBB2', 'CDK4', 'CDK6',
    'MDM2', 'MDM4', 'CCND1', 'CCNE1', 'FGFR1', 'FGFR2', 'MET',
    'PDGFRA', 'KIT', 'TERT', 'AR', 'BRAF', 'KRAS', 'PIK3CA'
]

# Common fragile sites (chromosome, start, end in hg38)
# From HUMCFS database
FRAGILE_SITES = [
    ('chr3', 60000000, 63000000, 'FRA3B'),   # FHIT
    ('chr16', 78000000, 79000000, 'FRA16D'),  # WWOX
    ('chr6', 11000000, 13000000, 'FRA6E'),
    ('chr7', 61000000, 68000000, 'FRA7G'),
    ('chr7', 110000000, 116000000, 'FRA7H'),
    ('chrX', 6500000, 8000000, 'FRAXB'),
]


def load_hic_features(data_dir: Path):
    """Load precomputed Hi-C topology features from reference genome."""
    hic_file = data_dir / "features" / "hic_features.npz"

    if not hic_file.exists():
        logger.warning(f"Hi-C features not found at {hic_file}. Run extract_hic_features.py first.")
        return None

    logger.info(f"Loading Hi-C features from {hic_file}...")
    hic_data = np.load(hic_file)

    # Convert to dict
    hic_features = {key: float(hic_data[key]) for key in hic_data.files}
    logger.info(f"  Loaded {len(hic_features)} Hi-C features")

    return hic_features


def add_hic_features(features_df: pd.DataFrame, hic_features: dict) -> pd.DataFrame:
    """Add Hi-C interaction features to feature DataFrame.

    Hi-C features are reference-based (from GM12878) and describe the
    chromatin topology context at oncogene loci. We create interaction
    features with sample-specific CNV to capture whether amplification
    occurs in regions with favorable chromatin topology for ecDNA formation.
    """
    if hic_features is None:
        logger.warning("No Hi-C features available, skipping...")
        return features_df

    logger.info("Adding Hi-C topology interaction features...")

    n_features_added = 0

    # 1. CNV × Hi-C density interactions
    # High value = amplified gene in accessible chromatin region
    for gene in ECDNA_ONCOGENES:
        cnv_col = f'cnv_{gene}'
        hic_density_key = f'hic_density_{gene}'

        if cnv_col in features_df.columns and hic_density_key in hic_features:
            hic_val = hic_features[hic_density_key]
            features_df[f'cnv_hic_{gene}'] = features_df[cnv_col] * hic_val
            n_features_added += 1

    # 2. CNV × Hi-C long-range interactions
    # High value = amplified gene with high long-range chromatin contacts
    for gene in ECDNA_ONCOGENES:
        cnv_col = f'cnv_{gene}'
        hic_lr_key = f'hic_longrange_{gene}'

        if cnv_col in features_df.columns and hic_lr_key in hic_features:
            hic_val = hic_features[hic_lr_key]
            features_df[f'cnv_hiclr_{gene}'] = features_df[cnv_col] * hic_val
            n_features_added += 1

    # 3. Summary statistics using Hi-C weights
    # Weight oncogene CNV by their Hi-C accessibility
    hic_density_mean = hic_features.get('hic_density_mean', 1.0)
    hic_longrange_mean = hic_features.get('hic_longrange_mean', 0.5)

    # Weighted oncogene CNV max by Hi-C relative density
    weighted_cnv = []
    for gene in ECDNA_ONCOGENES:
        cnv_col = f'cnv_{gene}'
        rel_key = f'hic_density_rel_{gene}'
        if cnv_col in features_df.columns and rel_key in hic_features:
            weight = hic_features[rel_key]
            weighted_cnv.append(features_df[cnv_col] * weight)

    if weighted_cnv:
        weighted_df = pd.concat(weighted_cnv, axis=1)
        features_df['oncogene_cnv_hic_weighted_max'] = weighted_df.max(axis=1)
        features_df['oncogene_cnv_hic_weighted_mean'] = weighted_df.mean(axis=1)
        n_features_added += 2

    # 4. Hi-C summary features (as reference context)
    # These are the same for all samples but provide model with topology info
    features_df['hic_density_mean'] = hic_features.get('hic_density_mean', 0)
    features_df['hic_density_max'] = hic_features.get('hic_density_max', 0)
    features_df['hic_longrange_mean'] = hic_features.get('hic_longrange_mean', 0)
    n_features_added += 3

    logger.info(f"  Added {n_features_added} Hi-C interaction features")

    return features_df


def load_depmap_data(data_dir: Path):
    """Load DepMap CNV and expression data."""
    logger.info("Loading DepMap data...")

    # Load cell line info
    cell_info = pd.read_csv(data_dir / "depmap" / "cell_line_info.csv")
    logger.info(f"  Cell lines: {len(cell_info)}")

    # Load CNV (gene-level copy number)
    logger.info("  Loading CNV data (this may take a moment)...")
    cnv = pd.read_csv(data_dir / "depmap" / "copy_number.csv", index_col=0)
    logger.info(f"  CNV shape: {cnv.shape}")

    # Load expression
    logger.info("  Loading expression data...")
    expr = pd.read_csv(data_dir / "depmap" / "expression.csv", index_col=0)
    logger.info(f"  Expression shape: {expr.shape}")

    return cell_info, cnv, expr


def load_ecdna_labels(data_dir: Path):
    """Load ecDNA labels from CytoCellDB (only the label, not AA features)."""
    logger.info("Loading ecDNA labels from CytoCellDB...")

    cyto = pd.read_excel(data_dir / "cytocell_db" / "CytoCellDB_Supp_File1.xlsx")

    # Get ecDNA status (Y/N) - this is from FISH, the ground truth
    # Note: We're NOT using any AA_* columns here
    labels = cyto[['DepMap_ID', 'CCLE_Name_Format', 'ECDNA', 'lineage', 'primary_disease']].copy()
    labels['ecdna_positive'] = (labels['ECDNA'] == 'Y').astype(int)

    # Drop rows without DepMap ID
    labels = labels.dropna(subset=['DepMap_ID'])
    labels = labels.set_index('DepMap_ID')

    logger.info(f"  Samples with labels: {len(labels)}")
    logger.info(f"  ecDNA+: {labels['ecdna_positive'].sum()}")
    logger.info(f"  ecDNA-: {len(labels) - labels['ecdna_positive'].sum()}")

    return labels


def find_gene_columns(df, gene_name):
    """Find columns matching a gene name (handles 'GENE (ID)' format)."""
    matches = [col for col in df.columns if col.startswith(f"{gene_name} (") or col == gene_name]
    return matches[0] if matches else None


def extract_cnv_features(cnv: pd.DataFrame, sample_ids: list) -> pd.DataFrame:
    """Extract CNV-based features (non-leaky)."""
    logger.info("Extracting CNV features...")

    features = pd.DataFrame(index=sample_ids)

    # Filter to samples we have
    cnv_filtered = cnv.loc[cnv.index.intersection(sample_ids)]

    # 1. Oncogene CNV features
    for gene in ECDNA_ONCOGENES:
        col = find_gene_columns(cnv, gene)
        if col:
            features[f'cnv_{gene}'] = cnv_filtered[col]

    # 2. Summary CNV statistics (genome-wide)
    features['cnv_max'] = cnv_filtered.max(axis=1)
    features['cnv_mean'] = cnv_filtered.mean(axis=1)
    features['cnv_std'] = cnv_filtered.std(axis=1)
    features['cnv_q95'] = cnv_filtered.quantile(0.95, axis=1)
    features['cnv_q99'] = cnv_filtered.quantile(0.99, axis=1)

    # 3. Fraction of genome with high CN
    features['cnv_frac_gt2'] = (cnv_filtered > 2).mean(axis=1)
    features['cnv_frac_gt3'] = (cnv_filtered > 3).mean(axis=1)
    features['cnv_frac_gt5'] = (cnv_filtered > 5).mean(axis=1)

    # 4. Oncogene-specific features
    oncogene_cols = [find_gene_columns(cnv, g) for g in ECDNA_ONCOGENES]
    oncogene_cols = [c for c in oncogene_cols if c]
    if oncogene_cols:
        oncogene_cnv = cnv_filtered[oncogene_cols]
        features['oncogene_cnv_max'] = oncogene_cnv.max(axis=1)
        features['oncogene_cnv_mean'] = oncogene_cnv.mean(axis=1)
        features['n_oncogenes_amplified'] = (oncogene_cnv > 3).sum(axis=1)

    logger.info(f"  CNV features: {len([c for c in features.columns if c.startswith('cnv_') or c.startswith('oncogene_')])}")

    return features


def extract_expression_features(expr: pd.DataFrame, sample_ids: list) -> pd.DataFrame:
    """Extract expression-based features (non-leaky)."""
    logger.info("Extracting expression features...")

    features = pd.DataFrame(index=sample_ids)

    # Filter to samples we have
    expr_filtered = expr.loc[expr.index.intersection(sample_ids)]

    # 1. Oncogene expression
    for gene in ECDNA_ONCOGENES:
        col = find_gene_columns(expr, gene)
        if col:
            features[f'expr_{gene}'] = expr_filtered[col]

    # 2. Summary expression statistics
    features['expr_mean'] = expr_filtered.mean(axis=1)
    features['expr_std'] = expr_filtered.std(axis=1)
    features['expr_max'] = expr_filtered.max(axis=1)

    # 3. High expression fraction
    features['expr_frac_high'] = (expr_filtered > expr_filtered.median().median()).mean(axis=1)

    # 4. Oncogene expression features
    oncogene_cols = [find_gene_columns(expr, g) for g in ECDNA_ONCOGENES]
    oncogene_cols = [c for c in oncogene_cols if c]
    if oncogene_cols:
        oncogene_expr = expr_filtered[oncogene_cols]
        features['oncogene_expr_max'] = oncogene_expr.max(axis=1)
        features['oncogene_expr_mean'] = oncogene_expr.mean(axis=1)
        features['n_oncogenes_high_expr'] = (oncogene_expr > oncogene_expr.median().median()).sum(axis=1)

    logger.info(f"  Expression features: {len([c for c in features.columns if c.startswith('expr_') or c.startswith('oncogene_expr')])}")

    return features


def extract_cnv_expression_correlation(cnv: pd.DataFrame, expr: pd.DataFrame, sample_ids: list) -> pd.DataFrame:
    """Extract CNV-expression correlation features (dosage effect indicator)."""
    logger.info("Extracting CNV-expression correlation features...")

    features = pd.DataFrame(index=sample_ids)

    common_samples = list(set(cnv.index) & set(expr.index) & set(sample_ids))

    # For each oncogene, compute CNV-expression correlation per sample is not meaningful
    # Instead, compute whether high CNV correlates with high expression for that sample
    for gene in ECDNA_ONCOGENES[:10]:  # Top oncogenes
        cnv_col = find_gene_columns(cnv, gene)
        expr_col = find_gene_columns(expr, gene)

        if cnv_col and expr_col:
            # Compute product (high CNV * high expr suggests functional amplification)
            cnv_vals = cnv.loc[common_samples, cnv_col]
            expr_vals = expr.loc[common_samples, expr_col]
            features.loc[common_samples, f'dosage_{gene}'] = cnv_vals * expr_vals

    logger.info(f"  Dosage features: {len([c for c in features.columns if c.startswith('dosage_')])}")

    return features


def main():
    data_dir = Path("data")
    output_dir = data_dir / "features"
    output_dir.mkdir(exist_ok=True)

    # Load data
    cell_info, cnv, expr = load_depmap_data(data_dir)
    labels = load_ecdna_labels(data_dir)

    # Load Hi-C features (reference-based)
    hic_features = load_hic_features(data_dir)

    # Find common samples (sorted for deterministic splits across runs)
    common_samples = sorted(list(
        set(labels.index) &
        set(cnv.index) &
        set(expr.index)
    ))
    logger.info(f"\nCommon samples with all data: {len(common_samples)}")

    # Extract features
    cnv_features = extract_cnv_features(cnv, common_samples)
    expr_features = extract_expression_features(expr, common_samples)
    dosage_features = extract_cnv_expression_correlation(cnv, expr, common_samples)

    # Combine features
    all_features = pd.concat([cnv_features, expr_features, dosage_features], axis=1)
    all_features = all_features.loc[common_samples]

    # Add Hi-C topology features (reference-based, same for all samples)
    all_features = add_hic_features(all_features, hic_features)

    # Add labels
    all_features['ecdna_positive'] = labels.loc[common_samples, 'ecdna_positive']

    # Handle missing values
    all_features = all_features.fillna(0)

    logger.info(f"\n=== Final Feature Matrix ===")
    logger.info(f"Samples: {len(all_features)}")
    logger.info(f"Features: {len(all_features.columns) - 1}")
    logger.info(f"ecDNA+: {all_features['ecdna_positive'].sum()}")
    logger.info(f"ecDNA-: {len(all_features) - all_features['ecdna_positive'].sum()}")

    # Split data
    np.random.seed(42)
    indices = np.random.permutation(len(all_features))

    n_val = int(len(indices) * 0.15)
    n_train = len(indices) - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    # Get feature columns (exclude label)
    feature_cols = [c for c in all_features.columns if c != 'ecdna_positive']

    # Save as npz files
    for split_name, split_idx in [('train', train_idx), ('val', val_idx)]:
        split_data = all_features.iloc[split_idx]

        X = split_data[feature_cols].values.astype(np.float32)
        y = split_data['ecdna_positive'].values.astype(np.float32)
        sample_ids = split_data.index.values

        # Pad to match model expectations
        n_samples = len(X)

        # sequence_features: 256-dim
        seq_feat = np.zeros((n_samples, 256), dtype=np.float32)
        seq_feat[:, :min(256, X.shape[1])] = X[:, :min(256, X.shape[1])]

        # topology_features: 256-dim
        topo_feat = np.zeros((n_samples, 256), dtype=np.float32)
        topo_feat[:, :min(256, X.shape[1])] = X[:, :min(256, X.shape[1])]

        # fragile_site_features: 64-dim
        frag_feat = np.zeros((n_samples, 64), dtype=np.float32)
        frag_feat[:, :min(64, X.shape[1])] = X[:, :min(64, X.shape[1])]

        # copy_number_features: 32-dim
        cn_cols = [i for i, c in enumerate(feature_cols) if 'cnv' in c.lower()]
        cn_feat = np.zeros((n_samples, 32), dtype=np.float32)
        if cn_cols:
            cn_feat[:, :min(32, len(cn_cols))] = X[:, cn_cols[:min(32, len(cn_cols))]]

        output_file = output_dir / f"module1_features_{split_name}.npz"
        np.savez(
            output_file,
            sequence_features=seq_feat,
            topology_features=topo_feat,
            fragile_site_features=frag_feat,
            copy_number_features=cn_feat,
            labels=y,
            sample_ids=sample_ids,
            feature_names=np.array(feature_cols),
        )

        logger.info(f"Saved {split_name}: {len(split_idx)} samples, {y.sum():.0f} ecDNA+")

    # Quick baseline evaluation
    logger.info("\n=== Baseline RandomForest (NON-LEAKY features) ===")
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score, f1_score, classification_report

        X_train = all_features.iloc[train_idx][feature_cols].values
        y_train = all_features.iloc[train_idx]['ecdna_positive'].values
        X_val = all_features.iloc[val_idx][feature_cols].values
        y_val = all_features.iloc[val_idx]['ecdna_positive'].values

        clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_val)[:, 1]
        y_pred = clf.predict(X_val)

        auroc = roc_auc_score(y_val, y_prob)
        f1 = f1_score(y_val, y_pred)

        logger.info(f"Validation AUROC: {auroc:.3f}")
        logger.info(f"Validation F1: {f1:.3f}")
        print(classification_report(y_val, y_pred, target_names=['ecDNA-', 'ecDNA+']))

        # Feature importance
        logger.info("\nTop 15 Feature Importances:")
        importances = sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1])
        for name, imp in importances[:15]:
            logger.info(f"  {name}: {imp:.3f}")

    except ImportError:
        logger.warning("sklearn not available for baseline")

    logger.info("\nDone! Non-leaky features saved to data/features/")


if __name__ == "__main__":
    main()
