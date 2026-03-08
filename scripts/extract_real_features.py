#!/usr/bin/env python3
"""
Extract REAL biologically meaningful features for ecDNA prediction.
Uses CytoCellDB which has:
- ecDNA status (ground truth)
- Copy number (THE key feature)
- Oncogenes on ecDNA
- DepMap IDs (links to CRISPR/expression)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ast
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_dict_string(s):
    """Parse string representation of dict."""
    if pd.isna(s) or s == 'NaN':
        return {}
    try:
        return ast.literal_eval(s)
    except:
        return {}


def parse_list_string(s):
    """Parse string representation of list."""
    if pd.isna(s) or s == 'NaN':
        return []
    try:
        result = ast.literal_eval(s)
        return result if isinstance(result, list) else []
    except:
        return []


def extract_features(data_dir: Path):
    """Extract real features from CytoCellDB."""

    # Load CytoCellDB
    logger.info("Loading CytoCellDB...")
    cyto = pd.read_excel(data_dir / "cytocell_db" / "CytoCellDB_Supp_File1.xlsx")
    logger.info(f"Loaded {len(cyto)} cell lines")

    # Key oncogenes associated with ecDNA
    KEY_ONCOGENES = ['MYC', 'MYCL', 'MYCN', 'EGFR', 'ERBB2', 'CDK4', 'CDK6',
                     'MDM2', 'CCND1', 'CCNE1', 'FGFR1', 'MET', 'PDGFRA']

    features = []

    for idx, row in cyto.iterrows():
        feat = {}

        # Sample ID
        feat['sample_id'] = row['DepMap_ID'] if pd.notna(row['DepMap_ID']) else row['CCLE_Name_Format']
        feat['depmap_id'] = row['DepMap_ID'] if pd.notna(row['DepMap_ID']) else None

        # === TARGET: ecDNA status ===
        feat['ecdna_positive'] = 1 if row['ECDNA'] == 'Y' else 0

        # === COPY NUMBER FEATURES (CRITICAL) ===
        # Parse max copy number from AmpliconArchitect
        cn_dict = parse_dict_string(row.get('AA_AMP_Max_CN', '{}'))
        if cn_dict:
            cn_values = list(cn_dict.values())
            feat['max_copy_number'] = max(cn_values)
            feat['mean_copy_number'] = np.mean(cn_values)
            feat['num_amplicons'] = len(cn_values)
        else:
            feat['max_copy_number'] = 2.0  # Diploid default
            feat['mean_copy_number'] = 2.0
            feat['num_amplicons'] = 0

        # Copy number thresholds (biologically meaningful)
        feat['cn_gt_10'] = 1 if feat['max_copy_number'] > 10 else 0
        feat['cn_gt_20'] = 1 if feat['max_copy_number'] > 20 else 0
        feat['cn_gt_50'] = 1 if feat['max_copy_number'] > 50 else 0
        feat['log_max_cn'] = np.log2(feat['max_copy_number'] + 1)

        # === ONCOGENE FEATURES (HIGH IMPORTANCE) ===
        genes = parse_list_string(row.get('AA - genes_on_ecDNA', '[]'))
        genes_str = ' '.join(genes).upper() if genes else ''

        # Check for key oncogenes
        for oncogene in KEY_ONCOGENES:
            feat[f'has_{oncogene}'] = 1 if oncogene in genes_str else 0

        feat['num_oncogenes'] = sum(1 for og in KEY_ONCOGENES if og in genes_str)
        feat['total_genes_on_ecdna'] = len(genes)

        # === STRUCTURAL FEATURES ===
        # Amplification type
        amp_types = parse_list_string(row.get('AA -AMP Type', '[]'))
        feat['has_ecdna_type'] = 1 if 'ecDNA' in str(amp_types) else 0
        feat['has_bfb_type'] = 1 if 'BFB' in str(amp_types) else 0
        feat['has_linear_type'] = 1 if 'Linear' in str(amp_types) else 0
        feat['has_complex_type'] = 1 if 'Complex' in str(amp_types) else 0

        # HSR status
        feat['has_hsr'] = 1 if row.get('HSR') == 'Y' else 0

        # === METADATA FEATURES ===
        feat['lineage'] = row.get('lineage', 'unknown')
        feat['primary_disease'] = row.get('primary_disease', 'unknown')

        features.append(feat)

    df = pd.DataFrame(features)
    logger.info(f"Extracted features for {len(df)} samples")
    logger.info(f"  ecDNA+: {df['ecdna_positive'].sum()}")
    logger.info(f"  ecDNA-: {len(df) - df['ecdna_positive'].sum()}")

    return df


def create_feature_matrix(df):
    """Create numeric feature matrix for training."""

    # Numeric features to use
    numeric_cols = [
        # Copy number (MOST IMPORTANT)
        'max_copy_number', 'mean_copy_number', 'log_max_cn',
        'cn_gt_10', 'cn_gt_20', 'cn_gt_50', 'num_amplicons',

        # Oncogenes
        'has_MYC', 'has_MYCL', 'has_MYCN', 'has_EGFR', 'has_ERBB2',
        'has_CDK4', 'has_CDK6', 'has_MDM2', 'has_CCND1', 'has_CCNE1',
        'has_FGFR1', 'has_MET', 'has_PDGFRA',
        'num_oncogenes', 'total_genes_on_ecdna',

        # Structural
        'has_ecdna_type', 'has_bfb_type', 'has_linear_type', 'has_complex_type',
        'has_hsr',
    ]

    X = df[numeric_cols].fillna(0).values.astype(np.float32)
    y = df['ecdna_positive'].values.astype(np.float32)
    sample_ids = df['sample_id'].values

    return X, y, sample_ids, numeric_cols


def main():
    data_dir = Path("data")
    output_dir = data_dir / "features"
    output_dir.mkdir(exist_ok=True)

    # Extract features
    df = extract_features(data_dir)

    # Create feature matrix
    X, y, sample_ids, feature_names = create_feature_matrix(df)

    logger.info(f"\n=== Feature Matrix ===")
    logger.info(f"Shape: {X.shape}")
    logger.info(f"Features: {feature_names}")

    # === Quick baseline check ===
    logger.info("\n=== Feature Importance (correlation with ecDNA) ===")
    correlations = []
    for i, name in enumerate(feature_names):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        correlations.append((name, corr))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, corr in correlations[:10]:
        logger.info(f"  {name}: {corr:.3f}")

    # Split data
    np.random.seed(42)
    n = len(y)
    indices = np.random.permutation(n)

    n_val = int(n * 0.15)
    n_train = n - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    # Save features
    for split_name, split_idx in [('train', train_idx), ('val', val_idx)]:
        # Create feature dict matching model expectations
        n_samples = len(split_idx)

        # Pad/reshape to match model dimensions
        # sequence_features: 256-dim (use copy number + oncogenes)
        # topology_features: 256-dim (use structural features)
        # fragile_site_features: 64-dim (zeros for now)
        # copy_number_features: 32-dim (copy number features)

        seq_feat = np.zeros((n_samples, 256), dtype=np.float32)
        seq_feat[:, :X.shape[1]] = X[split_idx]

        topo_feat = np.zeros((n_samples, 256), dtype=np.float32)
        # Use same features, let model learn different representations
        topo_feat[:, :X.shape[1]] = X[split_idx]

        frag_feat = np.zeros((n_samples, 64), dtype=np.float32)
        # Encode some features here too
        frag_feat[:, :X.shape[1]] = X[split_idx, :min(64, X.shape[1])]

        cn_feat = np.zeros((n_samples, 32), dtype=np.float32)
        # Copy number features specifically
        cn_cols = [i for i, name in enumerate(feature_names) if 'copy' in name.lower() or 'cn' in name.lower()]
        cn_feat[:, :len(cn_cols)] = X[split_idx][:, cn_cols]

        output_file = output_dir / f"module1_features_{split_name}.npz"
        np.savez(
            output_file,
            sequence_features=seq_feat,
            topology_features=topo_feat,
            fragile_site_features=frag_feat,
            copy_number_features=cn_feat,
            labels=y[split_idx],
            sample_ids=sample_ids[split_idx],
            feature_names=np.array(feature_names),
        )

        logger.info(f"\nSaved {split_name}: {len(split_idx)} samples, {y[split_idx].sum():.0f} ecDNA+")

    # === Baseline model ===
    logger.info("\n=== Baseline RandomForest ===")
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score, f1_score, classification_report

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]

        auroc = roc_auc_score(y_val, y_prob)
        f1 = f1_score(y_val, y_pred)

        logger.info(f"Validation AUROC: {auroc:.3f}")
        logger.info(f"Validation F1: {f1:.3f}")
        logger.info("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=['ecDNA-', 'ecDNA+']))

        # Feature importance
        logger.info("\nTop 10 Feature Importances:")
        importances = list(zip(feature_names, clf.feature_importances_))
        importances.sort(key=lambda x: x[1], reverse=True)
        for name, imp in importances[:10]:
            logger.info(f"  {name}: {imp:.3f}")

    except ImportError:
        logger.warning("sklearn not available for baseline")

    logger.info("\nDone! Features saved to data/features/")


if __name__ == "__main__":
    main()
