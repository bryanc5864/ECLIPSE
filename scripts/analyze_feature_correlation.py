#!/usr/bin/env python3
"""
Feature correlation analysis for ecDNA-Former.

Analyzes:
1. Feature-label correlations (point-biserial)
2. Inter-feature correlations (identify redundancy)
3. Top discriminative features by effect size

Usage:
    python scripts/analyze_feature_correlation.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    data_dir = Path("data")

    # Load features
    train_data = np.load(data_dir / "features" / "module1_features_train.npz", allow_pickle=True)
    val_data = np.load(data_dir / "features" / "module1_features_val.npz", allow_pickle=True)

    feature_names = list(train_data["feature_names"])

    # Combine train + val for full analysis
    all_labels = np.concatenate([train_data["labels"], val_data["labels"]])

    # Reconstruct raw feature matrix from the packed arrays
    # sequence_features has raw features in first N positions
    all_seq = np.concatenate([train_data["sequence_features"], val_data["sequence_features"]])
    n_features = len(feature_names)
    X = all_seq[:, :n_features]

    logger.info(f"Samples: {len(all_labels)}, Features: {n_features}, ecDNA+: {int(all_labels.sum())}")

    # 1. Feature-label correlations (point-biserial)
    logger.info(f"\n{'='*60}")
    logger.info("FEATURE-LABEL CORRELATIONS (point-biserial)")
    logger.info(f"{'='*60}")

    correlations = []
    for i, name in enumerate(feature_names):
        feat = X[:, i]
        if feat.std() == 0:
            correlations.append({"feature": name, "correlation": 0, "p_value": 1.0, "abs_corr": 0})
            continue
        r, p = stats.pointbiserialr(all_labels, feat)
        correlations.append({"feature": name, "correlation": r, "p_value": p, "abs_corr": abs(r)})

    corr_df = pd.DataFrame(correlations).sort_values("abs_corr", ascending=False)

    logger.info(f"\nTop 20 features by |correlation| with ecDNA label:")
    logger.info(f"{'Feature':>40s} {'Corr':>8s} {'p-value':>12s}")
    logger.info("-" * 65)
    for _, row in corr_df.head(20).iterrows():
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        logger.info(f"{row['feature']:>40s} {row['correlation']:>+8.4f} {row['p_value']:>12.2e} {sig}")

    # 2. Effect sizes (Cohen's d between ecDNA+ and ecDNA-)
    logger.info(f"\n{'='*60}")
    logger.info("EFFECT SIZES (Cohen's d)")
    logger.info(f"{'='*60}")

    pos_mask = all_labels == 1
    neg_mask = all_labels == 0
    effect_sizes = []

    for i, name in enumerate(feature_names):
        pos_vals = X[pos_mask, i]
        neg_vals = X[neg_mask, i]
        pooled_std = np.sqrt((pos_vals.std()**2 + neg_vals.std()**2) / 2)
        if pooled_std > 0:
            d = (pos_vals.mean() - neg_vals.mean()) / pooled_std
        else:
            d = 0
        effect_sizes.append({
            "feature": name,
            "mean_pos": pos_vals.mean(),
            "mean_neg": neg_vals.mean(),
            "cohens_d": d,
            "abs_d": abs(d),
        })

    effect_df = pd.DataFrame(effect_sizes).sort_values("abs_d", ascending=False)

    logger.info(f"\nTop 20 features by |Cohen's d|:")
    logger.info(f"{'Feature':>40s} {'d':>8s} {'Mean+':>10s} {'Mean-':>10s}")
    logger.info("-" * 75)
    for _, row in effect_df.head(20).iterrows():
        logger.info(f"{row['feature']:>40s} {row['cohens_d']:>+8.3f} {row['mean_pos']:>10.3f} {row['mean_neg']:>10.3f}")

    # 3. Inter-feature correlations (find highly correlated pairs)
    logger.info(f"\n{'='*60}")
    logger.info("HIGHLY CORRELATED FEATURE PAIRS (|r| > 0.8)")
    logger.info(f"{'='*60}")

    corr_matrix = np.corrcoef(X.T)
    high_corr_pairs = []
    for i in range(n_features):
        for j in range(i+1, n_features):
            r = corr_matrix[i, j]
            if abs(r) > 0.8 and not np.isnan(r):
                high_corr_pairs.append({
                    "feature_1": feature_names[i],
                    "feature_2": feature_names[j],
                    "correlation": r,
                })

    if high_corr_pairs:
        pairs_df = pd.DataFrame(high_corr_pairs).sort_values("correlation", ascending=False)
        for _, row in pairs_df.iterrows():
            logger.info(f"  {row['feature_1']:>35s} ↔ {row['feature_2']:<35s} r={row['correlation']:+.3f}")
        logger.info(f"\n  Total highly correlated pairs: {len(high_corr_pairs)}")
    else:
        logger.info("  No pairs with |r| > 0.8")

    # Save results
    output_dir = data_dir / "validation"
    output_dir.mkdir(exist_ok=True)

    corr_df.to_csv(output_dir / "feature_label_correlations.csv", index=False)
    effect_df.to_csv(output_dir / "feature_effect_sizes.csv", index=False)
    if high_corr_pairs:
        pd.DataFrame(high_corr_pairs).to_csv(output_dir / "feature_intercorrelations.csv", index=False)

    logger.info(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
