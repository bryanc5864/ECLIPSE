#!/usr/bin/env python3
"""
Bootstrap significance tests for ecDNA-Former (Module 1).

1. Bootstrap CI on validation AUROC
2. Bootstrap p-value comparing ecDNA-Former vs Random Forest baseline
3. Permutation test for AUROC significance

Usage:
    python scripts/compute_significance.py
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def bootstrap_auroc(y_true, y_prob, n_bootstrap=10000, seed=42):
    """Bootstrap confidence interval for AUROC."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(seed)
    n = len(y_true)
    aurocs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        # Ensure both classes present
        if len(np.unique(y_true[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(y_true[idx], y_prob[idx]))

    aurocs = np.array(aurocs)
    return {
        "mean": aurocs.mean(),
        "std": aurocs.std(),
        "ci_2.5": np.percentile(aurocs, 2.5),
        "ci_97.5": np.percentile(aurocs, 97.5),
        "n_valid": len(aurocs),
    }


def bootstrap_auroc_diff(y_true, y_prob_a, y_prob_b, n_bootstrap=10000, seed=42):
    """Bootstrap p-value for AUROC difference between two models."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(seed)
    n = len(y_true)

    observed_diff = roc_auc_score(y_true, y_prob_a) - roc_auc_score(y_true, y_prob_b)
    diffs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        diff = roc_auc_score(y_true[idx], y_prob_a[idx]) - roc_auc_score(y_true[idx], y_prob_b[idx])
        diffs.append(diff)

    diffs = np.array(diffs)
    # Two-sided p-value: fraction of bootstraps where diff <= 0
    p_value = (diffs <= 0).mean()

    return {
        "observed_diff": observed_diff,
        "mean_diff": diffs.mean(),
        "std_diff": diffs.std(),
        "ci_2.5": np.percentile(diffs, 2.5),
        "ci_97.5": np.percentile(diffs, 97.5),
        "p_value": p_value,
    }


def permutation_test(y_true, y_prob, n_permutations=10000, seed=42):
    """Permutation test: is AUROC significantly better than random?"""
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(seed)
    observed_auroc = roc_auc_score(y_true, y_prob)

    null_aurocs = []
    for _ in range(n_permutations):
        perm_labels = rng.permutation(y_true)
        if len(np.unique(perm_labels)) < 2:
            continue
        null_aurocs.append(roc_auc_score(perm_labels, y_prob))

    null_aurocs = np.array(null_aurocs)
    p_value = (null_aurocs >= observed_auroc).mean()

    return {
        "observed_auroc": observed_auroc,
        "null_mean": null_aurocs.mean(),
        "null_std": null_aurocs.std(),
        "p_value": p_value,
    }


def main():
    data_dir = Path("data")
    features_dir = data_dir / "features"

    logger.info("Loading validation data and predictions...")

    # Load validation features and labels
    val_data = np.load(features_dir / "module1_features_val.npz", allow_pickle=True)
    y_val = val_data["labels"]
    X_val_seq = val_data["sequence_features"]
    X_val_topo = val_data["topology_features"]
    X_val_frag = val_data["fragile_site_features"]
    X_val_cn = val_data["copy_number_features"]

    logger.info(f"Validation set: {len(y_val)} samples, {int(y_val.sum())} ecDNA+")

    # Load trained ecDNA-Former and get predictions
    import torch
    from src.models import ECDNAFormer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ECDNAFormer()

    ckpt_path = Path("checkpoints/best.pt")
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        logger.info("Loaded ecDNA-Former checkpoint")
    else:
        logger.error(f"Checkpoint not found at {ckpt_path}")
        return

    model.to(device)
    model.eval()

    with torch.no_grad():
        batch = {
            "sequence_features": torch.FloatTensor(X_val_seq).to(device),
            "topology_features": torch.FloatTensor(X_val_topo).to(device),
            "fragile_site_features": torch.FloatTensor(X_val_frag).to(device),
            "copy_number_features": torch.FloatTensor(X_val_cn).to(device),
        }
        outputs = model(**batch)
        y_prob_former = outputs["formation_probability"].cpu().numpy().flatten()

    # Random Forest baseline
    logger.info("Training Random Forest baseline...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    train_data = np.load(features_dir / "module1_features_train.npz", allow_pickle=True)
    feature_names = list(train_data["feature_names"])

    # Reconstruct flat feature matrix from NPZ
    X_train = np.hstack([
        train_data["sequence_features"][:, :len(feature_names)],
    ])[:, :len(feature_names)]
    y_train = train_data["labels"]

    X_val_flat = np.hstack([
        X_val_seq[:, :len(feature_names)],
    ])[:, :len(feature_names)]

    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    y_prob_rf = rf.predict_proba(X_val_flat)[:, 1]

    auroc_former = roc_auc_score(y_val, y_prob_former)
    auroc_rf = roc_auc_score(y_val, y_prob_rf)

    logger.info(f"ecDNA-Former AUROC: {auroc_former:.3f}")
    logger.info(f"Random Forest AUROC: {auroc_rf:.3f}")

    # 1. Bootstrap CI for ecDNA-Former
    logger.info("\n--- Bootstrap CI (ecDNA-Former) ---")
    ci_former = bootstrap_auroc(y_val, y_prob_former)
    logger.info(f"AUROC: {ci_former['mean']:.3f} ± {ci_former['std']:.3f}")
    logger.info(f"95% CI: [{ci_former['ci_2.5']:.3f}, {ci_former['ci_97.5']:.3f}]")

    # 2. Bootstrap CI for RF
    logger.info("\n--- Bootstrap CI (Random Forest) ---")
    ci_rf = bootstrap_auroc(y_val, y_prob_rf)
    logger.info(f"AUROC: {ci_rf['mean']:.3f} ± {ci_rf['std']:.3f}")
    logger.info(f"95% CI: [{ci_rf['ci_2.5']:.3f}, {ci_rf['ci_97.5']:.3f}]")

    # 3. Bootstrap comparison (ecDNA-Former vs RF)
    logger.info("\n--- Bootstrap Comparison (Former vs RF) ---")
    comparison = bootstrap_auroc_diff(y_val, y_prob_former, y_prob_rf)
    logger.info(f"AUROC diff: {comparison['observed_diff']:.3f}")
    logger.info(f"95% CI of diff: [{comparison['ci_2.5']:.3f}, {comparison['ci_97.5']:.3f}]")
    logger.info(f"P-value (Former > RF): {comparison['p_value']:.4f}")

    # 4. Permutation test
    logger.info("\n--- Permutation Test (Former vs random) ---")
    perm = permutation_test(y_val, y_prob_former)
    logger.info(f"Observed AUROC: {perm['observed_auroc']:.3f}")
    logger.info(f"Null AUROC: {perm['null_mean']:.3f} ± {perm['null_std']:.3f}")
    logger.info(f"P-value: {perm['p_value']:.4f}")

    # Save all results
    output_dir = data_dir / "validation"
    output_dir.mkdir(exist_ok=True)

    results = {
        "former_auroc": auroc_former,
        "former_ci_low": ci_former["ci_2.5"],
        "former_ci_high": ci_former["ci_97.5"],
        "rf_auroc": auroc_rf,
        "rf_ci_low": ci_rf["ci_2.5"],
        "rf_ci_high": ci_rf["ci_97.5"],
        "diff_auroc": comparison["observed_diff"],
        "diff_ci_low": comparison["ci_2.5"],
        "diff_ci_high": comparison["ci_97.5"],
        "diff_p_value": comparison["p_value"],
        "perm_p_value": perm["p_value"],
        "null_auroc_mean": perm["null_mean"],
    }

    pd.DataFrame([results]).to_csv(output_dir / "significance_results.csv", index=False)

    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"ecDNA-Former: {auroc_former:.3f} [{ci_former['ci_2.5']:.3f}, {ci_former['ci_97.5']:.3f}]")
    logger.info(f"Random Forest: {auroc_rf:.3f} [{ci_rf['ci_2.5']:.3f}, {ci_rf['ci_97.5']:.3f}]")
    logger.info(f"Difference: {comparison['observed_diff']:+.3f} (p={comparison['p_value']:.4f})")
    logger.info(f"vs Random: p={perm['p_value']:.4f}")
    logger.info(f"\nResults saved to {output_dir / 'significance_results.csv'}")


if __name__ == "__main__":
    main()
