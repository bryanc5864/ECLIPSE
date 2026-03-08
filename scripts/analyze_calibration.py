#!/usr/bin/env python3
"""
Calibration and threshold analysis for ecDNA-Former.

1. Reliability diagram (calibration curve)
2. Precision-recall curve with optimal threshold
3. Threshold sweep (F1, MCC, balanced accuracy at each threshold)
4. Brier score decomposition

Usage:
    python scripts/analyze_calibration.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, f1_score,
    brier_score_loss, matthews_corrcoef, balanced_accuracy_score,
    average_precision_score, confusion_matrix,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calibration_curve_custom(y_true, y_prob, n_bins=10):
    """Compute calibration curve."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_means = []
    bin_true_fracs = []
    bin_counts = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() > 0:
            bin_means.append(y_prob[mask].mean())
            bin_true_fracs.append(y_true[mask].mean())
            bin_counts.append(mask.sum())

    return np.array(bin_means), np.array(bin_true_fracs), np.array(bin_counts)


def main():
    data_dir = Path("data")

    # Load val data
    val_data = np.load(data_dir / "features" / "module1_features_val.npz", allow_pickle=True)
    y_val = val_data["labels"]

    # Load model and predict
    from src.models import ECDNAFormer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ECDNAFormer()

    ckpt = Path("checkpoints/best.pt")
    checkpoint = torch.load(ckpt, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    with torch.no_grad():
        batch = {
            "sequence_features": torch.FloatTensor(val_data["sequence_features"]).to(device),
            "topology_features": torch.FloatTensor(val_data["topology_features"]).to(device),
            "fragile_site_features": torch.FloatTensor(val_data["fragile_site_features"]).to(device),
            "copy_number_features": torch.FloatTensor(val_data["copy_number_features"]).to(device),
        }
        outputs = model(**batch)
        y_prob = outputs["formation_probability"].cpu().numpy().flatten()

    logger.info(f"Validation: {len(y_val)} samples, {int(y_val.sum())} ecDNA+")
    logger.info(f"Prediction range: [{y_prob.min():.3f}, {y_prob.max():.3f}]")
    logger.info(f"Mean prediction: {y_prob.mean():.3f} (base rate: {y_val.mean():.3f})")

    # 1. Calibration curve
    logger.info(f"\n{'='*60}")
    logger.info("CALIBRATION CURVE")
    logger.info(f"{'='*60}")

    bin_means, bin_fracs, bin_counts = calibration_curve_custom(y_val, y_prob, n_bins=10)
    for m, f, c in zip(bin_means, bin_fracs, bin_counts):
        logger.info(f"  Predicted: {m:.2f}, Actual: {f:.2f}, Count: {c}")

    brier = brier_score_loss(y_val, y_prob)
    logger.info(f"\nBrier score: {brier:.4f}")

    # Expected calibration error
    ece = np.sum(np.abs(bin_fracs - bin_means) * bin_counts) / np.sum(bin_counts)
    logger.info(f"Expected Calibration Error (ECE): {ece:.4f}")

    # 2. Threshold sweep
    logger.info(f"\n{'='*60}")
    logger.info("THRESHOLD SWEEP")
    logger.info(f"{'='*60}")

    thresholds = np.arange(0.05, 0.95, 0.05)
    sweep_results = []

    logger.info(f"{'Threshold':>10} {'F1':>6} {'MCC':>6} {'BalAcc':>7} {'Prec':>6} {'Recall':>7} {'Spec':>6} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}")
    logger.info("-" * 80)

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
            continue

        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        f1 = f1_score(y_val, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_val, y_pred)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        prec = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)

        sweep_results.append({
            "threshold": t, "f1": f1, "mcc": mcc, "balanced_accuracy": bal_acc,
            "precision": prec, "recall": recall, "specificity": spec,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })

        logger.info(f"{t:>10.2f} {f1:>6.3f} {mcc:>6.3f} {bal_acc:>7.3f} "
                    f"{prec:>6.3f} {recall:>7.3f} {spec:>6.3f} "
                    f"{tp:>4d} {fp:>4d} {tn:>4d} {fn:>4d}")

    sweep_df = pd.DataFrame(sweep_results)

    # Find optimal thresholds
    if len(sweep_df) > 0:
        best_f1_row = sweep_df.loc[sweep_df["f1"].idxmax()]
        best_mcc_row = sweep_df.loc[sweep_df["mcc"].idxmax()]
        best_bal_row = sweep_df.loc[sweep_df["balanced_accuracy"].idxmax()]

        logger.info(f"\nOptimal thresholds:")
        logger.info(f"  Best F1={best_f1_row['f1']:.3f} at threshold={best_f1_row['threshold']:.2f}")
        logger.info(f"  Best MCC={best_mcc_row['mcc']:.3f} at threshold={best_mcc_row['threshold']:.2f}")
        logger.info(f"  Best BalAcc={best_bal_row['balanced_accuracy']:.3f} at threshold={best_bal_row['threshold']:.2f}")

    # 3. Precision-Recall curve
    logger.info(f"\n{'='*60}")
    logger.info("PRECISION-RECALL CURVE")
    logger.info(f"{'='*60}")

    prec_curve, recall_curve, pr_thresholds = precision_recall_curve(y_val, y_prob)
    auprc = average_precision_score(y_val, y_prob)
    logger.info(f"AUPRC: {auprc:.3f}")
    logger.info(f"AUROC: {roc_auc_score(y_val, y_prob):.3f}")

    # Save
    output_dir = data_dir / "validation"
    output_dir.mkdir(exist_ok=True)

    sweep_df.to_csv(output_dir / "threshold_sweep.csv", index=False)

    cal_df = pd.DataFrame({
        "bin_mean_predicted": bin_means,
        "bin_actual_fraction": bin_fracs,
        "bin_count": bin_counts,
    })
    cal_df.to_csv(output_dir / "calibration_curve.csv", index=False)

    summary = {
        "brier_score": brier,
        "ece": ece,
        "auroc": roc_auc_score(y_val, y_prob),
        "auprc": auprc,
        "best_f1_threshold": best_f1_row["threshold"] if len(sweep_df) > 0 else None,
        "best_f1": best_f1_row["f1"] if len(sweep_df) > 0 else None,
        "best_mcc_threshold": best_mcc_row["threshold"] if len(sweep_df) > 0 else None,
        "best_mcc": best_mcc_row["mcc"] if len(sweep_df) > 0 else None,
    }
    pd.DataFrame([summary]).to_csv(output_dir / "calibration_summary.csv", index=False)

    logger.info(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
