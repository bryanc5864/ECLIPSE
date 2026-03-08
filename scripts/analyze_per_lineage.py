#!/usr/bin/env python3
"""
Per-lineage prediction analysis for ecDNA-Former.

Breaks down model predictions by cancer lineage to identify:
1. Which lineages the model performs best/worst on
2. Whether certain lineages are systematically over/under-predicted
3. Lineage-specific ecDNA prevalence vs model prediction

Usage:
    python scripts/analyze_per_lineage.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    data_dir = Path("data")

    # Load CytoCellDB for lineage info
    cyto = pd.read_excel(data_dir / "cytocell_db" / "CytoCellDB_Supp_File1.xlsx")
    cyto = cyto.dropna(subset=["DepMap_ID"])
    lineage_map = dict(zip(cyto["DepMap_ID"], cyto["lineage"]))
    ecdna_map = dict(zip(cyto["DepMap_ID"], cyto["ECDNA"]))

    # Load val data
    val_data = np.load(data_dir / "features" / "module1_features_val.npz", allow_pickle=True)
    val_ids = val_data["sample_ids"]
    val_labels = val_data["labels"]

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
        predictions = outputs["formation_probability"].cpu().numpy().flatten()

    # Also do training set
    train_data = np.load(data_dir / "features" / "module1_features_train.npz", allow_pickle=True)
    train_ids = train_data["sample_ids"]
    train_labels = train_data["labels"]

    with torch.no_grad():
        batch = {
            "sequence_features": torch.FloatTensor(train_data["sequence_features"]).to(device),
            "topology_features": torch.FloatTensor(train_data["topology_features"]).to(device),
            "fragile_site_features": torch.FloatTensor(train_data["fragile_site_features"]).to(device),
            "copy_number_features": torch.FloatTensor(train_data["copy_number_features"]).to(device),
        }
        outputs = model(**batch)
        train_predictions = outputs["formation_probability"].cpu().numpy().flatten()

    # Combine for full analysis
    all_ids = np.concatenate([train_ids, val_ids])
    all_labels = np.concatenate([train_labels, val_labels])
    all_preds = np.concatenate([train_predictions, predictions])
    all_splits = np.array(["train"] * len(train_ids) + ["val"] * len(val_ids))

    # Build per-sample DataFrame
    records = []
    for i, sid in enumerate(all_ids):
        records.append({
            "sample_id": sid,
            "lineage": lineage_map.get(sid, "unknown"),
            "ecdna_label": ecdna_map.get(sid, "unknown"),
            "training_label": int(all_labels[i]),
            "prediction": all_preds[i],
            "split": all_splits[i],
        })
    df = pd.DataFrame(records)

    # Per-lineage analysis (val set only)
    logger.info(f"\n{'='*60}")
    logger.info("PER-LINEAGE BREAKDOWN (validation set)")
    logger.info(f"{'='*60}")

    val_df = df[df["split"] == "val"]
    lineage_results = []

    for lineage in sorted(val_df["lineage"].dropna().unique()):
        ldf = val_df[val_df["lineage"] == lineage]
        n_total = len(ldf)
        n_pos = int(ldf["training_label"].sum())
        mean_pred = ldf["prediction"].mean()

        result = {
            "lineage": lineage,
            "n_val": n_total,
            "n_pos": n_pos,
            "prevalence": n_pos / n_total if n_total > 0 else 0,
            "mean_prediction": mean_pred,
        }

        if n_pos > 0 and n_pos < n_total:
            try:
                result["auroc"] = roc_auc_score(ldf["training_label"], ldf["prediction"])
            except ValueError:
                result["auroc"] = float("nan")
        else:
            result["auroc"] = float("nan")

        lineage_results.append(result)

    lineage_df = pd.DataFrame(lineage_results).sort_values("n_val", ascending=False)

    logger.info(f"\n{'Lineage':>25s} {'N':>4s} {'Pos':>4s} {'Prev':>6s} {'MeanPred':>9s} {'AUROC':>7s}")
    logger.info("-" * 65)
    for _, row in lineage_df.iterrows():
        auroc_str = f"{row['auroc']:.3f}" if not np.isnan(row.get("auroc", float("nan"))) else "N/A"
        logger.info(f"{row['lineage']:>25s} {row['n_val']:>4.0f} {row['n_pos']:>4.0f} "
                    f"{row['prevalence']:>6.1%} {row['mean_prediction']:>9.3f} {auroc_str:>7s}")

    # Full dataset lineage breakdown
    logger.info(f"\n{'='*60}")
    logger.info("PER-LINEAGE BREAKDOWN (all samples)")
    logger.info(f"{'='*60}")

    full_results = []
    for lineage in sorted(df["lineage"].dropna().unique()):
        ldf = df[df["lineage"] == lineage]
        n_total = len(ldf)
        n_pos = int(ldf["training_label"].sum())
        n_labeled_y = int((ldf["ecdna_label"] == "Y").sum())
        n_labeled_n = int((ldf["ecdna_label"] == "N").sum())
        n_unlabeled = n_total - n_labeled_y - n_labeled_n - int((ldf["ecdna_label"] == "P").sum())
        mean_pred = ldf["prediction"].mean()

        full_results.append({
            "lineage": lineage,
            "n_total": n_total,
            "n_Y": n_labeled_y,
            "n_N": n_labeled_n,
            "n_unlabeled": n_unlabeled,
            "prevalence_Y": n_labeled_y / n_total if n_total > 0 else 0,
            "mean_prediction": mean_pred,
            "pred_gt_035": int((ldf["prediction"] > 0.35).sum()),
        })

    full_df = pd.DataFrame(full_results).sort_values("n_total", ascending=False)

    logger.info(f"\n{'Lineage':>25s} {'Total':>5s} {'Y':>4s} {'N':>4s} {'Unl':>4s} {'Prev':>6s} {'MeanP':>7s} {'>0.35':>5s}")
    logger.info("-" * 70)
    for _, row in full_df.head(20).iterrows():
        logger.info(f"{row['lineage']:>25s} {row['n_total']:>5.0f} {row['n_Y']:>4.0f} "
                    f"{row['n_N']:>4.0f} {row['n_unlabeled']:>4.0f} {row['prevalence_Y']:>6.1%} "
                    f"{row['mean_prediction']:>7.3f} {row['pred_gt_035']:>5.0f}")

    # Save results
    output_dir = data_dir / "validation"
    output_dir.mkdir(exist_ok=True)
    lineage_df.to_csv(output_dir / "per_lineage_val.csv", index=False)
    full_df.to_csv(output_dir / "per_lineage_all.csv", index=False)
    df.to_csv(output_dir / "per_sample_predictions.csv", index=False)

    logger.info(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
