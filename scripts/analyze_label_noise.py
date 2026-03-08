#!/usr/bin/env python3
"""
Label noise sensitivity analysis.

839/1383 training samples have NO ecDNA label (NaN) but are treated as negative.
This script analyzes: what does the current model predict for these unlabeled samples?
Are any predicted as ecDNA+? How does this affect metrics?

Usage:
    python scripts/analyze_label_noise.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    data_dir = Path("data")

    # Load CytoCellDB with full labels
    cyto = pd.read_excel(data_dir / "cytocell_db" / "CytoCellDB_Supp_File1.xlsx")
    cyto = cyto.dropna(subset=["DepMap_ID"])

    # Load features
    train_data = np.load(data_dir / "features" / "module1_features_train.npz", allow_pickle=True)
    val_data = np.load(data_dir / "features" / "module1_features_val.npz", allow_pickle=True)

    # Combine
    all_ids = np.concatenate([train_data["sample_ids"], val_data["sample_ids"]])
    all_labels = np.concatenate([train_data["labels"], val_data["labels"]])
    all_seq = np.concatenate([train_data["sequence_features"], val_data["sequence_features"]])
    all_topo = np.concatenate([train_data["topology_features"], val_data["topology_features"]])
    all_frag = np.concatenate([train_data["fragile_site_features"], val_data["fragile_site_features"]])
    all_cn = np.concatenate([train_data["copy_number_features"], val_data["copy_number_features"]])

    # Map to CytoCellDB labels
    cyto_labels = dict(zip(cyto["DepMap_ID"], cyto["ECDNA"]))
    label_status = np.array([cyto_labels.get(sid, "unknown") for sid in all_ids])

    labeled_y = label_status == "Y"
    labeled_n = label_status == "N"
    labeled_p = label_status == "P"
    unlabeled = ~(labeled_y | labeled_n | labeled_p)

    logger.info(f"Total samples: {len(all_ids)}")
    logger.info(f"  Y (ecDNA+): {labeled_y.sum()}")
    logger.info(f"  N (ecDNA-): {labeled_n.sum()}")
    logger.info(f"  P (Possible): {labeled_p.sum()}")
    logger.info(f"  Unlabeled (NaN): {unlabeled.sum()}")

    # Load model and predict on ALL samples
    from src.models import ECDNAFormer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ECDNAFormer()

    ckpt = Path("checkpoints/best.pt")
    if ckpt.exists():
        checkpoint = torch.load(ckpt, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        logger.error("No checkpoint found")
        return

    model.to(device)
    model.eval()

    # Predict in batches
    predictions = []
    batch_size = 64
    for i in range(0, len(all_ids), batch_size):
        with torch.no_grad():
            batch = {
                "sequence_features": torch.FloatTensor(all_seq[i:i+batch_size]).to(device),
                "topology_features": torch.FloatTensor(all_topo[i:i+batch_size]).to(device),
                "fragile_site_features": torch.FloatTensor(all_frag[i:i+batch_size]).to(device),
                "copy_number_features": torch.FloatTensor(all_cn[i:i+batch_size]).to(device),
            }
            out = model(**batch)
            predictions.extend(out["formation_probability"].cpu().numpy().flatten().tolist())

    predictions = np.array(predictions)

    # Analysis by label group
    logger.info(f"\n{'='*60}")
    logger.info("PREDICTIONS BY LABEL GROUP")
    logger.info(f"{'='*60}")

    for name, mask in [("Y (ecDNA+)", labeled_y), ("N (ecDNA-)", labeled_n),
                       ("P (Possible)", labeled_p), ("Unlabeled", unlabeled)]:
        if mask.sum() == 0:
            continue
        preds = predictions[mask]
        logger.info(f"\n  {name} (n={mask.sum()}):")
        logger.info(f"    Mean prediction: {preds.mean():.3f}")
        logger.info(f"    Median: {np.median(preds):.3f}")
        logger.info(f"    >0.5: {(preds > 0.5).sum()} ({(preds > 0.5).mean():.1%})")
        logger.info(f"    >0.35: {(preds > 0.35).sum()} ({(preds > 0.35).mean():.1%})")
        logger.info(f"    >0.2: {(preds > 0.2).sum()} ({(preds > 0.2).mean():.1%})")
        logger.info(f"    Range: [{preds.min():.3f}, {preds.max():.3f}]")

    # AUROC on labeled-only vs all
    logger.info(f"\n{'='*60}")
    logger.info("AUROC COMPARISON: Labeled-only vs All samples")
    logger.info(f"{'='*60}")

    # Current: Y=1, everything else=0
    auroc_all = roc_auc_score(all_labels, predictions)
    logger.info(f"  AUROC (all {len(all_labels)} samples): {auroc_all:.3f}")

    # Labeled only: Y=1, N=0, exclude P and unlabeled
    labeled_mask = labeled_y | labeled_n
    if labeled_mask.sum() > 0 and labeled_y.sum() > 0:
        auroc_labeled = roc_auc_score(all_labels[labeled_mask], predictions[labeled_mask])
        logger.info(f"  AUROC (Y+N only, {labeled_mask.sum()} samples): {auroc_labeled:.3f}")

    # Labeled + P: Y+P=1, N=0
    labeled_p_mask = labeled_y | labeled_n | labeled_p
    labels_with_p = np.zeros(len(all_labels))
    labels_with_p[labeled_y] = 1
    labels_with_p[labeled_p] = 1
    if labeled_p_mask.sum() > 0:
        auroc_with_p = roc_auc_score(labels_with_p[labeled_p_mask], predictions[labeled_p_mask])
        logger.info(f"  AUROC (Y+P as pos, N as neg, {labeled_p_mask.sum()} samples): {auroc_with_p:.3f}")

    # Unlabeled samples that model predicts as ecDNA+
    logger.info(f"\n{'='*60}")
    logger.info("POTENTIALLY MISLABELED SAMPLES")
    logger.info(f"{'='*60}")

    high_pred_unlabeled = unlabeled & (predictions > 0.35)
    if high_pred_unlabeled.sum() > 0:
        logger.info(f"\n  Unlabeled samples predicted ecDNA+ (>0.35): {high_pred_unlabeled.sum()}")
        for idx in np.where(high_pred_unlabeled)[0][:20]:
            sid = all_ids[idx]
            lineage = cyto[cyto["DepMap_ID"] == sid]["lineage"].values
            lineage_str = lineage[0] if len(lineage) > 0 else "unknown"
            logger.info(f"    {sid}: pred={predictions[idx]:.3f}, lineage={lineage_str}")

    high_pred_neg = labeled_n & (predictions > 0.35)
    if high_pred_neg.sum() > 0:
        logger.info(f"\n  N-labeled samples predicted ecDNA+ (>0.35): {high_pred_neg.sum()}")
        for idx in np.where(high_pred_neg)[0][:10]:
            sid = all_ids[idx]
            lineage = cyto[cyto["DepMap_ID"] == sid]["lineage"].values
            lineage_str = lineage[0] if len(lineage) > 0 else "unknown"
            logger.info(f"    {sid}: pred={predictions[idx]:.3f}, lineage={lineage_str}")

    # Save detailed results
    output_dir = data_dir / "validation"
    output_dir.mkdir(exist_ok=True)

    results_df = pd.DataFrame({
        "sample_id": all_ids,
        "label_status": label_status,
        "training_label": all_labels,
        "prediction": predictions,
    })
    results_df.to_csv(output_dir / "label_noise_analysis.csv", index=False)

    summary = {
        "auroc_all": auroc_all,
        "auroc_labeled_only": auroc_labeled if labeled_mask.sum() > 0 else None,
        "auroc_with_possible": auroc_with_p if labeled_p_mask.sum() > 0 else None,
        "n_unlabeled_pred_positive": int(high_pred_unlabeled.sum()),
        "n_neg_pred_positive": int(high_pred_neg.sum()),
        "mean_pred_Y": float(predictions[labeled_y].mean()),
        "mean_pred_N": float(predictions[labeled_n].mean()),
        "mean_pred_P": float(predictions[labeled_p].mean()) if labeled_p.sum() > 0 else None,
        "mean_pred_unlabeled": float(predictions[unlabeled].mean()) if unlabeled.sum() > 0 else None,
    }
    pd.DataFrame([summary]).to_csv(output_dir / "label_noise_summary.csv", index=False)

    logger.info(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
