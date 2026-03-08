#!/usr/bin/env python3
"""
Prediction confidence and uncertainty analysis for ecDNA-Former.

Analyzes:
1. Prediction score distribution (histogram data)
2. High-confidence correct vs incorrect predictions
3. Entropy of predictions as uncertainty measure
4. Predictions on specific known ecDNA cell lines

Usage:
    python scripts/analyze_prediction_confidence.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    data_dir = Path("data")

    # Load CytoCellDB for context
    cyto = pd.read_excel(data_dir / "cytocell_db" / "CytoCellDB_Supp_File1.xlsx")
    cyto = cyto.dropna(subset=["DepMap_ID"])
    ecdna_map = dict(zip(cyto["DepMap_ID"], cyto["ECDNA"]))
    lineage_map = dict(zip(cyto["DepMap_ID"], cyto["lineage"]))
    cell_line_map = dict(zip(cyto["DepMap_ID"], cyto.get("stripped_cell_line_name", cyto.get("CellLineName", cyto["DepMap_ID"]))))

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

    logger.info(f"Validation: {len(val_ids)} samples, {int(val_labels.sum())} ecDNA+")

    # 1. Score distribution
    logger.info(f"\n{'='*60}")
    logger.info("PREDICTION SCORE DISTRIBUTION")
    logger.info(f"{'='*60}")

    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    counts, _ = np.histogram(predictions, bins=bins)
    pos_counts = np.zeros(len(bins) - 1, dtype=int)
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (predictions >= lo) & (predictions < hi)
        pos_counts[i] = int(val_labels[mask].sum())

    logger.info(f"\n{'Bin':>12s} {'Count':>6s} {'ecDNA+':>7s} {'Pos Rate':>9s}")
    logger.info("-" * 40)
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        rate = pos_counts[i] / counts[i] if counts[i] > 0 else 0
        logger.info(f"  [{lo:.1f}, {hi:.1f}) {counts[i]:>6d} {pos_counts[i]:>7d} {rate:>9.1%}")

    # 2. Binary entropy as uncertainty
    logger.info(f"\n{'='*60}")
    logger.info("UNCERTAINTY (Binary Entropy)")
    logger.info(f"{'='*60}")

    eps = 1e-7
    entropy = -(predictions * np.log2(predictions + eps) + (1 - predictions) * np.log2(1 - predictions + eps))

    logger.info(f"  Mean entropy: {entropy.mean():.3f} (max possible: 1.0)")
    logger.info(f"  High confidence (entropy < 0.5): {(entropy < 0.5).sum()} ({(entropy < 0.5).mean():.1%})")
    logger.info(f"  Uncertain (entropy > 0.8): {(entropy > 0.8).sum()} ({(entropy > 0.8).mean():.1%})")

    # 3. Confident correct vs incorrect
    logger.info(f"\n{'='*60}")
    logger.info("CONFIDENT PREDICTIONS ANALYSIS")
    logger.info(f"{'='*60}")

    threshold = 0.35  # Best F1 threshold from calibration analysis
    y_pred = (predictions >= threshold).astype(int)
    correct = (y_pred == val_labels)

    # True positives (confident correct)
    tp_mask = (y_pred == 1) & (val_labels == 1)
    fp_mask = (y_pred == 1) & (val_labels == 0)
    fn_mask = (y_pred == 0) & (val_labels == 1)
    tn_mask = (y_pred == 0) & (val_labels == 0)

    logger.info(f"\n  At threshold {threshold}:")
    logger.info(f"    True positives: {tp_mask.sum()}")
    logger.info(f"    False positives: {fp_mask.sum()}")
    logger.info(f"    False negatives: {fn_mask.sum()}")
    logger.info(f"    True negatives: {tn_mask.sum()}")

    if tp_mask.sum() > 0:
        logger.info(f"\n  True positives (ecDNA+ correctly identified):")
        for idx in np.where(tp_mask)[0]:
            sid = val_ids[idx]
            logger.info(f"    {cell_line_map.get(sid, sid)}: pred={predictions[idx]:.3f}, "
                       f"lineage={lineage_map.get(sid, 'unknown')}")

    if fn_mask.sum() > 0:
        logger.info(f"\n  False negatives (ecDNA+ missed):")
        for idx in np.where(fn_mask)[0]:
            sid = val_ids[idx]
            logger.info(f"    {cell_line_map.get(sid, sid)}: pred={predictions[idx]:.3f}, "
                       f"lineage={lineage_map.get(sid, 'unknown')}")

    if fp_mask.sum() > 0:
        logger.info(f"\n  False positives (top 10 by prediction score):")
        fp_indices = np.where(fp_mask)[0]
        fp_sorted = fp_indices[np.argsort(-predictions[fp_indices])]
        for idx in fp_sorted[:10]:
            sid = val_ids[idx]
            ecdna_label = ecdna_map.get(sid, "unknown")
            logger.info(f"    {cell_line_map.get(sid, sid)}: pred={predictions[idx]:.3f}, "
                       f"label={ecdna_label}, lineage={lineage_map.get(sid, 'unknown')}")

    # 4. Known ecDNA cell lines
    logger.info(f"\n{'='*60}")
    logger.info("KNOWN ecDNA CELL LINES IN VALIDATION")
    logger.info(f"{'='*60}")

    known_ecdna = ["COLO 320DM", "PC-3", "NCI-H716", "SNU-16", "KELLY", "SK-N-DZ"]

    for _, row in cyto.iterrows():
        if row["DepMap_ID"] in val_ids:
            idx = np.where(val_ids == row["DepMap_ID"])[0]
            if len(idx) > 0:
                idx = idx[0]
                cell_name = cell_line_map.get(row["DepMap_ID"], row["DepMap_ID"])
                ecdna_label = ecdna_map.get(row["DepMap_ID"], "?")
                if ecdna_label == "Y":
                    logger.info(f"  {cell_name}: pred={predictions[idx]:.3f}, "
                               f"label=Y, lineage={lineage_map.get(row['DepMap_ID'], '?')}")

    # Save
    output_dir = data_dir / "validation"
    output_dir.mkdir(exist_ok=True)

    confidence_df = pd.DataFrame({
        "sample_id": val_ids,
        "cell_line": [cell_line_map.get(sid, sid) for sid in val_ids],
        "lineage": [lineage_map.get(sid, "unknown") for sid in val_ids],
        "ecdna_label": [ecdna_map.get(sid, "unknown") for sid in val_ids],
        "training_label": val_labels,
        "prediction": predictions,
        "entropy": entropy,
        "predicted_class": y_pred,
        "correct": correct.astype(int),
    })
    confidence_df.to_csv(output_dir / "prediction_confidence.csv", index=False)

    hist_df = pd.DataFrame({
        "bin_low": bins[:-1],
        "bin_high": bins[1:],
        "count": counts,
        "n_positive": pos_counts,
    })
    hist_df.to_csv(output_dir / "prediction_histogram.csv", index=False)

    logger.info(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
