#!/usr/bin/env python3
"""
Leave-one-lineage-out cross-validation for ecDNA-Former (Module 1).

Holds out each cancer lineage, trains on remaining lineages, evaluates
on held-out lineage. Tests whether the model generalizes across tissue types.

Saves results to data/validation/lineage_loocv_results.csv.

Usage:
    python scripts/run_lineage_loocv.py --epochs 200 --patience 30
    python scripts/run_lineage_loocv.py --epochs 200 --patience 30 --min-samples 20
"""

import argparse
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_lineage_info(data_dir: Path):
    """Load lineage info for each sample in our dataset."""
    cyto = pd.read_excel(data_dir / "cytocell_db" / "CytoCellDB_Supp_File1.xlsx")
    lineage_map = dict(zip(cyto["DepMap_ID"], cyto["lineage"]))
    return lineage_map


def load_full_dataset(data_dir: Path):
    """Load all features and labels (train + val combined)."""
    train = np.load(data_dir / "features" / "module1_features_train.npz", allow_pickle=True)
    val = np.load(data_dir / "features" / "module1_features_val.npz", allow_pickle=True)

    combined = {}
    for key in ["sequence_features", "topology_features", "fragile_site_features", "copy_number_features"]:
        combined[key] = np.concatenate([train[key], val[key]], axis=0)

    combined["labels"] = np.concatenate([train["labels"], val["labels"]], axis=0)
    combined["sample_ids"] = np.concatenate([train["sample_ids"], val["sample_ids"]], axis=0)
    combined["feature_names"] = train["feature_names"]

    return combined


def save_npz(data, indices, output_path):
    """Save a subset of data as NPZ."""
    np.savez(
        output_path,
        sequence_features=data["sequence_features"][indices],
        topology_features=data["topology_features"][indices],
        fragile_site_features=data["fragile_site_features"][indices],
        copy_number_features=data["copy_number_features"][indices],
        labels=data["labels"][indices],
        sample_ids=data["sample_ids"][indices],
        feature_names=data["feature_names"],
    )


def train_and_evaluate(lineage_name, fold_data_dir, checkpoint_dir, epochs, patience, device):
    """Train on all lineages except one, evaluate on held-out lineage."""
    from src.data import ECDNADataset, create_dataloader
    from src.models import ECDNAFormer
    from src.training import ECDNAFormerTrainer

    fold_ckpt = Path(checkpoint_dir) / f"lineage_{lineage_name}"
    fold_ckpt.mkdir(parents=True, exist_ok=True)

    train_dataset = ECDNADataset.from_data_dir(data_dir=fold_data_dir, split="train")
    val_dataset = ECDNADataset.from_data_dir(data_dir=fold_data_dir, split="val")

    n_val_pos = int(val_dataset.labels.sum()) if hasattr(val_dataset, 'labels') else -1

    # Skip if val set has 0 positives (can't compute AUROC)
    if n_val_pos == 0:
        logger.warning(f"  Lineage {lineage_name}: 0 ecDNA+ in held-out set, skipping training")
        return {
            "lineage": lineage_name,
            "n_train": len(train_dataset),
            "n_val": len(val_dataset),
            "n_val_pos": 0,
            "auroc": float("nan"),
            "note": "skipped (0 positives)",
        }

    train_loader = create_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=32, shuffle=False)

    model = ECDNAFormer()
    trainer = ECDNAFormerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=str(fold_ckpt),
        use_wandb=False,
    )

    trainer.train(num_epochs=epochs, early_stopping_patience=patience)

    # Extract best metrics
    log_dir = fold_ckpt / "logs"
    val_logs = sorted(log_dir.glob("validation_log_*.csv"))
    if val_logs:
        val_df = pd.read_csv(val_logs[-1])
        best_row = val_df.loc[val_df["auroc"].idxmax()]
        return {
            "lineage": lineage_name,
            "n_train": len(train_dataset),
            "n_val": len(val_dataset),
            "n_val_pos": n_val_pos,
            "best_epoch": int(best_row["epoch"]),
            "auroc": best_row["auroc"],
            "auprc": best_row["auprc"],
            "f1_score": best_row["f1_score"],
            "mcc": best_row["mcc"],
            "recall": best_row["recall"],
            "precision": best_row["precision"],
        }

    return {"lineage": lineage_name, "auroc": float("nan"), "note": "no val logs"}


def main():
    parser = argparse.ArgumentParser(description="Leave-one-lineage-out CV")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/lineage_loocv")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-samples", type=int, default=10,
                        help="Min samples in lineage to include as held-out fold")
    parser.add_argument("--min-positives", type=int, default=2,
                        help="Min ecDNA+ in lineage to include as held-out fold")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    data_dir = Path(args.data_dir)
    features_dir = data_dir / "features"

    # Load data and lineage info
    combined = load_full_dataset(data_dir)
    lineage_map = load_lineage_info(data_dir)

    sample_ids = combined["sample_ids"]
    labels = combined["labels"]
    lineages = np.array([lineage_map.get(sid, "unknown") for sid in sample_ids])

    # Find eligible lineages
    lineage_stats = []
    for lin in sorted(set(lineages)):
        mask = lineages == lin
        n_total = mask.sum()
        n_pos = labels[mask].sum()
        lineage_stats.append({"lineage": lin, "total": n_total, "ecDNA_pos": int(n_pos)})

    stats_df = pd.DataFrame(lineage_stats).sort_values("total", ascending=False)
    logger.info(f"\nLineage distribution (n={len(stats_df)}):")
    for _, row in stats_df.head(15).iterrows():
        logger.info(f"  {row['lineage']:>30s}: {row['total']:4d} total, {row['ecDNA_pos']:3d} ecDNA+")

    eligible = stats_df[
        (stats_df["total"] >= args.min_samples) &
        (stats_df["ecDNA_pos"] >= args.min_positives)
    ]
    logger.info(f"\nEligible lineages (>={args.min_samples} samples, >={args.min_positives} ecDNA+): "
                f"{len(eligible)}")

    # Use a temp directory for fold data to avoid overwriting originals
    import tempfile
    import os
    fold_data_dir = Path(tempfile.mkdtemp(prefix="eclipse_lineage_"))
    fold_features_dir = fold_data_dir / "features"
    fold_features_dir.mkdir(parents=True, exist_ok=True)
    # Symlink required subdirectories so ECDNADataset.from_data_dir works
    for subdir in ["ecdna_labels", "cytocell_db", "depmap", "hic", "supplementary"]:
        src = data_dir / subdir
        if src.exists():
            os.symlink(src.resolve(), fold_data_dir / subdir)
    logger.info(f"Using temp directory for fold data: {fold_data_dir}")

    all_results = []

    try:
        for _, row in eligible.iterrows():
            lineage_name = row["lineage"]
            logger.info(f"\n{'='*60}")
            logger.info(f"HELD-OUT: {lineage_name} ({int(row['total'])} samples, "
                        f"{int(row['ecDNA_pos'])} ecDNA+)")
            logger.info(f"{'='*60}")

            # Create train (everything else) and val (this lineage) indices
            val_mask = lineages == lineage_name
            train_mask = ~val_mask

            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]

            # Save fold NPZ to temp directory
            save_npz(combined, train_idx, fold_features_dir / "module1_features_train.npz")
            save_npz(combined, val_idx, fold_features_dir / "module1_features_val.npz")

            start = time.time()
            result = train_and_evaluate(
                lineage_name, str(fold_data_dir), args.checkpoint_dir,
                args.epochs, args.patience, device
            )
            result["time_sec"] = time.time() - start
            all_results.append(result)

            auroc = result.get("auroc", float("nan"))
            logger.info(f"  AUROC: {auroc:.3f}" if not np.isnan(auroc) else "  AUROC: N/A")

    finally:
        # Clean up temp directory
        shutil.rmtree(fold_data_dir, ignore_errors=True)
        logger.info("\nCleaned up temp fold data directory")

    # Summary
    results_df = pd.DataFrame(all_results)
    output_dir = data_dir / "validation"
    output_dir.mkdir(exist_ok=True)
    results_df.to_csv(output_dir / "lineage_loocv_results.csv", index=False)

    valid = results_df.dropna(subset=["auroc"])
    logger.info(f"\n{'='*60}")
    logger.info("LEAVE-ONE-LINEAGE-OUT RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Lineages evaluated: {len(valid)}/{len(results_df)}")
    if len(valid) > 0:
        logger.info(f"Mean AUROC: {valid['auroc'].mean():.3f} ± {valid['auroc'].std():.3f}")
        logger.info(f"Range: [{valid['auroc'].min():.3f}, {valid['auroc'].max():.3f}]")
        logger.info(f"\nPer-lineage:")
        for _, row in valid.sort_values("auroc", ascending=False).iterrows():
            logger.info(f"  {row['lineage']:>30s}: AUROC={row['auroc']:.3f} "
                        f"(n={row['n_val']}, pos={row.get('n_val_pos', '?')})")

    logger.info(f"\nResults saved to {output_dir / 'lineage_loocv_results.csv'}")


if __name__ == "__main__":
    main()
