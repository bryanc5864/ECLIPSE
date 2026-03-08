#!/usr/bin/env python3
"""
5-fold stratified cross-validation for ecDNA-Former (Module 1).

Creates 5 train/val splits, trains each from scratch, reports mean ± std AUROC.
Saves per-fold results to data/validation/crossval_results.csv.

Usage:
    python scripts/run_crossval.py --epochs 200 --patience 30
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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

    logger.info(f"Combined dataset: {len(combined['labels'])} samples, "
                f"{int(combined['labels'].sum())} ecDNA+")
    return combined


def save_fold_npz(data, indices, output_path):
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


def train_fold(fold_idx, fold_data_dir, checkpoint_dir, epochs, patience, device):
    """Train one fold using the existing training pipeline."""
    from src.data import ECDNADataset, create_dataloader
    from src.models import ECDNAFormer
    from src.training import ECDNAFormerTrainer

    fold_ckpt = Path(checkpoint_dir) / f"fold_{fold_idx}"
    fold_ckpt.mkdir(parents=True, exist_ok=True)

    # Load fold data from temp directory (not the original)
    train_dataset = ECDNADataset.from_data_dir(data_dir=fold_data_dir, split="train")
    val_dataset = ECDNADataset.from_data_dir(data_dir=fold_data_dir, split="val")

    train_loader = create_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=32, shuffle=False)

    logger.info(f"  Fold {fold_idx}: train={len(train_dataset)}, val={len(val_dataset)}, "
                f"train ecDNA+={int(train_dataset.labels.sum())}, "
                f"val ecDNA+={int(val_dataset.labels.sum())}")

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

    # Extract best metrics from validation log
    log_dir = fold_ckpt / "logs"
    val_logs = sorted(log_dir.glob("validation_log_*.csv"))
    if val_logs:
        val_df = pd.read_csv(val_logs[-1])
        best_auroc_row = val_df.loc[val_df["auroc"].idxmax()]
        metrics = {
            "fold": fold_idx,
            "best_epoch": int(best_auroc_row["epoch"]),
            "auroc": best_auroc_row["auroc"],
            "auprc": best_auroc_row["auprc"],
            "f1_score": best_auroc_row["f1_score"],
            "precision": best_auroc_row["precision"],
            "recall": best_auroc_row["recall"],
            "balanced_accuracy": best_auroc_row["balanced_accuracy"],
            "mcc": best_auroc_row["mcc"],
            "val_loss": best_auroc_row["val_loss"],
        }
    else:
        metrics = {"fold": fold_idx, "auroc": 0.0}

    return metrics


def main():
    parser = argparse.ArgumentParser(description="5-fold stratified CV for ecDNA-Former")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/crossval")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    data_dir = Path(args.data_dir)
    features_dir = data_dir / "features"

    logger.info(f"Running {args.n_folds}-fold stratified CV on {device}")

    # Load full dataset
    combined = load_full_dataset(data_dir)
    labels = combined["labels"]
    n_samples = len(labels)

    # Create stratified folds
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    folds = list(skf.split(np.zeros(n_samples), labels))

    # Use a temp directory for fold data to avoid overwriting originals
    import tempfile
    import os
    fold_data_dir = Path(tempfile.mkdtemp(prefix="eclipse_cv_"))
    fold_features_dir = fold_data_dir / "features"
    fold_features_dir.mkdir(parents=True, exist_ok=True)
    # Symlink required subdirectories so ECDNADataset.from_data_dir works
    for subdir in ["ecdna_labels", "cytocell_db", "depmap", "hic", "supplementary"]:
        src = data_dir / subdir
        if src.exists():
            os.symlink(src.resolve(), fold_data_dir / subdir)
    logger.info(f"Using temp directory for fold data: {fold_data_dir}")

    all_metrics = []

    try:
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            logger.info(f"\n{'='*60}")
            logger.info(f"FOLD {fold_idx + 1}/{args.n_folds}")
            logger.info(f"{'='*60}")

            # Save fold-specific NPZ files to temp directory
            save_fold_npz(combined, train_idx, fold_features_dir / "module1_features_train.npz")
            save_fold_npz(combined, val_idx, fold_features_dir / "module1_features_val.npz")

            start = time.time()
            metrics = train_fold(
                fold_idx=fold_idx,
                fold_data_dir=str(fold_data_dir),
                checkpoint_dir=args.checkpoint_dir,
                epochs=args.epochs,
                patience=args.patience,
                device=device,
            )
            elapsed = time.time() - start
            metrics["time_sec"] = elapsed

            all_metrics.append(metrics)
            logger.info(f"  Fold {fold_idx} AUROC: {metrics['auroc']:.3f} "
                        f"(epoch {metrics.get('best_epoch', '?')}, {elapsed:.0f}s)")

    finally:
        # Clean up temp directory
        import shutil
        shutil.rmtree(fold_data_dir, ignore_errors=True)
        logger.info("Cleaned up temp fold data directory")

    # Summary
    results_df = pd.DataFrame(all_metrics)
    output_dir = data_dir / "validation"
    output_dir.mkdir(exist_ok=True)
    results_df.to_csv(output_dir / "crossval_results.csv", index=False)

    logger.info(f"\n{'='*60}")
    logger.info("CROSS-VALIDATION RESULTS")
    logger.info(f"{'='*60}")
    for metric in ["auroc", "auprc", "f1_score", "mcc", "balanced_accuracy"]:
        if metric in results_df.columns:
            vals = results_df[metric]
            logger.info(f"  {metric}: {vals.mean():.3f} ± {vals.std():.3f} "
                        f"(range: {vals.min():.3f} - {vals.max():.3f})")

    logger.info(f"\nResults saved to {output_dir / 'crossval_results.csv'}")


if __name__ == "__main__":
    main()
