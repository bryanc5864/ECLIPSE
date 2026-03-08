#!/usr/bin/env python3
"""
Feature group ablation study for ecDNA-Former (Module 1).

Zeroes out each feature group and retrains to measure contribution.
Saves results to data/validation/ablation_results.csv.

Feature groups:
  - Hi-C: cnv_hic_*, cnv_hiclr_*, oncogene_cnv_hic_*, hic_*
  - CNV: cnv_* (oncogene + stats)
  - Expression: expr_*, oncogene_expr_*, n_oncogenes_high_expr
  - Dosage: dosage_*
  - All features (baseline)

Usage:
    python scripts/run_ablation.py --epochs 200 --patience 30
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

# Feature groups defined by prefix
ABLATION_GROUPS = {
    "Hi-C": lambda name: (
        name.startswith("cnv_hic_") or
        name.startswith("cnv_hiclr_") or
        name.startswith("oncogene_cnv_hic_") or
        name.startswith("hic_")
    ),
    "CNV": lambda name: (
        name.startswith("cnv_") and
        not name.startswith("cnv_hic_") and
        not name.startswith("cnv_hiclr_")
    ) or name in ["oncogene_cnv_max", "oncogene_cnv_mean", "n_oncogenes_amplified"],
    "Expression": lambda name: (
        name.startswith("expr_") or
        name.startswith("oncogene_expr_") or
        name == "n_oncogenes_high_expr"
    ),
    "Dosage": lambda name: name.startswith("dosage_"),
}


def zero_feature_group(npz_path, feature_names, group_fn, output_path):
    """Create a copy of NPZ with one feature group zeroed out."""
    data = np.load(npz_path, allow_pickle=True)

    # Find which raw feature indices to zero
    group_indices = [i for i, name in enumerate(feature_names) if group_fn(name)]
    n_zeroed = len(group_indices)
    logger.info(f"    Zeroing {n_zeroed} features")

    # The raw features are packed into 4 arrays. We need to figure out
    # which packed positions correspond to which raw feature indices.
    # From extract_nonleaky_features.py:
    #   sequence_features[:, :N] = X[:, :N]  (first N of all features)
    #   topology_features[:, :N] = X[:, :N]  (same)
    #   fragile_site_features[:, :N] = X[:, :min(64,N)]
    #   copy_number_features = X[:, cnv_cols[:32]]
    #
    # The simplest approach: zero the group indices across ALL packed arrays.
    arrays = {}
    for key in data.files:
        arr = data[key].copy() if hasattr(data[key], 'copy') else data[key]
        arrays[key] = arr

    # Zero in sequence_features (first 112 positions hold raw features)
    for idx in group_indices:
        if idx < arrays["sequence_features"].shape[1]:
            arrays["sequence_features"][:, idx] = 0.0
        if idx < arrays["topology_features"].shape[1]:
            arrays["topology_features"][:, idx] = 0.0
        if idx < arrays["fragile_site_features"].shape[1]:
            arrays["fragile_site_features"][:, idx] = 0.0
        if idx < arrays["copy_number_features"].shape[1]:
            arrays["copy_number_features"][:, idx] = 0.0

    np.savez(output_path, **arrays)
    return n_zeroed


def train_ablation(ablation_name, fold_data_dir, checkpoint_dir, epochs, patience, device):
    """Train one ablation variant."""
    from src.data import ECDNADataset, create_dataloader
    from src.models import ECDNAFormer
    from src.training import ECDNAFormerTrainer

    abl_ckpt = Path(checkpoint_dir) / f"ablation_{ablation_name}"
    abl_ckpt.mkdir(parents=True, exist_ok=True)

    train_dataset = ECDNADataset.from_data_dir(data_dir=fold_data_dir, split="train")
    val_dataset = ECDNADataset.from_data_dir(data_dir=fold_data_dir, split="val")

    train_loader = create_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=32, shuffle=False)

    model = ECDNAFormer()
    trainer = ECDNAFormerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=str(abl_ckpt),
        use_wandb=False,
    )

    trainer.train(num_epochs=epochs, early_stopping_patience=patience)

    # Extract best AUROC metrics
    log_dir = abl_ckpt / "logs"
    val_logs = sorted(log_dir.glob("validation_log_*.csv"))
    if val_logs:
        val_df = pd.read_csv(val_logs[-1])
        best_row = val_df.loc[val_df["auroc"].idxmax()]
        return {
            "ablation": ablation_name,
            "best_epoch": int(best_row["epoch"]),
            "auroc": best_row["auroc"],
            "auprc": best_row["auprc"],
            "f1_score": best_row["f1_score"],
            "mcc": best_row["mcc"],
            "balanced_accuracy": best_row["balanced_accuracy"],
            "recall": best_row["recall"],
            "precision": best_row["precision"],
        }
    return {"ablation": ablation_name, "auroc": 0.0}


def main():
    parser = argparse.ArgumentParser(description="Feature group ablation for ecDNA-Former")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/ablation")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    data_dir = Path(args.data_dir)
    features_dir = data_dir / "features"

    # Load feature names
    train_data = np.load(features_dir / "module1_features_train.npz", allow_pickle=True)
    feature_names = list(train_data["feature_names"])
    logger.info(f"Total features: {len(feature_names)}")

    # Count features per group
    for group_name, group_fn in ABLATION_GROUPS.items():
        count = sum(1 for name in feature_names if group_fn(name))
        logger.info(f"  {group_name}: {count} features")

    # Use a temp directory to avoid overwriting originals
    import tempfile
    import os
    tmp_data_dir = Path(tempfile.mkdtemp(prefix="eclipse_ablation_"))
    tmp_features_dir = tmp_data_dir / "features"
    tmp_features_dir.mkdir(parents=True, exist_ok=True)
    # Symlink required subdirectories so ECDNADataset.from_data_dir works
    for subdir in ["ecdna_labels", "cytocell_db", "depmap", "hic", "supplementary"]:
        src = data_dir / subdir
        if src.exists():
            os.symlink(src.resolve(), tmp_data_dir / subdir)
    logger.info(f"Using temp directory for ablation data: {tmp_data_dir}")

    # Copy original NPZ files to temp directory
    train_npz = features_dir / "module1_features_train.npz"
    val_npz = features_dir / "module1_features_val.npz"
    tmp_train_npz = tmp_features_dir / "module1_features_train.npz"
    tmp_val_npz = tmp_features_dir / "module1_features_val.npz"

    shutil.copy2(train_npz, tmp_train_npz)
    shutil.copy2(val_npz, tmp_val_npz)

    all_results = []

    try:
        # First: train full model baseline
        logger.info(f"\n{'='*60}")
        logger.info("BASELINE (full features)")
        logger.info(f"{'='*60}")
        start = time.time()
        baseline = train_ablation("Full", str(tmp_data_dir), args.checkpoint_dir, args.epochs, args.patience, device)
        baseline["time_sec"] = time.time() - start
        baseline["n_zeroed"] = 0
        all_results.append(baseline)
        logger.info(f"  Baseline AUROC: {baseline['auroc']:.3f}")

        # Ablate each group
        for group_name, group_fn in ABLATION_GROUPS.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"ABLATION: -{group_name}")
            logger.info(f"{'='*60}")

            # Create ablated NPZ files from the original copies
            n_train = zero_feature_group(str(train_npz), feature_names, group_fn, str(tmp_train_npz))
            n_val = zero_feature_group(str(val_npz), feature_names, group_fn, str(tmp_val_npz))

            start = time.time()
            result = train_ablation(
                f"minus_{group_name}", str(tmp_data_dir), args.checkpoint_dir,
                args.epochs, args.patience, device
            )
            result["time_sec"] = time.time() - start
            result["n_zeroed"] = n_train
            all_results.append(result)

            delta = result["auroc"] - baseline["auroc"]
            logger.info(f"  -{group_name} AUROC: {result['auroc']:.3f} (Δ = {delta:+.3f})")

    finally:
        # Clean up temp directory
        shutil.rmtree(tmp_data_dir, ignore_errors=True)
        logger.info("Cleaned up temp ablation data directory")

    # Summary
    results_df = pd.DataFrame(all_results)
    output_dir = data_dir / "validation"
    output_dir.mkdir(exist_ok=True)
    results_df.to_csv(output_dir / "ablation_results.csv", index=False)

    logger.info(f"\n{'='*60}")
    logger.info("ABLATION RESULTS")
    logger.info(f"{'='*60}")
    baseline_auroc = results_df.loc[results_df["ablation"] == "Full", "auroc"].values[0]
    for _, row in results_df.iterrows():
        delta = row["auroc"] - baseline_auroc
        logger.info(f"  {row['ablation']:>20s}: AUROC={row['auroc']:.3f} (Δ = {delta:+.3f})")

    logger.info(f"\nResults saved to {output_dir / 'ablation_results.csv'}")


if __name__ == "__main__":
    main()
