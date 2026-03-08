#!/usr/bin/env python3
"""
Retrain ecDNA-Former without dosage features (Module 1).

Feature ablation showed that removing the 9 dosage_* features improves
AUROC from 0.787 to 0.811. This script retrains the full model without
dosage features and runs a bootstrap comparison.

Usage:
    python scripts/retrain_no_dosage.py --epochs 200 --patience 30
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.run_ablation import ABLATION_GROUPS, zero_feature_group
from scripts.compute_significance import bootstrap_auroc_diff

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Retrain ecDNA-Former without dosage features")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/no_dosage")
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

    dosage_fn = ABLATION_GROUPS["Dosage"]
    n_dosage = sum(1 for name in feature_names if dosage_fn(name))
    logger.info(f"Total features: {len(feature_names)}, Dosage features to zero: {n_dosage}")

    # Create temp data directory with dosage features zeroed
    import tempfile
    tmp_data_dir = Path(tempfile.mkdtemp(prefix="eclipse_nodosage_"))
    tmp_features_dir = tmp_data_dir / "features"
    tmp_features_dir.mkdir(parents=True, exist_ok=True)

    # Symlink required subdirectories
    for subdir in ["ecdna_labels", "cytocell_db", "depmap", "hic", "supplementary"]:
        src = data_dir / subdir
        if src.exists():
            os.symlink(src.resolve(), tmp_data_dir / subdir)

    train_npz = features_dir / "module1_features_train.npz"
    val_npz = features_dir / "module1_features_val.npz"
    tmp_train_npz = tmp_features_dir / "module1_features_train.npz"
    tmp_val_npz = tmp_features_dir / "module1_features_val.npz"

    logger.info("Zeroing dosage features in train and val NPZ files...")
    zero_feature_group(str(train_npz), feature_names, dosage_fn, str(tmp_train_npz))
    zero_feature_group(str(val_npz), feature_names, dosage_fn, str(tmp_val_npz))

    # Train model
    from src.data import ECDNADataset, create_dataloader
    from src.models import ECDNAFormer
    from src.training import ECDNAFormerTrainer

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_dataset = ECDNADataset.from_data_dir(data_dir=str(tmp_data_dir), split="train")
        val_dataset = ECDNADataset.from_data_dir(data_dir=str(tmp_data_dir), split="val")

        train_loader = create_dataloader(train_dataset, batch_size=32, shuffle=True)
        val_loader = create_dataloader(val_dataset, batch_size=32, shuffle=False)

        logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

        model = ECDNAFormer()
        trainer = ECDNAFormerTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            checkpoint_dir=str(ckpt_dir),
            use_wandb=False,
        )

        logger.info(f"\n{'='*60}")
        logger.info("Training ecDNA-Former (no dosage features)")
        logger.info(f"  Epochs: {args.epochs}, Patience: {args.patience}")
        logger.info(f"{'='*60}\n")

        trainer.train(num_epochs=args.epochs, early_stopping_patience=args.patience)

        # Extract best metrics
        log_dir = ckpt_dir / "logs"
        val_logs = sorted(log_dir.glob("validation_log_*.csv"))
        if val_logs:
            val_df = pd.read_csv(val_logs[-1])
            best_row = val_df.loc[val_df["auroc"].idxmax()]
            no_dosage_auroc = best_row["auroc"]

            results = {
                "config": "no_dosage",
                "best_epoch": int(best_row["epoch"]),
                "auroc": best_row["auroc"],
                "auprc": best_row["auprc"],
                "f1_score": best_row["f1_score"],
                "mcc": best_row["mcc"],
                "balanced_accuracy": best_row["balanced_accuracy"],
                "recall": best_row["recall"],
                "precision": best_row["precision"],
            }
            logger.info(f"\nNo-dosage AUROC: {no_dosage_auroc:.3f}")
        else:
            results = {"config": "no_dosage", "auroc": 0.0}
            no_dosage_auroc = 0.0

        # Save results
        output_dir = data_dir / "validation"
        output_dir.mkdir(exist_ok=True)
        pd.DataFrame([results]).to_csv(output_dir / "no_dosage_results.csv", index=False)

        # Bootstrap comparison vs full model predictions
        logger.info(f"\n{'='*60}")
        logger.info("Bootstrap comparison: no-dosage vs full model")
        logger.info(f"{'='*60}")

        # Get predictions from no-dosage model
        model.eval()
        val_data = np.load(str(tmp_val_npz), allow_pickle=True)
        y_val = val_data["labels"]

        with torch.no_grad():
            batch = {
                "sequence_features": torch.FloatTensor(val_data["sequence_features"]).to(device),
                "topology_features": torch.FloatTensor(val_data["topology_features"]).to(device),
                "fragile_site_features": torch.FloatTensor(val_data["fragile_site_features"]).to(device),
                "copy_number_features": torch.FloatTensor(val_data["copy_number_features"]).to(device),
            }
            outputs = model(**batch)
            no_dosage_probs = outputs["formation_probability"].cpu().numpy().flatten()

        # Get predictions from full model (if checkpoint exists)
        full_ckpt = Path("checkpoints/best.pt")
        if full_ckpt.exists():
            full_model = ECDNAFormer()
            checkpoint = torch.load(full_ckpt, map_location=device)
            if "model_state_dict" in checkpoint:
                full_model.load_state_dict(checkpoint["model_state_dict"])
            else:
                full_model.load_state_dict(checkpoint)
            full_model.to(device)
            full_model.eval()

            # Use original (non-zeroed) val data for full model
            orig_val = np.load(str(val_npz), allow_pickle=True)
            with torch.no_grad():
                batch_full = {
                    "sequence_features": torch.FloatTensor(orig_val["sequence_features"]).to(device),
                    "topology_features": torch.FloatTensor(orig_val["topology_features"]).to(device),
                    "fragile_site_features": torch.FloatTensor(orig_val["fragile_site_features"]).to(device),
                    "copy_number_features": torch.FloatTensor(orig_val["copy_number_features"]).to(device),
                }
                full_outputs = full_model(**batch_full)
                full_probs = full_outputs["formation_probability"].cpu().numpy().flatten()

            comparison = bootstrap_auroc_diff(y_val, no_dosage_probs, full_probs,
                                              n_bootstrap=10000, seed=42)
            logger.info(f"  No-dosage vs Full: diff={comparison['observed_diff']:+.3f}, "
                        f"95% CI=[{comparison['ci_2.5']:+.3f}, {comparison['ci_97.5']:+.3f}], "
                        f"p={comparison['p_value']:.4f}")

            bootstrap_df = pd.DataFrame([{
                "comparison": "no_dosage_vs_full",
                "observed_diff": comparison["observed_diff"],
                "ci_low": comparison["ci_2.5"],
                "ci_high": comparison["ci_97.5"],
                "p_value": comparison["p_value"],
            }])
            bootstrap_df.to_csv(output_dir / "no_dosage_bootstrap.csv", index=False)
        else:
            logger.warning("Full model checkpoint not found; skipping bootstrap comparison")

    finally:
        shutil.rmtree(tmp_data_dir, ignore_errors=True)
        logger.info("Cleaned up temp data directory")

    logger.info(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
