#!/usr/bin/env python3
"""
MLP baseline for ecDNA formation prediction (Module 1).

Trains an MLP (112 -> 256 -> 128 -> 1) with BatchNorm, Dropout(0.3),
and class-weighted BCE on the same 112 features used by ecDNA-Former.
Runs 5-fold stratified CV alongside RF 5-fold CV for fair comparison,
then computes bootstrap AUROC differences.

Usage:
    python scripts/train_mlp_baseline.py --epochs 200 --patience 30
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Import reusable bootstrap function from compute_significance.py
# ---------------------------------------------------------------------------
from scripts.compute_significance import bootstrap_auroc_diff


# ---------------------------------------------------------------------------
# MLP Model
# ---------------------------------------------------------------------------
class MLPBaseline(nn.Module):
    """Simple MLP baseline: 112 -> 256 -> 128 -> 1."""

    def __init__(self, input_dim=112, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_full_features(data_dir: Path):
    """Load train + val features and labels, returning flat 112-dim matrix."""
    train = np.load(data_dir / "features" / "module1_features_train.npz", allow_pickle=True)
    val = np.load(data_dir / "features" / "module1_features_val.npz", allow_pickle=True)

    feature_names = list(train["feature_names"])
    n_features = len(feature_names)

    # Flat feature matrix: first n_features columns of sequence_features
    X_train = train["sequence_features"][:, :n_features]
    X_val = val["sequence_features"][:, :n_features]

    X = np.concatenate([X_train, X_val], axis=0)
    y = np.concatenate([train["labels"], val["labels"]], axis=0)

    logger.info(f"Loaded {X.shape[0]} samples, {X.shape[1]} features, "
                f"{int(y.sum())} ecDNA+")
    return X, y, feature_names


def train_mlp_fold(X_train, y_train, X_val, y_val, args, device):
    """Train one MLP fold and return validation predictions."""
    n_features = X_train.shape[1]

    # Class weights for BCE
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=device)

    model = MLPBaseline(input_dim=n_features, dropout=0.3).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # DataLoaders
    train_ds = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    best_auroc = 0.0
    best_probs = None
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(-1), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            logits = model(X_val_t).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()

        if len(np.unique(y_val)) >= 2:
            auroc = roc_auc_score(y_val, probs)
        else:
            auroc = 0.0

        if auroc > best_auroc:
            best_auroc = auroc
            best_probs = probs.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    return best_probs, best_auroc


def train_rf_fold(X_train, y_train, X_val, y_val):
    """Train one RF fold and return validation predictions."""
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    )
    rf.fit(X_train, y_train)
    probs = rf.predict_proba(X_val)[:, 1]

    if len(np.unique(y_val)) >= 2:
        auroc = roc_auc_score(y_val, probs)
    else:
        auroc = 0.0
    return probs, auroc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MLP baseline for ecDNA-Former")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    data_dir = Path(args.data_dir)

    logger.info(f"Running {args.n_folds}-fold CV: MLP + RF baselines on {device}")

    X, y, feature_names = load_full_features(data_dir)

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    folds = list(skf.split(X, y))

    mlp_results = []
    rf_results = []

    # Collect all out-of-fold predictions for bootstrap comparison
    mlp_oof_probs = np.zeros(len(y))
    rf_oof_probs = np.zeros(len(y))

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold_idx + 1}/{args.n_folds}")
        logger.info(f"{'='*60}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        logger.info(f"  Train: {len(y_train)} ({int(y_train.sum())} ecDNA+), "
                     f"Val: {len(y_val)} ({int(y_val.sum())} ecDNA+)")

        # MLP
        mlp_probs, mlp_auroc = train_mlp_fold(X_train, y_train, X_val, y_val, args, device)
        mlp_auprc = average_precision_score(y_val, mlp_probs) if len(np.unique(y_val)) >= 2 else 0.0
        mlp_f1 = f1_score(y_val, (mlp_probs > 0.5).astype(int), zero_division=0)
        mlp_oof_probs[val_idx] = mlp_probs

        mlp_results.append({
            "fold": fold_idx, "model": "MLP",
            "auroc": mlp_auroc, "auprc": mlp_auprc, "f1_score": mlp_f1,
        })
        logger.info(f"  MLP  AUROC={mlp_auroc:.3f}, AUPRC={mlp_auprc:.3f}")

        # RF
        rf_probs, rf_auroc = train_rf_fold(X_train, y_train, X_val, y_val)
        rf_auprc = average_precision_score(y_val, rf_probs) if len(np.unique(y_val)) >= 2 else 0.0
        rf_f1 = f1_score(y_val, (rf_probs > 0.5).astype(int), zero_division=0)
        rf_oof_probs[val_idx] = rf_probs

        rf_results.append({
            "fold": fold_idx, "model": "RF",
            "auroc": rf_auroc, "auprc": rf_auprc, "f1_score": rf_f1,
        })
        logger.info(f"  RF   AUROC={rf_auroc:.3f}, AUPRC={rf_auprc:.3f}")

    # ---------- Summary ----------
    all_results = mlp_results + rf_results
    results_df = pd.DataFrame(all_results)

    output_dir = data_dir / "validation"
    output_dir.mkdir(exist_ok=True)
    results_df.to_csv(output_dir / "mlp_crossval_results.csv", index=False)

    logger.info(f"\n{'='*60}")
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info(f"{'='*60}")
    for model_name in ["MLP", "RF"]:
        subset = results_df[results_df["model"] == model_name]
        for metric in ["auroc", "auprc", "f1_score"]:
            vals = subset[metric]
            logger.info(f"  {model_name:4s} {metric}: {vals.mean():.3f} +/- {vals.std():.3f}")

    # ---------- Bootstrap comparison (OOF predictions) ----------
    logger.info(f"\n{'='*60}")
    logger.info("BOOTSTRAP AUROC COMPARISONS (out-of-fold)")
    logger.info(f"{'='*60}")

    comparison = bootstrap_auroc_diff(y, mlp_oof_probs, rf_oof_probs, n_bootstrap=10000, seed=42)
    logger.info(f"  MLP vs RF: diff={comparison['observed_diff']:+.3f}, "
                f"95% CI=[{comparison['ci_2.5']:+.3f}, {comparison['ci_97.5']:+.3f}], "
                f"p={comparison['p_value']:.4f}")

    bootstrap_df = pd.DataFrame([{
        "comparison": "MLP_vs_RF",
        "observed_diff": comparison["observed_diff"],
        "mean_diff": comparison["mean_diff"],
        "ci_low": comparison["ci_2.5"],
        "ci_high": comparison["ci_97.5"],
        "p_value": comparison["p_value"],
    }])
    bootstrap_df.to_csv(output_dir / "mlp_bootstrap_comparison.csv", index=False)

    logger.info(f"\nResults saved to:")
    logger.info(f"  {output_dir / 'mlp_crossval_results.csv'}")
    logger.info(f"  {output_dir / 'mlp_bootstrap_comparison.csv'}")


if __name__ == "__main__":
    main()
