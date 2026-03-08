#!/usr/bin/env python3
"""
Physics constraint ablation study for CircularODE (Module 2).

Shows that physics constraints are not circular by ablating them and
testing generalization across treatments and time horizons.

Experiment A — Physics weight sweep: Train 4 configs (no-physics,
    weak=0.01, default=0.1, strong=1.0).
Experiment B — Cross-treatment holdout: Train on 3 of 4 treatments,
    test on the held-out treatment. 4 rounds.
Experiment C — Temporal extrapolation: Train on first 25 timepoints,
    predict last 25. Compare physics vs no-physics.

Usage:
    python scripts/circularode_physics_ablation.py --epochs 100 --patience 20
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
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.models.circular_ode.model import CircularODE
from src.models.circular_ode.treatment import TreatmentEncoder
from scripts.train_circularode_full import (
    FullTrajectoryDataset,
    collate_trajectories,
    train_epoch,
    validate,
    TREATMENT_ID_TO_CATEGORY,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def make_model(use_physics, latent_dim=8, hidden_dim=128):
    """Create a CircularODE model with or without physics constraints."""
    return CircularODE(
        latent_dim=latent_dim,
        treatment_dim=16,
        hidden_dim=hidden_dim,
        num_drift_layers=3,
        use_physics_constraints=use_physics,
        segregation_scale=0.5,
        min_diffusion=0.01,
    )


def train_and_evaluate(model, train_loader, val_loader, args, device, physics_weight):
    """Train model with early stopping and return best validation metrics."""
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_val_loss = float('inf')
    best_metrics = {}
    patience_counter = 0

    for epoch in range(args.epochs):
        train_epoch(model, train_loader, optimizer, device, physics_weight=physics_weight)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_metrics = val_metrics.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    return best_metrics


# ---------------------------------------------------------------------------
# Experiment A: Physics Weight Sweep
# ---------------------------------------------------------------------------
def run_physics_sweep(df, args, device):
    """Train 4 configs with different physics weights."""
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT A: Physics Weight Sweep")
    logger.info(f"{'='*60}")

    dataset = FullTrajectoryDataset(df)
    n_total = len(dataset)
    indices = np.random.RandomState(42).permutation(n_total)
    n_train = int(0.8 * n_total)

    train_ds = Subset(dataset, indices[:n_train])
    val_ds = Subset(dataset, indices[n_train:])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_trajectories)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_trajectories)

    configs = [
        ("no-physics", False, 0.0),
        ("weak",       True,  0.01),
        ("default",    True,  0.1),
        ("strong",     True,  1.0),
    ]

    results = []
    for name, use_physics, pw in configs:
        logger.info(f"\n--- Config: {name} (physics={use_physics}, weight={pw}) ---")
        model = make_model(use_physics).to(device)
        metrics = train_and_evaluate(model, train_loader, val_loader, args, device, pw)
        results.append({
            "config": name,
            "use_physics": use_physics,
            "physics_weight": pw,
            "val_mse": metrics["mse"],
            "val_mae": metrics["mae"],
            "val_correlation": metrics["correlation"],
        })
        logger.info(f"  MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, "
                    f"Corr={metrics['correlation']:.3f}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Experiment B: Cross-Treatment Holdout
# ---------------------------------------------------------------------------
def run_cross_treatment(df, args, device):
    """
    Train on 3 treatments, test on held-out treatment.

    Since each trajectory_id spans all 4 treatments (the dataset has
    trajectory_id × treatment_id rows), we assign each trajectory to its
    primary treatment (the treatment_id of the first row) and split at
    the trajectory level.
    """
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT B: Cross-Treatment Holdout")
    logger.info(f"{'='*60}")

    # Build full dataset (groups by trajectory_id; each group = 50 rows)
    dataset = FullTrajectoryDataset(df)

    # Determine primary treatment per trajectory
    traj_treatments = []
    for traj in dataset.trajectories:
        traj_treatments.append(traj['treatment_category'])
    traj_treatments = np.array(traj_treatments)
    unique_treatments = sorted(set(traj_treatments))

    logger.info(f"  Treatment distribution: {dict(zip(*np.unique(traj_treatments, return_counts=True)))}")

    results = []

    for holdout_cat in unique_treatments:
        logger.info(f"\n--- Holdout treatment category: {holdout_cat} ---")

        train_idx = [i for i, t in enumerate(traj_treatments) if t != holdout_cat]
        val_idx = [i for i, t in enumerate(traj_treatments) if t == holdout_cat]

        if len(val_idx) < 2:
            logger.info(f"  Skipping: only {len(val_idx)} trajectories")
            continue

        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_trajectories)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_trajectories)

        # Physics model
        model_phys = make_model(use_physics=True).to(device)
        metrics_phys = train_and_evaluate(model_phys, train_loader, val_loader,
                                          args, device, physics_weight=0.1)

        # No-physics model
        model_nophys = make_model(use_physics=False).to(device)
        metrics_nophys = train_and_evaluate(model_nophys, train_loader, val_loader,
                                            args, device, physics_weight=0.0)

        results.append({
            "holdout_treatment": int(holdout_cat),
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "physics_mse": metrics_phys["mse"],
            "physics_corr": metrics_phys["correlation"],
            "nophysics_mse": metrics_nophys["mse"],
            "nophysics_corr": metrics_nophys["correlation"],
        })
        logger.info(f"  Physics:    MSE={metrics_phys['mse']:.4f}, Corr={metrics_phys['correlation']:.3f}")
        logger.info(f"  No-physics: MSE={metrics_nophys['mse']:.4f}, Corr={metrics_nophys['correlation']:.3f}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Experiment C: Temporal Extrapolation
# ---------------------------------------------------------------------------
class TemporalSubsetDataset(torch.utils.data.Dataset):
    """Wraps FullTrajectoryDataset but truncates or shifts time windows."""

    def __init__(self, base_dataset, start_idx, end_idx):
        self.base = base_dataset
        self.start = start_idx
        self.end = end_idx

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        return {
            'initial_state': item['initial_state'],
            'time_points': item['time_points'][self.start:self.end],
            'copy_numbers': item['copy_numbers'][self.start:self.end],
            'raw_copy_numbers': item['raw_copy_numbers'][self.start:self.end],
            'treatment_category': item['treatment_category'],
        }


def run_temporal_extrapolation(df, args, device):
    """Train on first 25 timepoints, predict last 25."""
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT C: Temporal Extrapolation")
    logger.info(f"{'='*60}")

    full_dataset = FullTrajectoryDataset(df)
    n_total = len(full_dataset)
    indices = np.random.RandomState(42).permutation(n_total)
    n_train = int(0.8 * n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    # Determine midpoint
    sample_item = full_dataset[0]
    n_timepoints = len(sample_item['time_points'])
    mid = n_timepoints // 2
    logger.info(f"  Total timepoints: {n_timepoints}, training on [0:{mid}], testing on [{mid}:{n_timepoints}]")

    results = []

    for name, use_physics, pw in [("physics", True, 0.1), ("no-physics", False, 0.0)]:
        logger.info(f"\n--- {name} ---")

        # Train on first half of time
        train_ds = TemporalSubsetDataset(Subset(full_dataset, train_idx), 0, mid)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_trajectories)

        # Validate on first half (to select best model)
        val_first_ds = TemporalSubsetDataset(Subset(full_dataset, val_idx), 0, mid)
        val_first_loader = DataLoader(val_first_ds, batch_size=args.batch_size, shuffle=False,
                                      collate_fn=collate_trajectories)

        model = make_model(use_physics).to(device)
        _ = train_and_evaluate(model, train_loader, val_first_loader, args, device, pw)

        # Test on second half (extrapolation)
        val_second_ds = TemporalSubsetDataset(Subset(full_dataset, val_idx), mid, n_timepoints)
        val_second_loader = DataLoader(val_second_ds, batch_size=args.batch_size, shuffle=False,
                                       collate_fn=collate_trajectories)

        # Also evaluate on first half (interpolation) for comparison
        interp_metrics = validate(model, val_first_loader, device)
        extrap_metrics = validate(model, val_second_loader, device)

        results.append({
            "config": name,
            "interpolation_mse": interp_metrics["mse"],
            "interpolation_corr": interp_metrics["correlation"],
            "extrapolation_mse": extrap_metrics["mse"],
            "extrapolation_corr": extrap_metrics["correlation"],
        })
        logger.info(f"  Interpolation:  MSE={interp_metrics['mse']:.4f}, Corr={interp_metrics['correlation']:.3f}")
        logger.info(f"  Extrapolation:  MSE={extrap_metrics['mse']:.4f}, Corr={extrap_metrics['correlation']:.3f}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CircularODE physics ablation study")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=20)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load trajectory data
    data_path = Path("data/ecdna_trajectories/parsed_trajectories.csv")
    if not data_path.exists():
        logger.error(f"Trajectory data not found at {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {df['trajectory_id'].nunique()} trajectories, {len(df)} rows")

    output_dir = Path("data/validation")
    output_dir.mkdir(exist_ok=True)

    # Experiment A
    sweep_df = run_physics_sweep(df, args, device)
    sweep_df.to_csv(output_dir / "circularode_physics_ablation.csv", index=False)

    # Experiment B
    cross_df = run_cross_treatment(df, args, device)
    cross_df.to_csv(output_dir / "circularode_cross_treatment.csv", index=False)

    # Experiment C
    extrap_df = run_temporal_extrapolation(df, args, device)
    extrap_df.to_csv(output_dir / "circularode_temporal_extrapolation.csv", index=False)

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"\nExperiment A (physics sweep):")
    for _, row in sweep_df.iterrows():
        logger.info(f"  {row['config']:>12s}: MSE={row['val_mse']:.4f}, Corr={row['val_correlation']:.3f}")

    logger.info(f"\nExperiment B (cross-treatment):")
    logger.info(f"  Physics    mean corr: {cross_df['physics_corr'].mean():.3f}")
    logger.info(f"  No-physics mean corr: {cross_df['nophysics_corr'].mean():.3f}")

    logger.info(f"\nExperiment C (temporal extrapolation):")
    for _, row in extrap_df.iterrows():
        logger.info(f"  {row['config']:>12s}: interp_corr={row['interpolation_corr']:.3f}, "
                    f"extrap_corr={row['extrapolation_corr']:.3f}")

    logger.info(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
