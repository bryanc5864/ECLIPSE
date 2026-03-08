#!/usr/bin/env python3
"""
Train the full CircularODE model for ecDNA copy number dynamics.

Unlike train_circularode.py (which uses a simplified GRU-based model),
this trains the actual Physics-Informed Neural SDE from
src/models/circular_ode/model.py, which includes:
  - DriftNetwork with physics-informed fitness landscape
  - DiffusionNetwork with segregation-scaled noise
  - TreatmentEncoder with category/dose/duration encoding
  - SegregationPhysics constraints (binomial variance)
  - Euler-Maruyama SDE solver (fallback without torchsde)

Data: Reuses data/ecdna_trajectories/parsed_trajectories.csv
      (500 trajectories, 50 timepoints each)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import argparse
import json

from src.models.circular_ode.model import CircularODE
from src.models.circular_ode.treatment import TreatmentEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Treatment ID mapping: trajectory data uses int IDs (0-3),
# TreatmentEncoder expects category indices from TREATMENT_CATEGORIES
TREATMENT_ID_TO_CATEGORY = {
    0: TreatmentEncoder.TREATMENT_CATEGORIES["targeted"],     # 0
    1: TreatmentEncoder.TREATMENT_CATEGORIES["chemo"],        # 1
    2: TreatmentEncoder.TREATMENT_CATEGORIES["none"],         # 5
    3: TreatmentEncoder.TREATMENT_CATEGORIES["ecdna_specific"],  # 3
}


class FullTrajectoryDataset(Dataset):
    """
    Dataset that provides full trajectories for the SDE model.

    Data is normalized for numerical stability:
      - Time is scaled to [0, 1]
      - Copy numbers are transformed via log1p

    Each sample is one trajectory:
      - initial_state: [3] (log1p(CN), 0.0, 1.0)
      - time_points: [num_times] scaled to [0, 1]
      - copy_numbers: [num_times] log1p-transformed
      - treatment_category: int
    """

    def __init__(self, df: pd.DataFrame):
        self.trajectories = []

        # Compute normalization constants
        all_times = df['time'].values
        self.time_max = float(all_times.max())

        for traj_id, group in df.groupby('trajectory_id'):
            group = group.sort_values('time')
            raw_cns = group['copy_number'].values.astype(np.float32)
            raw_times = group['time'].values.astype(np.float32)
            treatment_id = int(group['treatment_id'].iloc[0])

            # Normalize time to [0, 1]
            times = raw_times / self.time_max

            # Transform CN via log1p for stability
            cns = np.log1p(np.maximum(raw_cns, 0.0)).astype(np.float32)

            # Initial state: [log1p(CN), 0.0, 1.0]
            initial_state = np.array([cns[0], 0.0, 1.0], dtype=np.float32)

            # Map treatment int to TreatmentEncoder category
            category = TREATMENT_ID_TO_CATEGORY.get(treatment_id, 5)

            self.trajectories.append({
                'initial_state': initial_state,
                'time_points': times,
                'copy_numbers': cns,
                'raw_copy_numbers': raw_cns,
                'treatment_category': category,
            })

        logger.info(f"Created dataset with {len(self.trajectories)} trajectories")
        logger.info(f"  Time range: [0, {self.time_max}] -> normalized to [0, 1]")
        logger.info(f"  CN transform: log1p")

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        return {
            'initial_state': torch.tensor(traj['initial_state']),
            'time_points': torch.tensor(traj['time_points']),
            'copy_numbers': torch.tensor(traj['copy_numbers']),
            'raw_copy_numbers': torch.tensor(traj['raw_copy_numbers']),
            'treatment_category': torch.tensor(traj['treatment_category'], dtype=torch.long),
        }


def collate_trajectories(batch):
    """Custom collate: stack trajectories (all same length)."""
    return {
        'initial_state': torch.stack([b['initial_state'] for b in batch]),
        'time_points': batch[0]['time_points'],  # shared across batch
        'copy_numbers': torch.stack([b['copy_numbers'] for b in batch]),
        'raw_copy_numbers': torch.stack([b['raw_copy_numbers'] for b in batch]),
        'treatment_category': torch.stack([b['treatment_category'] for b in batch]),
    }


def train_epoch(model, dataloader, optimizer, device, physics_weight=0.1):
    """Train for one epoch using the full CircularODE."""
    model.train()
    total_loss = 0
    total_data_loss = 0
    total_physics_loss = 0
    n_batches = 0

    for batch in dataloader:
        initial_state = batch['initial_state'].to(device)
        time_points = batch['time_points'].to(device)
        copy_numbers = batch['copy_numbers'].to(device)  # log1p-transformed
        treatment_cats = batch['treatment_category'].to(device)

        optimizer.zero_grad()

        treatment_info = {'categories': treatment_cats}

        # Forward: SDE solver produces trajectory in log1p space
        predictions = model(
            initial_state=initial_state,
            time_points=time_points,
            treatment_info=treatment_info,
            n_samples=1,
            return_trajectories=True,
        )

        # Loss on log1p-transformed CN (model.get_loss uses log1p internally,
        # but since data is already log1p'd, use direct MSE here)
        pred_cn = predictions['copy_number_trajectory']
        data_loss = torch.nn.functional.mse_loss(pred_cn, copy_numbers)

        # Physics constraint: segregation variance
        physics_loss = torch.tensor(0.0, device=device)
        if model.use_physics_constraints:
            traj_var = pred_cn.var(dim=1)
            mean_cn = pred_cn.mean(dim=1)
            expected_var = model.segregation_physics.expected_variance(mean_cn)
            physics_loss = physics_weight * torch.nn.functional.mse_loss(traj_var, expected_var)

        # Non-negativity (soft, on raw scale — CN decoder uses Softplus so this is mild)
        nonneg_loss = 0.01 * torch.nn.functional.relu(-pred_cn).mean()

        loss = data_loss + physics_loss + nonneg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_data_loss += data_loss.item()
        total_physics_loss += physics_loss.item()
        n_batches += 1

    return {
        'total_loss': total_loss / max(n_batches, 1),
        'data_loss': total_data_loss / max(n_batches, 1),
        'physics_loss': total_physics_loss / max(n_batches, 1),
    }


def validate(model, dataloader, device):
    """Validate the full CircularODE model."""
    model.eval()
    all_pred_cn = []
    all_true_cn = []
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            initial_state = batch['initial_state'].to(device)
            time_points = batch['time_points'].to(device)
            copy_numbers = batch['copy_numbers'].to(device)  # log1p space
            raw_copy_numbers = batch['raw_copy_numbers'].to(device)
            treatment_cats = batch['treatment_category'].to(device)

            treatment_info = {'categories': treatment_cats}

            predictions = model(
                initial_state=initial_state,
                time_points=time_points,
                treatment_info=treatment_info,
                n_samples=1,
                return_trajectories=True,
            )

            pred_cn = predictions['copy_number_trajectory']
            loss = torch.nn.functional.mse_loss(pred_cn, copy_numbers)
            total_loss += loss.item()

            # Convert back to raw scale for metrics: expm1(log1p_pred)
            pred_raw = torch.expm1(pred_cn.clamp(max=20)).cpu().numpy()
            true_raw = raw_copy_numbers.cpu().numpy()
            all_pred_cn.append(pred_raw.flatten())
            all_true_cn.append(true_raw.flatten())
            n_batches += 1

    pred = np.concatenate(all_pred_cn)
    true = np.concatenate(all_true_cn)

    mse = float(np.mean((pred - true) ** 2))
    mae = float(np.mean(np.abs(pred - true)))

    if np.std(pred) > 1e-8 and np.std(true) > 1e-8:
        corr = float(np.corrcoef(pred, true)[0, 1])
    else:
        corr = 0.0

    return {
        'loss': total_loss / max(n_batches, 1),
        'mse': mse,
        'mae': mae,
        'correlation': corr if not np.isnan(corr) else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Train full CircularODE model")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--latent_dim', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--physics_weight', type=float, default=0.1,
                        help="Weight for physics constraint loss")
    parser.add_argument('--patience', type=int, default=20,
                        help="Early stopping patience")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # ── Load Data ────────────────────────────────────────────────────────
    data_dir = Path("data/ecdna_trajectories")
    cache_file = data_dir / "parsed_trajectories.csv"

    if not cache_file.exists():
        logger.error(f"Trajectory data not found at {cache_file}")
        logger.error("Run scripts/train_circularode.py first to generate trajectory data")
        sys.exit(1)

    logger.info(f"Loading trajectory data from {cache_file}...")
    df = pd.read_csv(cache_file)
    n_trajectories = df['trajectory_id'].nunique()
    logger.info(f"Loaded {n_trajectories} trajectories, {len(df)} total rows")

    # ── Create Dataset ───────────────────────────────────────────────────
    dataset = FullTrajectoryDataset(df)

    # Train/val split (80/20 by trajectory)
    n_total = len(dataset)
    indices = np.random.RandomState(42).permutation(n_total)
    n_train = int(0.8 * n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_trajectories,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_trajectories,
    )
    logger.info(f"Train: {len(train_dataset)} trajectories, Val: {len(val_dataset)} trajectories")

    # ── Create Model ─────────────────────────────────────────────────────
    model_config = {
        'latent_dim': args.latent_dim,
        'treatment_dim': 16,
        'hidden_dim': args.hidden_dim,
        'num_drift_layers': 3,
        'use_physics_constraints': True,
        'segregation_scale': 0.5,
        'min_diffusion': 0.01,
    }

    model = CircularODE(**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Full CircularODE model: {n_params:,} parameters")

    # ── Optimizer & Scheduler ────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # ── Training Loop ────────────────────────────────────────────────────
    output_dir = Path("checkpoints/circularode_full")
    output_dir.mkdir(exist_ok=True, parents=True)

    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    logger.info(f"\n{'='*60}")
    logger.info(f"Training Full CircularODE")
    logger.info(f"  Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    logger.info(f"  Physics weight: {args.physics_weight}")
    logger.info(f"  Model: latent_dim={args.latent_dim}, hidden_dim={args.hidden_dim}")
    logger.info(f"{'='*60}\n")

    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            physics_weight=args.physics_weight,
        )
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        history.append({
            'epoch': epoch,
            'train_loss': train_metrics['total_loss'],
            'train_data_loss': train_metrics['data_loss'],
            'train_physics_loss': train_metrics['physics_loss'],
            'val_loss': val_metrics['loss'],
            'val_mse': val_metrics['mse'],
            'val_mae': val_metrics['mae'],
            'val_corr': val_metrics['correlation'],
        })

        logger.info(
            f"Epoch {epoch:3d}: "
            f"train={train_metrics['total_loss']:.4f} "
            f"(data={train_metrics['data_loss']:.4f}, physics={train_metrics['physics_loss']:.4f}) | "
            f"val={val_metrics['loss']:.4f}, MSE={val_metrics['mse']:.4f}, "
            f"corr={val_metrics['correlation']:.3f}"
        )

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0

            torch.save({
                'model_state_dict': model.state_dict(),
                'config': model_config,
                'epoch': epoch,
                'val_loss': best_val_loss,
                'val_mse': val_metrics['mse'],
                'val_mae': val_metrics['mae'],
                'val_correlation': val_metrics['correlation'],
                'time_max': dataset.time_max,
            }, output_dir / "best_model.pt")
            logger.info(f"  -> Saved best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # ── Save Training History ────────────────────────────────────────────
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    # ── Final Evaluation ─────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("FINAL EVALUATION (best checkpoint)")
    logger.info(f"{'='*60}")

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    final_metrics = validate(model, val_loader, device)

    logger.info(f"  MSE:         {final_metrics['mse']:.6f}")
    logger.info(f"  MAE:         {final_metrics['mae']:.6f}")
    logger.info(f"  Correlation: {final_metrics['correlation']:.4f}")
    logger.info(f"  Best epoch:  {checkpoint['epoch']}")

    # Compare with simplified model
    logger.info("\nComparison with SimpleCircularODE:")
    logger.info(f"  SimpleCircularODE: MSE=0.0141, MAE=0.0685, Corr=0.993")
    logger.info(f"  Full CircularODE:  MSE={final_metrics['mse']:.4f}, "
                f"MAE={final_metrics['mae']:.4f}, Corr={final_metrics['correlation']:.3f}")

    # Save results
    results = {
        'model': 'CircularODE (full)',
        'best_epoch': int(checkpoint['epoch']),
        'val_mse': final_metrics['mse'],
        'val_mae': final_metrics['mae'],
        'val_correlation': final_metrics['correlation'],
        'config': model_config,
        'n_parameters': n_params,
        'simplified_comparison': {
            'mse': 0.0141,
            'mae': 0.0685,
            'correlation': 0.993,
        },
    }
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
