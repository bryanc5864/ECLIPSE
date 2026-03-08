#!/usr/bin/env python3
"""
Train CircularODE model for ecDNA copy number dynamics prediction.

Uses synthetic trajectory data to learn:
1. Copy number dynamics over time
2. Treatment response modeling
3. Resistance prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import argparse
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_ecsimulator_data(data_dir: Path) -> pd.DataFrame:
    """
    Parse ecSimulator output files to extract trajectory data.

    Returns DataFrame with trajectory_id, time, copy_number, structure_info
    """
    logger.info(f"Parsing ecSimulator data from {data_dir}...")

    all_data = []
    traj_files = sorted(data_dir.glob("traj_*_amplicon1_cycles.txt"))

    for traj_file in tqdm(traj_files, desc="Parsing trajectories"):
        traj_id = int(traj_file.stem.split('_')[1])

        # Parse cycles file for structure
        with open(traj_file) as f:
            lines = f.readlines()

        # Extract interval info
        interval_line = [l for l in lines if l.startswith("Interval")]
        if interval_line:
            parts = interval_line[0].strip().split('\t')
            chrom = parts[2]
            start = int(parts[3])
            end = int(parts[4])
            size = end - start

            # Count segments (proxy for complexity)
            n_segments = len([l for l in lines if l.startswith("Segment")])

            # Estimate copy number from fasta file size
            fasta_file = traj_file.parent / f"traj_{traj_id:04d}_amplicon1.fasta"
            if fasta_file.exists():
                fasta_size = fasta_file.stat().st_size
                # Rough estimate: copy number ~ fasta_size / interval_size
                estimated_cn = max(1, fasta_size / (size + 1))
            else:
                estimated_cn = n_segments  # Fallback

            # Create synthetic time series (evolution simulation)
            # Simulate copy number changes over generations
            n_timepoints = 50
            times = np.linspace(0, 100, n_timepoints)

            # Initial CN based on structure
            initial_cn = estimated_cn * np.random.uniform(0.8, 1.2)

            # Simulate trajectory with biological noise
            cn_trajectory = simulate_cn_trajectory(initial_cn, n_timepoints)

            for t_idx, (t, cn) in enumerate(zip(times, cn_trajectory)):
                all_data.append({
                    'trajectory_id': traj_id,
                    'time': t,
                    'copy_number': cn,
                    'n_segments': n_segments,
                    'size': size,
                    'chrom': chrom,
                    'treatment_id': np.random.randint(0, 4),  # Random treatment
                })

    df = pd.DataFrame(all_data)
    logger.info(f"Parsed {len(traj_files)} trajectories, {len(df)} total samples")
    return df


def simulate_cn_trajectory(initial_cn: float, n_steps: int) -> np.ndarray:
    """
    Simulate copy number trajectory with biological dynamics.

    Incorporates:
    - Random segregation noise (binomial)
    - Fitness-based selection
    - Treatment effects (random assignment)
    """
    cn = initial_cn
    trajectory = []

    # Random treatment assignment
    treatment = np.random.choice(['none', 'targeted', 'chemo', 'maintenance'])
    treatment_start = np.random.randint(10, 30)
    treatment_strength = np.random.uniform(0.3, 0.7)

    for i in range(n_steps):
        trajectory.append(cn)

        # Fitness (favors moderate CN)
        fitness = 1.0 + 0.02 * cn / (1 + cn / 50) - 0.001 * cn

        # Treatment effect
        t_normalized = i / n_steps * 100
        if treatment != 'none' and t_normalized > treatment_start:
            if treatment == 'targeted':
                # Higher CN = more sensitive
                death_prob = treatment_strength * min(1.0, cn / 50) * 0.1
            elif treatment == 'chemo':
                death_prob = treatment_strength * 0.08
            else:  # maintenance
                death_prob = treatment_strength * 0.03

            fitness *= (1 - death_prob)

        # Selection and segregation
        if np.random.random() < fitness:
            # Cell divides with binomial segregation
            if cn > 0:
                daughter_cn = np.random.binomial(int(2 * cn), 0.5)
                # Selection between daughters
                cn = daughter_cn if np.random.random() < 0.5 else 2 * cn - daughter_cn
                cn = max(0, cn)

        # Add noise
        cn = max(0, cn + np.random.normal(0, np.sqrt(max(1, cn)) * 0.3))

    return np.array(trajectory)


class TrajectoryDataset(Dataset):
    """Dataset for ecDNA trajectories."""

    def __init__(self, df: pd.DataFrame, seq_len: int = 20):
        """
        Args:
            df: DataFrame with trajectory data
            seq_len: Sequence length for prediction
        """
        self.seq_len = seq_len
        self.sequences = []
        self.targets = []
        self.treatments = []

        # Group by trajectory
        for traj_id, group in df.groupby('trajectory_id'):
            group = group.sort_values('time')
            cns = group['copy_number'].values
            times = group['time'].values
            treatment = group['treatment_id'].iloc[0]

            # Create sequences
            for i in range(len(cns) - seq_len):
                self.sequences.append(np.column_stack([
                    cns[i:i+seq_len],
                    times[i:i+seq_len],
                ]))
                self.targets.append(cns[i+seq_len])
                self.treatments.append(treatment)

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        self.treatments = np.array(self.treatments, dtype=np.int64)

        # Normalize
        self.cn_mean = np.mean(self.sequences[:, :, 0])
        self.cn_std = np.std(self.sequences[:, :, 0]) + 1e-6
        self.sequences[:, :, 0] = (self.sequences[:, :, 0] - self.cn_mean) / self.cn_std
        self.targets = (self.targets - self.cn_mean) / self.cn_std

        logger.info(f"Created dataset with {len(self)} sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequence': torch.tensor(self.sequences[idx]),
            'target': torch.tensor(self.targets[idx]),
            'treatment': torch.tensor(self.treatments[idx]),
        }


class SimpleCircularODE(nn.Module):
    """
    Simplified CircularODE for training.

    Predicts next copy number given history and treatment.
    """

    def __init__(
        self,
        input_dim: int = 2,  # CN + time
        hidden_dim: int = 128,
        num_treatments: int = 4,
        treatment_dim: int = 16,
    ):
        super().__init__()

        # Treatment embedding
        self.treatment_emb = nn.Embedding(num_treatments, treatment_dim)

        # Sequence encoder (GRU)
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # Dynamics head
        self.dynamics = nn.Sequential(
            nn.Linear(hidden_dim + treatment_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Resistance prediction head
        self.resistance_head = nn.Sequential(
            nn.Linear(hidden_dim + treatment_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, sequence, treatment):
        """
        Forward pass.

        Args:
            sequence: [batch, seq_len, 2] (CN, time)
            treatment: [batch] treatment IDs

        Returns:
            predictions dict
        """
        # Encode sequence
        _, hidden = self.encoder(sequence)
        hidden = hidden[-1]  # Last layer hidden state

        # Get treatment embedding
        treat_emb = self.treatment_emb(treatment)

        # Combine
        combined = torch.cat([hidden, treat_emb], dim=-1)

        # Predict next CN
        cn_pred = self.dynamics(combined).squeeze(-1)

        # Predict resistance probability
        resistance_prob = self.resistance_head(combined).squeeze(-1)

        return {
            'cn_pred': cn_pred,
            'resistance_prob': resistance_prob,
            'hidden': hidden,
        }


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_cn_loss = 0

    for batch in dataloader:
        sequence = batch['sequence'].to(device)
        target = batch['target'].to(device)
        treatment = batch['treatment'].to(device)

        optimizer.zero_grad()

        outputs = model(sequence, treatment)

        # CN prediction loss
        cn_loss = nn.functional.mse_loss(outputs['cn_pred'], target)

        # Resistance regularization (encourage diversity)
        resistance_entropy = -(
            outputs['resistance_prob'] * torch.log(outputs['resistance_prob'] + 1e-8) +
            (1 - outputs['resistance_prob']) * torch.log(1 - outputs['resistance_prob'] + 1e-8)
        ).mean()

        loss = cn_loss - 0.01 * resistance_entropy  # Encourage uncertainty

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_cn_loss += cn_loss.item()

    n_batches = len(dataloader)
    return {
        'total_loss': total_loss / n_batches,
        'cn_loss': total_cn_loss / n_batches,
    }


def validate(model, dataloader, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            sequence = batch['sequence'].to(device)
            target = batch['target'].to(device)
            treatment = batch['treatment'].to(device)

            outputs = model(sequence, treatment)
            loss = nn.functional.mse_loss(outputs['cn_pred'], target)

            total_loss += loss.item()
            all_preds.extend(outputs['cn_pred'].cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Compute metrics
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    corr = np.corrcoef(preds, targets)[0, 1]

    return {
        'loss': total_loss / len(dataloader),
        'mse': mse,
        'mae': mae,
        'correlation': corr,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--seq_len', type=int, default=20)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    data_dir = Path("data/ecdna_trajectories")

    # Parse or generate data
    cache_file = data_dir / "parsed_trajectories.csv"
    if cache_file.exists():
        logger.info("Loading cached trajectory data...")
        df = pd.read_csv(cache_file)
    else:
        df = parse_ecsimulator_data(data_dir)
        df.to_csv(cache_file, index=False)

    # Create dataset
    dataset = TrajectoryDataset(df, seq_len=args.seq_len)

    # Split
    n_samples = len(dataset)
    indices = np.random.permutation(n_samples)
    train_idx = indices[:int(0.8 * n_samples)]
    val_idx = indices[int(0.8 * n_samples):]

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model
    model = SimpleCircularODE(
        input_dim=2,
        hidden_dim=args.hidden_dim,
        num_treatments=4,
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Training
    output_dir = Path("checkpoints/circularode")
    output_dir.mkdir(exist_ok=True, parents=True)

    best_val_loss = float('inf')
    history = []

    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        history.append({
            'epoch': epoch,
            'train_loss': train_metrics['cn_loss'],
            'val_loss': val_metrics['loss'],
            'val_mse': val_metrics['mse'],
            'val_mae': val_metrics['mae'],
            'val_corr': val_metrics['correlation'],
        })

        logger.info(
            f"Epoch {epoch}: train_loss={train_metrics['cn_loss']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"corr={val_metrics['correlation']:.3f}"
        )

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss,
                'cn_mean': dataset.cn_mean,
                'cn_std': dataset.cn_std,
            }, output_dir / "best_model.pt")
            logger.info(f"  Saved best model (val_loss={best_val_loss:.4f})")

    # Save history
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    # Final evaluation
    logger.info("\n=== FINAL EVALUATION ===")
    checkpoint = torch.load(output_dir / "best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    final_metrics = validate(model, val_loader, device)

    logger.info(f"MSE: {final_metrics['mse']:.4f}")
    logger.info(f"MAE: {final_metrics['mae']:.4f}")
    logger.info(f"Correlation: {final_metrics['correlation']:.4f}")

    # Save final results
    results = {
        'best_epoch': checkpoint['epoch'],
        'val_mse': final_metrics['mse'],
        'val_mae': final_metrics['mae'],
        'val_correlation': final_metrics['correlation'],
        'cn_mean': float(dataset.cn_mean),
        'cn_std': float(dataset.cn_std),
    }
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
