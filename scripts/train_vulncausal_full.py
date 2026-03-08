#!/usr/bin/env python3
"""
Train the full VulnCausal model for ecDNA-specific vulnerability discovery.

Unlike train_vulncausal.py (which uses SimplifiedVulnCausal with a linear
interaction term), this trains the actual causal inference model from
src/models/vuln_causal/model.py, which includes:
  - CausalRepresentationLearner (VAE with 6 disentangled factors, 96-dim latent)
  - InvariantRiskMinimization (IRM penalty across lineage environments)
  - NeuralCausalDiscovery (NOTEARS DAG learning over 86 variables)
  - DoCalculusNetwork (causal effect estimation)
  - VulnerabilityScoringNetwork (gene ranking)

Data: DepMap CRISPR (1062 samples x 17453 genes), expression (x 19193 genes),
      CytoCellDB ecDNA labels.

Post-training: runs discover_vulnerabilities() to rank genes and saves
               data/vulnerabilities/causal_vulnerabilities.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
import json
from tqdm import tqdm

from src.models.vuln_causal.model import VulnCausal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FullVulnerabilityDataset(Dataset):
    """
    Dataset for the full VulnCausal model.

    Provides expression, CRISPR scores, ecDNA labels, and environment IDs.
    Samples a random subset of genes per batch for the IRM predictor.
    """

    def __init__(
        self,
        crispr: pd.DataFrame,
        expression: pd.DataFrame,
        labels: pd.DataFrame,
        sample_genes_per_batch: int = 100,
    ):
        # Find common samples
        common = sorted(set(crispr.index) & set(expression.index) & set(labels.index))
        logger.info(f"Common samples: {len(common)}")

        self.sample_ids = common
        self.crispr = crispr.loc[common].values.astype(np.float32)
        self.expression = expression.loc[common].values.astype(np.float32)
        self.ecdna_labels = labels.loc[common, 'is_ecdna'].values.astype(np.float32)

        # Encode lineages as integers
        lineages = labels.loc[common, 'lineage'].fillna('Unknown')
        unique_lineages = sorted(lineages.unique())
        self.lineage_to_idx = {l: i for i, l in enumerate(unique_lineages)}
        self.environments = np.array([self.lineage_to_idx[l] for l in lineages])

        self.num_genes = self.crispr.shape[1]
        self.expression_dim = self.expression.shape[1]
        self.num_environments = len(unique_lineages)
        self.sample_genes = sample_genes_per_batch

        # Gene names
        self.gene_names = crispr.columns.tolist()

        logger.info(f"Dataset: {len(common)} samples, {self.num_genes} CRISPR genes, "
                     f"{self.expression_dim} expression genes")
        logger.info(f"ecDNA+: {int(self.ecdna_labels.sum())}, "
                     f"ecDNA-: {int((1 - self.ecdna_labels).sum())}")
        logger.info(f"Lineages: {self.num_environments}")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        # Sample random gene indices for this item
        gene_indices = np.random.choice(self.num_genes, self.sample_genes, replace=False)
        gene_indices = np.sort(gene_indices)

        return {
            'expression': torch.tensor(self.expression[idx]),
            'crispr': torch.tensor(self.crispr[idx]),
            'gene_ids': torch.tensor(gene_indices, dtype=torch.long),
            'ecdna_label': torch.tensor(self.ecdna_labels[idx]),
            'environment': torch.tensor(self.environments[idx], dtype=torch.long),
        }


def train_epoch(model, dataloader, optimizer, device, epoch, irm_warmup=10):
    """
    Train for one epoch using the full VulnCausal model.

    During IRM warmup (epoch < irm_warmup), only the encoder and graph
    losses are active. After warmup, the full IRM penalty kicks in.
    """
    model.train()
    total_loss = 0
    loss_components = {}
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        expression = batch['expression'].to(device)
        crispr = batch['crispr'].to(device)
        gene_ids = batch['gene_ids'].to(device)
        ecdna_labels = batch['ecdna_label'].to(device)
        environments = batch['environment'].to(device)

        optimizer.zero_grad()

        # Compute full loss via model.get_loss()
        losses = model.get_loss(
            expression=expression,
            crispr_scores=crispr,
            ecdna_labels=ecdna_labels,
            environments=environments,
            gene_ids=gene_ids,
        )

        # During IRM warmup, zero out the IRM penalty to let the
        # encoder stabilize before imposing invariance
        if epoch < irm_warmup:
            warmup_loss = torch.tensor(0.0, device=device)
            for k, v in losses.items():
                if k == 'total_loss':
                    continue
                if k.startswith('irm_'):
                    # Scale down IRM losses during warmup
                    scale = epoch / irm_warmup
                    warmup_loss = warmup_loss + v * scale
                else:
                    warmup_loss = warmup_loss + v
            loss = warmup_loss
        else:
            loss = losses['total_loss']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in losses.items():
            if k not in loss_components:
                loss_components[k] = 0
            loss_components[k] += v.item()
        n_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg = {k: v / max(n_batches, 1) for k, v in loss_components.items()}
    avg['effective_loss'] = total_loss / max(n_batches, 1)
    return avg


def validate(model, dataloader, device):
    """Validate the full VulnCausal model."""
    model.eval()
    total_loss = 0
    loss_components = {}
    all_ecdna_pred = []
    all_ecdna_true = []
    n_batches = 0

    # NOTE: We use torch.enable_grad() (not no_grad) because the IRM penalty
    # in get_loss() calls torch.autograd.grad on the dummy_w parameter,
    # which requires gradients to be enabled even during validation.
    for batch in dataloader:
        expression = batch['expression'].to(device)
        crispr = batch['crispr'].to(device)
        gene_ids = batch['gene_ids'].to(device)
        ecdna_labels = batch['ecdna_label'].to(device)
        environments = batch['environment'].to(device)

        # Forward pass without grad for predictions
        with torch.no_grad():
            outputs = model(
                expression=expression,
                crispr_scores=crispr,
                ecdna_labels=ecdna_labels,
                environments=environments,
                gene_ids=gene_ids,
                return_all=True,
            )

        # get_loss needs grad enabled for IRM penalty (autograd.grad on dummy_w)
        losses = model.get_loss(
            expression=expression,
            crispr_scores=crispr,
            ecdna_labels=ecdna_labels,
            environments=environments,
            gene_ids=gene_ids,
        )
        total_loss += losses['total_loss'].item()

        for k, v in losses.items():
            if k not in loss_components:
                loss_components[k] = 0
            loss_components[k] += v.item()

        # ecDNA prediction from the encoder's ecdna_status factor head
        ecdna_factor = outputs['ecdna_factor']
        ecdna_score = ecdna_factor.mean(dim=-1)
        all_ecdna_pred.extend(ecdna_score.detach().cpu().numpy())
        all_ecdna_true.extend(ecdna_labels.cpu().numpy())

        n_batches += 1

    # ecDNA prediction accuracy (using median as threshold)
    ecdna_pred_arr = np.array(all_ecdna_pred)
    ecdna_true_arr = np.array(all_ecdna_true)

    # Use the learned factor: higher = ecDNA positive
    if np.std(ecdna_pred_arr) > 1e-8 and np.std(ecdna_true_arr) > 1e-8:
        dep_corr = np.corrcoef(ecdna_pred_arr, ecdna_true_arr)[0, 1]
    else:
        dep_corr = 0.0

    # Check DAG constraint if available
    dag_violation = 0.0
    if hasattr(model, 'causal_graph') and model.use_causal_graph:
        adj = model.causal_graph.get_adjacency_matrix()
        dag_violation = model.causal_graph.dag_constraint(adj).item()

    avg_losses = {k: v / max(n_batches, 1) for k, v in loss_components.items()}

    return {
        'total_loss': total_loss / max(n_batches, 1),
        'ecdna_factor_corr': float(dep_corr),
        'dag_violation': float(dag_violation),
        **{f'avg_{k}': v for k, v in avg_losses.items()},
    }


def discover_and_save_vulnerabilities(
    model, dataset, device, output_path, top_k=100
):
    """
    Run vulnerability discovery on the full dataset and save results.
    """
    logger.info("Running vulnerability discovery...")

    # Load all data onto device (or in batches for large datasets)
    n = len(dataset.sample_ids)

    # Use all samples
    expression = torch.tensor(dataset.expression, device=device)
    crispr = torch.tensor(dataset.crispr, device=device)
    ecdna_labels = torch.tensor(dataset.ecdna_labels, device=device)
    environments = torch.tensor(dataset.environments, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        vulnerabilities = model.discover_vulnerabilities(
            expression=expression,
            crispr_scores=crispr,
            ecdna_labels=ecdna_labels,
            environments=environments,
            top_k=top_k,
        )

    # Build results dataframe
    gene_names_short = [g.split(' (')[0] for g in dataset.gene_names]

    rows = []
    for v in vulnerabilities:
        gene_id = v['gene_id']
        gene_name = gene_names_short[gene_id] if gene_id < len(gene_names_short) else f"gene_{gene_id}"
        gene_full = dataset.gene_names[gene_id] if gene_id < len(dataset.gene_names) else f"gene_{gene_id}"
        rows.append({
            'gene': gene_name,
            'gene_full': gene_full,
            'gene_id': gene_id,
            'causal_effect': v['causal_effect'],
            'specificity': v['specificity'],
            'vulnerability_score': v['vulnerability_score'],
            'druggability_score': v['druggability_score'],
            'final_score': v['final_score'],
        })

    vuln_df = pd.DataFrame(rows)
    vuln_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(vuln_df)} vulnerability scores to {output_path}")

    # Print top results
    logger.info(f"\nTop {min(30, len(vuln_df))} causal vulnerabilities:")
    print(vuln_df.head(30).to_string(index=False))

    return vuln_df


def main():
    parser = argparse.ArgumentParser(description="Train full VulnCausal model")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--factor_dim', type=int, default=16)
    parser.add_argument('--irm_warmup', type=int, default=10,
                        help="Epochs before full IRM penalty")
    parser.add_argument('--irm_penalty', type=float, default=1.0)
    parser.add_argument('--sparsity_penalty', type=float, default=0.1)
    parser.add_argument('--genes_per_batch', type=int, default=100,
                        help="Number of genes to sample per forward pass")
    parser.add_argument('--patience', type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument('--top_k', type=int, default=100,
                        help="Number of top vulnerabilities to save")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # ── Load Data ────────────────────────────────────────────────────────
    data_dir = Path("data")

    logger.info("Loading CRISPR data...")
    crispr = pd.read_csv(data_dir / "depmap" / "crispr.csv", index_col=0)
    logger.info(f"  CRISPR: {crispr.shape[0]} samples x {crispr.shape[1]} genes")

    logger.info("Loading expression data...")
    expression = pd.read_csv(data_dir / "depmap" / "expression.csv", index_col=0)
    logger.info(f"  Expression: {expression.shape[0]} samples x {expression.shape[1]} genes")

    logger.info("Loading ecDNA labels from CytoCellDB...")
    cyto = pd.read_excel(data_dir / "cytocell_db" / "CytoCellDB_Supp_File1.xlsx")
    labels = cyto[['DepMap_ID', 'ECDNA', 'lineage']].dropna(subset=['DepMap_ID'])
    labels['is_ecdna'] = (labels['ECDNA'] == 'Y').astype(int)
    labels = labels.set_index('DepMap_ID')

    # ── Create Dataset ───────────────────────────────────────────────────
    dataset = FullVulnerabilityDataset(
        crispr, expression, labels,
        sample_genes_per_batch=args.genes_per_batch,
    )

    # Train/val split (80/20)
    n_samples = len(dataset)
    indices = np.random.RandomState(42).permutation(n_samples)
    n_train = int(0.8 * n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # ── Create Model ─────────────────────────────────────────────────────
    model_config = {
        'num_genes': dataset.num_genes,
        'expression_dim': dataset.expression_dim,
        'num_environments': dataset.num_environments,
        'latent_dim': 128,
        'hidden_dim': args.hidden_dim,
        'factor_dim': args.factor_dim,
        'use_invariant_prediction': True,
        'use_causal_graph': True,
        'irm_penalty': args.irm_penalty,
        'sparsity_penalty': args.sparsity_penalty,
    }

    model = VulnCausal(**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Full VulnCausal model: {n_params:,} parameters")

    # ── Optimizer & Scheduler ────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # ── Training Loop ────────────────────────────────────────────────────
    output_dir = Path("checkpoints/vulncausal_full")
    output_dir.mkdir(exist_ok=True, parents=True)

    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    logger.info(f"\n{'='*60}")
    logger.info(f"Training Full VulnCausal")
    logger.info(f"  Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    logger.info(f"  IRM warmup: {args.irm_warmup} epochs, penalty: {args.irm_penalty}")
    logger.info(f"  Factor dim: {args.factor_dim} x 6 factors = {args.factor_dim * 6} latent")
    logger.info(f"  Causal graph: {args.factor_dim + 40 + 30} variables (NOTEARS)")
    logger.info(f"{'='*60}\n")

    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            epoch=epoch, irm_warmup=args.irm_warmup,
        )
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        history.append({
            'epoch': epoch,
            'train_loss': train_metrics['effective_loss'],
            'val_loss': val_metrics['total_loss'],
            'ecdna_factor_corr': val_metrics['ecdna_factor_corr'],
            'dag_violation': val_metrics['dag_violation'],
        })

        logger.info(
            f"Epoch {epoch:3d}: "
            f"train={train_metrics['effective_loss']:.4f} | "
            f"val={val_metrics['total_loss']:.4f}, "
            f"ecDNA_corr={val_metrics['ecdna_factor_corr']:.3f}, "
            f"DAG_h={val_metrics['dag_violation']:.4f}"
        )

        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            patience_counter = 0

            torch.save({
                'model_state_dict': model.state_dict(),
                'config': model_config,
                'epoch': epoch,
                'val_loss': best_val_loss,
                'gene_names': dataset.gene_names,
                'lineage_to_idx': dataset.lineage_to_idx,
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

    logger.info(f"  Val loss:          {final_metrics['total_loss']:.4f}")
    logger.info(f"  ecDNA factor corr: {final_metrics['ecdna_factor_corr']:.4f}")
    logger.info(f"  DAG violation:     {final_metrics['dag_violation']:.6f}")
    logger.info(f"  Best epoch:        {checkpoint['epoch']}")

    # ── Vulnerability Discovery ──────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("POST-TRAINING: Vulnerability Discovery")
    logger.info(f"{'='*60}")

    vuln_dir = data_dir / "vulnerabilities"
    vuln_dir.mkdir(exist_ok=True, parents=True)

    vuln_df = discover_and_save_vulnerabilities(
        model, dataset, device,
        output_path=vuln_dir / "causal_vulnerabilities.csv",
        top_k=args.top_k,
    )

    # Compare with existing analyses
    logger.info("\n=== COMPARISON WITH EXISTING ANALYSES ===")

    # Differential analysis
    diff_path = vuln_dir / "differential_dependency_full.csv"
    if diff_path.exists():
        diff_df = pd.read_csv(diff_path)
        top_causal = set(vuln_df.head(100)['gene'])
        top_diff = set(diff_df.head(100)['gene'])
        overlap_diff = top_causal & top_diff
        logger.info(f"Overlap with differential analysis (top 100): {len(overlap_diff)} genes")
        if overlap_diff:
            logger.info(f"  Overlapping: {sorted(overlap_diff)[:20]}")

    # Simplified model
    learned_path = vuln_dir / "learned_vulnerabilities.csv"
    if learned_path.exists():
        learned_df = pd.read_csv(learned_path)
        top_learned = set(learned_df.head(100)['gene'])
        overlap_learned = top_causal & top_learned
        logger.info(f"Overlap with simplified model (top 100): {len(overlap_learned)} genes")
        if overlap_learned:
            logger.info(f"  Overlapping: {sorted(overlap_learned)[:20]}")

    # Literature validation
    lit_path = vuln_dir / "literature_validation.csv"
    if lit_path.exists():
        lit_df = pd.read_csv(lit_path)
        lit_genes = set(lit_df['gene'])
        causal_genes = set(vuln_df['gene'])
        lit_overlap = lit_genes & causal_genes
        logger.info(f"Literature-validated genes in causal top {args.top_k}: "
                     f"{len(lit_overlap)}/{len(lit_genes)}")
        if lit_overlap:
            logger.info(f"  Validated: {sorted(lit_overlap)}")

    # Save final results
    results = {
        'model': 'VulnCausal (full)',
        'best_epoch': int(checkpoint['epoch']),
        'val_loss': final_metrics['total_loss'],
        'ecdna_factor_corr': final_metrics['ecdna_factor_corr'],
        'dag_violation': final_metrics['dag_violation'],
        'config': model_config,
        'n_parameters': n_params,
        'n_vulnerabilities_saved': len(vuln_df),
    }
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
