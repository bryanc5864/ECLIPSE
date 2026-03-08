#!/usr/bin/env python3
"""
Train VulnCausal model for ecDNA-specific vulnerability discovery.

Uses:
- CRISPR dependency scores from DepMap
- Expression data from DepMap
- ecDNA labels from CytoCellDB
- Lineage information as environments for IRM
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VulnerabilityDataset(Dataset):
    """Dataset for vulnerability discovery."""

    def __init__(
        self,
        crispr: pd.DataFrame,
        expression: pd.DataFrame,
        labels: pd.DataFrame,
        sample_genes_per_batch: int = 100,
    ):
        """
        Args:
            crispr: CRISPR scores [samples x genes]
            expression: Expression data [samples x genes]
            labels: DataFrame with 'is_ecdna' and 'lineage' columns
            sample_genes_per_batch: Number of genes to sample per forward pass
        """
        # Find common samples
        common = list(set(crispr.index) & set(expression.index) & set(labels.index))
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
        self.sample_genes = sample_genes_per_batch

        # Gene names
        self.gene_names = crispr.columns.tolist()

        logger.info(f"Dataset: {len(common)} samples, {self.num_genes} genes")
        logger.info(f"ecDNA+: {self.ecdna_labels.sum():.0f}, ecDNA-: {(1-self.ecdna_labels).sum():.0f}")
        logger.info(f"Lineages: {len(unique_lineages)}")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        # Sample random genes for this batch
        gene_indices = np.random.choice(self.num_genes, self.sample_genes, replace=False)
        gene_indices = np.sort(gene_indices)

        return {
            'expression': torch.tensor(self.expression[idx]),
            'crispr': torch.tensor(self.crispr[idx]),
            'crispr_sampled': torch.tensor(self.crispr[idx, gene_indices]),
            'gene_ids': torch.tensor(gene_indices, dtype=torch.long),
            'ecdna_label': torch.tensor(self.ecdna_labels[idx]),
            'environment': torch.tensor(self.environments[idx], dtype=torch.long),
        }


class SimplifiedVulnCausal(nn.Module):
    """
    Simplified VulnCausal for initial training.

    Focuses on learning ecDNA-specific gene dependencies with lineage correction.
    """

    def __init__(
        self,
        expression_dim: int,
        num_genes: int,
        num_environments: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
    ):
        super().__init__()

        self.num_genes = num_genes
        self.num_environments = num_environments

        # Expression encoder
        self.expr_encoder = nn.Sequential(
            nn.Linear(expression_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim),
        )

        # ecDNA factor predictor (from latent)
        self.ecdna_predictor = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Gene embedding
        self.gene_embedding = nn.Embedding(num_genes, 64)

        # CRISPR score predictor (ecDNA-aware)
        # Predicts whether gene is essential given sample's latent + ecDNA status
        self.dependency_predictor = nn.Sequential(
            nn.Linear(latent_dim + 64 + 1, hidden_dim),  # latent + gene_emb + ecdna
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Environment-specific bias (for lineage correction)
        self.env_bias = nn.Embedding(num_environments, 1)

        # ecDNA-gene interaction term (the key for vulnerability discovery)
        self.ecdna_gene_interaction = nn.Parameter(torch.zeros(num_genes))

    def forward(self, expression, gene_ids, ecdna_labels, environments):
        """
        Forward pass.

        Args:
            expression: [batch, expr_dim]
            gene_ids: [batch, num_sampled_genes]
            ecdna_labels: [batch]
            environments: [batch]

        Returns:
            predictions and auxiliary outputs
        """
        batch_size = expression.shape[0]
        num_genes = gene_ids.shape[1]

        # Encode expression
        latent = self.expr_encoder(expression)  # [batch, latent_dim]

        # Predict ecDNA from latent (auxiliary task)
        ecdna_pred = self.ecdna_predictor(latent).squeeze(-1)  # [batch]

        # Get gene embeddings
        gene_emb = self.gene_embedding(gene_ids)  # [batch, num_genes, 64]

        # Expand latent and ecdna for all genes
        latent_expanded = latent.unsqueeze(1).expand(-1, num_genes, -1)
        ecdna_expanded = ecdna_labels.unsqueeze(1).unsqueeze(2).expand(-1, num_genes, 1)

        # Concatenate features
        features = torch.cat([latent_expanded, gene_emb, ecdna_expanded], dim=-1)
        features = features.view(-1, features.shape[-1])

        # Predict dependency
        dep_pred = self.dependency_predictor(features)
        dep_pred = dep_pred.view(batch_size, num_genes)

        # Add environment bias
        env_bias = self.env_bias(environments)  # [batch, 1]
        dep_pred = dep_pred + env_bias

        # Add ecDNA-gene interaction
        interaction = self.ecdna_gene_interaction[gene_ids]  # [batch, num_genes]
        dep_pred = dep_pred + interaction * ecdna_labels.unsqueeze(1)

        return {
            'dependency_pred': dep_pred,
            'ecdna_pred': ecdna_pred,
            'latent': latent,
        }

    def get_vulnerability_scores(self):
        """
        Get ecDNA-specific vulnerability scores for all genes.

        Higher score = more vulnerable in ecDNA+ cells.
        """
        # The interaction term captures ecDNA-specific dependency
        # More negative = more essential in ecDNA+ cells
        scores = -self.ecdna_gene_interaction.detach().cpu().numpy()
        return scores


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_dep_loss = 0
    total_ecdna_loss = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        expression = batch['expression'].to(device)
        crispr_sampled = batch['crispr_sampled'].to(device)
        gene_ids = batch['gene_ids'].to(device)
        ecdna_labels = batch['ecdna_label'].to(device)
        environments = batch['environment'].to(device)

        optimizer.zero_grad()

        outputs = model(expression, gene_ids, ecdna_labels, environments)

        # Dependency prediction loss (MSE on CRISPR scores)
        dep_loss = nn.functional.mse_loss(outputs['dependency_pred'], crispr_sampled)

        # ecDNA prediction loss (auxiliary task)
        ecdna_loss = nn.functional.binary_cross_entropy_with_logits(
            outputs['ecdna_pred'], ecdna_labels
        )

        # Total loss
        loss = dep_loss + 0.1 * ecdna_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_dep_loss += dep_loss.item()
        total_ecdna_loss += ecdna_loss.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dep': f'{dep_loss.item():.4f}',
        })

    n_batches = len(dataloader)
    return {
        'total_loss': total_loss / n_batches,
        'dep_loss': total_dep_loss / n_batches,
        'ecdna_loss': total_ecdna_loss / n_batches,
    }


def validate(model, dataloader, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_dep_loss = 0

    all_ecdna_pred = []
    all_ecdna_true = []

    with torch.no_grad():
        for batch in dataloader:
            expression = batch['expression'].to(device)
            crispr_sampled = batch['crispr_sampled'].to(device)
            gene_ids = batch['gene_ids'].to(device)
            ecdna_labels = batch['ecdna_label'].to(device)
            environments = batch['environment'].to(device)

            outputs = model(expression, gene_ids, ecdna_labels, environments)

            dep_loss = nn.functional.mse_loss(outputs['dependency_pred'], crispr_sampled)
            total_dep_loss += dep_loss.item()

            all_ecdna_pred.extend(torch.sigmoid(outputs['ecdna_pred']).cpu().numpy())
            all_ecdna_true.extend(ecdna_labels.cpu().numpy())

    n_batches = len(dataloader)

    # ecDNA prediction accuracy
    ecdna_pred = np.array(all_ecdna_pred) > 0.5
    ecdna_true = np.array(all_ecdna_true) > 0.5
    ecdna_acc = (ecdna_pred == ecdna_true).mean()

    return {
        'dep_loss': total_dep_loss / n_batches,
        'ecdna_accuracy': ecdna_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--genes_per_batch', type=int, default=200)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    data_dir = Path("data")

    # Load data
    logger.info("Loading data...")
    crispr = pd.read_csv(data_dir / "depmap" / "crispr.csv", index_col=0)
    expression = pd.read_csv(data_dir / "depmap" / "expression.csv", index_col=0)

    # Load labels
    cyto = pd.read_excel(data_dir / "cytocell_db" / "CytoCellDB_Supp_File1.xlsx")
    labels = cyto[['DepMap_ID', 'ECDNA', 'lineage']].dropna(subset=['DepMap_ID'])
    labels['is_ecdna'] = (labels['ECDNA'] == 'Y').astype(int)
    labels = labels.set_index('DepMap_ID')

    # Create dataset
    dataset = VulnerabilityDataset(
        crispr, expression, labels,
        sample_genes_per_batch=args.genes_per_batch
    )

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
    model = SimplifiedVulnCausal(
        expression_dim=expression.shape[1],
        num_genes=crispr.shape[1],
        num_environments=len(dataset.lineage_to_idx),
        hidden_dim=args.hidden_dim,
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Training loop
    best_val_loss = float('inf')
    output_dir = Path("checkpoints") / "vulncausal"
    output_dir.mkdir(exist_ok=True, parents=True)

    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        logger.info(
            f"Epoch {epoch}: train_loss={train_metrics['total_loss']:.4f}, "
            f"val_dep_loss={val_metrics['dep_loss']:.4f}, "
            f"ecdna_acc={val_metrics['ecdna_accuracy']:.3f}"
        )

        if val_metrics['dep_loss'] < best_val_loss:
            best_val_loss = val_metrics['dep_loss']
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss,
                'gene_names': dataset.gene_names,
                'lineage_to_idx': dataset.lineage_to_idx,
            }, output_dir / "best_model.pt")
            logger.info(f"  Saved best model (val_loss={best_val_loss:.4f})")

    # Get final vulnerability scores
    logger.info("\n=== VULNERABILITY SCORES ===")
    vuln_scores = model.get_vulnerability_scores()

    # Create results dataframe
    gene_names = [g.split(' (')[0] for g in dataset.gene_names]
    vuln_df = pd.DataFrame({
        'gene': gene_names,
        'gene_full': dataset.gene_names,
        'vulnerability_score': vuln_scores,
    }).sort_values('vulnerability_score', ascending=False)

    vuln_df.to_csv(data_dir / "vulnerabilities" / "learned_vulnerabilities.csv", index=False)

    logger.info("\nTop 30 ecDNA-specific vulnerabilities (learned):")
    print(vuln_df.head(30).to_string())

    # Compare with differential analysis
    logger.info("\n=== COMPARISON WITH DIFFERENTIAL ANALYSIS ===")
    diff_df = pd.read_csv(data_dir / "vulnerabilities" / "top_100_vulnerabilities.csv")

    # Check overlap
    top_learned = set(vuln_df.head(100)['gene'])
    top_diff = set(diff_df.head(100)['gene'])
    overlap = top_learned & top_diff
    logger.info(f"Overlap in top 100: {len(overlap)} genes")
    logger.info(f"Overlapping genes: {sorted(overlap)[:20]}")


if __name__ == "__main__":
    main()
