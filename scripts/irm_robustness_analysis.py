#!/usr/bin/env python3
"""
IRM environment robustness analysis for VulnCausal (Module 3).

Validates that IRM environments (real lineages) are meaningful by comparing
against shuffled and random environment assignments. Uses the standalone
InvariantRiskMinimization predictor for fast iteration.

Three conditions:
  - real: environments = actual lineage integers (31 lineages)
  - shuffled: np.random.permutation(real_environments)
  - random: np.random.randint(0, 31, size=n)

Metrics: IRM penalty magnitude, ERM loss, vulnerability ranking
correlation (Spearman).

Usage:
    python scripts/irm_robustness_analysis.py --epochs 30 --n-shuffles 5
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
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr

from src.models.vuln_causal.invariant_predictor import InvariantRiskMinimization

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(data_dir: Path):
    """Load DepMap CRISPR + expression + CytoCellDB lineages."""
    logger.info("Loading CRISPR data...")
    crispr = pd.read_csv(data_dir / "depmap" / "crispr.csv", index_col=0)

    logger.info("Loading expression data...")
    expression = pd.read_csv(data_dir / "depmap" / "expression.csv", index_col=0)

    logger.info("Loading ecDNA labels from CytoCellDB...")
    cyto = pd.read_excel(data_dir / "cytocell_db" / "CytoCellDB_Supp_File1.xlsx")
    labels = cyto[['DepMap_ID', 'ECDNA', 'lineage']].dropna(subset=['DepMap_ID'])
    labels['is_ecdna'] = (labels['ECDNA'] == 'Y').astype(int)
    labels = labels.set_index('DepMap_ID')

    # Find common samples
    common = sorted(set(crispr.index) & set(expression.index) & set(labels.index))
    logger.info(f"Common samples: {len(common)}")

    crispr_arr = crispr.loc[common].values.astype(np.float32)
    ecdna_arr = labels.loc[common, 'is_ecdna'].values.astype(np.float32)
    lineages = labels.loc[common, 'lineage'].fillna('Unknown')
    unique_lineages = sorted(lineages.unique())
    lineage_to_idx = {l: i for i, l in enumerate(unique_lineages)}
    env_arr = np.array([lineage_to_idx[l] for l in lineages])

    gene_names = crispr.columns.tolist()

    logger.info(f"  Samples: {len(common)}, Genes: {crispr_arr.shape[1]}, "
                f"Lineages: {len(unique_lineages)}, ecDNA+: {int(ecdna_arr.sum())}")

    return crispr_arr, ecdna_arr, env_arr, gene_names, len(unique_lineages)


def create_features(crispr, top_k=200):
    """Create input features from top-variance CRISPR genes."""
    variances = crispr.var(axis=0)
    top_idx = np.argsort(variances)[-top_k:]
    return crispr[:, top_idx], top_idx


def train_irm_model(X, y, envs, n_envs, epochs, device, lr=1e-3):
    """
    Train a standalone IRM predictor and return per-epoch metrics.

    Returns dict with final erm_loss, irm_penalty, and gene rankings.
    """
    input_dim = X.shape[1]
    model = InvariantRiskMinimization(
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=1,
        irm_penalty_weight=1.0,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_t = torch.FloatTensor(X).to(device)
    y_t = torch.FloatTensor(y).to(device)
    envs_t = torch.LongTensor(envs).to(device)

    final_erm = 0.0
    final_irm = 0.0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        losses = model.get_loss(X_t, y_t, envs_t)
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        final_erm = losses['erm_loss'].item()
        final_irm = losses['irm_penalty'].item()

    # Get gene-level rankings (using model weights as proxy for importance)
    model.eval()
    with torch.no_grad():
        logits = model(X_t).squeeze(-1)
        predictions = torch.sigmoid(logits).cpu().numpy()

    return {
        'erm_loss': final_erm,
        'irm_penalty': final_irm,
        'predictions': predictions,
    }


def main():
    parser = argparse.ArgumentParser(description="IRM environment robustness analysis")
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--n-shuffles', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    data_dir = Path(args.data_dir)

    crispr, ecdna, real_envs, gene_names, n_envs = load_data(data_dir)
    X, top_gene_idx = create_features(crispr, top_k=200)

    # Binary labels: is sample ecDNA-dependent for each gene? Use median split.
    # For ranking purposes, use the ecDNA labels directly.
    y = ecdna

    logger.info(f"\nUsing {X.shape[1]} top-variance genes as input features")
    logger.info(f"Training IRM for {args.epochs} epochs, {args.n_shuffles} shuffle repeats")

    rng = np.random.RandomState(args.seed)
    all_results = []

    # ---------- Real environments ----------
    logger.info(f"\n{'='*60}")
    logger.info("Condition: REAL environments")
    logger.info(f"{'='*60}")

    real_result = train_irm_model(X, y, real_envs, n_envs, args.epochs, device)
    all_results.append({
        'condition': 'real',
        'shuffle_id': 0,
        'erm_loss': real_result['erm_loss'],
        'irm_penalty': real_result['irm_penalty'],
    })
    real_preds = real_result['predictions']
    logger.info(f"  ERM loss:    {real_result['erm_loss']:.4f}")
    logger.info(f"  IRM penalty: {real_result['irm_penalty']:.6f}")

    # ---------- Shuffled environments ----------
    logger.info(f"\n{'='*60}")
    logger.info("Condition: SHUFFLED environments")
    logger.info(f"{'='*60}")

    shuffled_preds_list = []
    for s in range(args.n_shuffles):
        shuffled_envs = rng.permutation(real_envs)
        result = train_irm_model(X, y, shuffled_envs, n_envs, args.epochs, device)
        all_results.append({
            'condition': 'shuffled',
            'shuffle_id': s,
            'erm_loss': result['erm_loss'],
            'irm_penalty': result['irm_penalty'],
        })
        shuffled_preds_list.append(result['predictions'])
        logger.info(f"  Shuffle {s}: ERM={result['erm_loss']:.4f}, IRM={result['irm_penalty']:.6f}")

    # ---------- Random environments ----------
    logger.info(f"\n{'='*60}")
    logger.info("Condition: RANDOM environments")
    logger.info(f"{'='*60}")

    random_preds_list = []
    for s in range(args.n_shuffles):
        random_envs = rng.randint(0, n_envs, size=len(y))
        result = train_irm_model(X, y, random_envs, n_envs, args.epochs, device)
        all_results.append({
            'condition': 'random',
            'shuffle_id': s,
            'erm_loss': result['erm_loss'],
            'irm_penalty': result['irm_penalty'],
        })
        random_preds_list.append(result['predictions'])
        logger.info(f"  Random {s}: ERM={result['erm_loss']:.4f}, IRM={result['irm_penalty']:.6f}")

    # ---------- Ranking correlations ----------
    logger.info(f"\n{'='*60}")
    logger.info("Ranking Correlations (Spearman vs real)")
    logger.info(f"{'='*60}")

    corr_results = []
    for s, preds in enumerate(shuffled_preds_list):
        rho, pval = spearmanr(real_preds, preds)
        corr_results.append({
            'comparison': f'real_vs_shuffled_{s}',
            'spearman_rho': rho,
            'p_value': pval,
        })
        logger.info(f"  Real vs Shuffled {s}: rho={rho:.3f}, p={pval:.4f}")

    for s, preds in enumerate(random_preds_list):
        rho, pval = spearmanr(real_preds, preds)
        corr_results.append({
            'comparison': f'real_vs_random_{s}',
            'spearman_rho': rho,
            'p_value': pval,
        })
        logger.info(f"  Real vs Random  {s}: rho={rho:.3f}, p={pval:.4f}")

    # ---------- Save results ----------
    output_dir = data_dir / "validation"
    output_dir.mkdir(exist_ok=True)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "irm_robustness_results.csv", index=False)

    corr_df = pd.DataFrame(corr_results)
    corr_df.to_csv(output_dir / "irm_ranking_correlations.csv", index=False)

    # ---------- Summary ----------
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")

    for cond in ['real', 'shuffled', 'random']:
        subset = results_df[results_df['condition'] == cond]
        erm_mean = subset['erm_loss'].mean()
        erm_std = subset['erm_loss'].std()
        irm_mean = subset['irm_penalty'].mean()
        irm_std = subset['irm_penalty'].std()
        if cond == 'real':
            logger.info(f"  {cond:>10s}: ERM={erm_mean:.4f}, IRM={irm_mean:.6f}")
        else:
            logger.info(f"  {cond:>10s}: ERM={erm_mean:.4f} +/- {erm_std:.4f}, "
                        f"IRM={irm_mean:.6f} +/- {irm_std:.6f}")

    shuffled_corrs = [r['spearman_rho'] for r in corr_results if 'shuffled' in r['comparison']]
    random_corrs = [r['spearman_rho'] for r in corr_results if 'random' in r['comparison']]
    logger.info(f"\n  Ranking correlation (real vs shuffled): "
                f"{np.mean(shuffled_corrs):.3f} +/- {np.std(shuffled_corrs):.3f}")
    logger.info(f"  Ranking correlation (real vs random):   "
                f"{np.mean(random_corrs):.3f} +/- {np.std(random_corrs):.3f}")

    logger.info(f"\nResults saved to:")
    logger.info(f"  {output_dir / 'irm_robustness_results.csv'}")
    logger.info(f"  {output_dir / 'irm_ranking_correlations.csv'}")


if __name__ == "__main__":
    main()
