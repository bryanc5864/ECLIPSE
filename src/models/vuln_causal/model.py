"""
VulnCausal: Causal Inference for Therapeutic Vulnerability Discovery.

Main model that integrates all causal inference components to discover
ecDNA-specific therapeutic vulnerabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List

from .causal_encoder import CausalRepresentationLearner
from .invariant_predictor import InvariantRiskMinimization, MultiEnvironmentPredictor
from .causal_graph import NeuralCausalDiscovery, CausalGraphPrior
from .intervention import DoCalculusNetwork, VulnerabilityScoringNetwork


class VulnCausal(nn.Module):
    """
    Causal Inference for Therapeutic Vulnerability Discovery.

    Discovers synthetic lethal interactions with ecDNA using:
    1. Causal representation learning (disentanglement)
    2. Invariant Risk Minimization (context-invariant effects)
    3. Neural causal discovery (NOTEARS)
    4. Do-calculus for intervention estimation

    Key innovation: First synthetic lethality model to apply
    formal causal inference rather than correlation-based approaches.
    """

    def __init__(
        self,
        num_genes: int = 18000,
        expression_dim: int = 20000,
        num_environments: int = 20,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        factor_dim: int = 16,
        use_invariant_prediction: bool = True,
        use_causal_graph: bool = True,
        irm_penalty: float = 1.0,
        sparsity_penalty: float = 0.1,
    ):
        """
        Initialize VulnCausal.

        Args:
            num_genes: Number of genes (for CRISPR screens)
            expression_dim: Expression feature dimension
            num_environments: Number of cellular environments (lineages)
            latent_dim: Latent representation dimension
            hidden_dim: Hidden layer dimension
            factor_dim: Dimension per causal factor
            use_invariant_prediction: Whether to use IRM
            use_causal_graph: Whether to learn causal graph
            irm_penalty: IRM penalty weight
            sparsity_penalty: Graph sparsity penalty
        """
        super().__init__()

        self.num_genes = num_genes
        self.use_invariant_prediction = use_invariant_prediction
        self.use_causal_graph = use_causal_graph

        # === Causal Representation Encoder ===
        self.causal_encoder = CausalRepresentationLearner(
            input_dim=expression_dim,
            latent_factors=[
                "ecdna_status", "oncogene_dosage", "lineage",
                "mutation_burden", "cell_cycle", "metabolic_state"
            ],
            factor_dim=factor_dim,
            hidden_dim=hidden_dim,
            beta=4.0,
            independence_penalty=1.0,
        )

        # === Invariant Synthetic Lethality Predictor ===
        if use_invariant_prediction:
            self.sl_predictor = InvariantRiskMinimization(
                input_dim=factor_dim + 64,  # ecDNA factor + gene embedding
                hidden_dim=hidden_dim,
                output_dim=1,
                irm_penalty_weight=irm_penalty,
            )

            self.env_predictor = MultiEnvironmentPredictor(
                input_dim=factor_dim + 64,
                num_environments=num_environments,
                invariant_dim=64,
                specific_dim=32,
            )
        else:
            self.sl_predictor = nn.Sequential(
                nn.Linear(factor_dim + 64, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        # === Gene Embedding ===
        self.gene_embedding = nn.Embedding(num_genes, 64)

        # === Causal Graph Learner ===
        if use_causal_graph:
            # Graph over: ecDNA factor (factor_dim) + pathways (40) + top CRISPR (30)
            # = factor_dim + 40 + 30
            causal_graph_dim = factor_dim + 40 + 30
            self.causal_graph = NeuralCausalDiscovery(
                num_variables=causal_graph_dim,
                hidden_dim=hidden_dim // 2,
                sparsity_penalty=sparsity_penalty,
            )

        # === Do-Calculus Intervention Model ===
        # covariate_dim must match the encoder's actual latent dim (num_factors * factor_dim)
        self.intervention_model = DoCalculusNetwork(
            treatment_dim=64,
            outcome_dim=1,
            covariate_dim=self.causal_encoder.latent_dim,
            hidden_dim=hidden_dim,
            num_treatments=num_genes,
        )

        # === Vulnerability Scoring ===
        self.vulnerability_scorer = VulnerabilityScoringNetwork(
            num_genes=num_genes,
            gene_feature_dim=64,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        expression: torch.Tensor,
        crispr_scores: torch.Tensor,
        ecdna_labels: torch.Tensor,
        environments: Optional[torch.Tensor] = None,
        gene_ids: Optional[torch.Tensor] = None,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VulnCausal.

        Args:
            expression: Gene expression [batch, expression_dim]
            crispr_scores: CRISPR dependency scores [batch, num_genes]
            ecdna_labels: ecDNA status labels [batch]
            environments: Environment IDs (lineages) [batch]
            gene_ids: Specific genes to evaluate [batch, num_target_genes]
            return_all: Whether to return all intermediate outputs

        Returns:
            Dictionary with predictions and scores
        """
        batch_size = expression.shape[0]
        device = expression.device

        # === Step 1: Learn Disentangled Representation ===
        causal_outputs = self.causal_encoder(expression, return_factors=True)
        causal_rep = causal_outputs["latent"]
        factors = causal_outputs["factors"]

        # Get ecDNA-specific factor
        ecdna_factor = factors["ecdna_status"]

        results = {
            "causal_representation": causal_rep,
            "ecdna_factor": ecdna_factor,
        }

        # === Step 2: Predict Synthetic Lethality ===
        if gene_ids is not None:
            # For specific genes
            gene_emb = self.gene_embedding(gene_ids)  # [batch, num_genes, 64]

            # Combine ecDNA factor with gene embedding
            ecdna_expanded = ecdna_factor.unsqueeze(1).expand(-1, gene_ids.shape[1], -1)
            combined = torch.cat([ecdna_expanded, gene_emb], dim=-1)

            # Flatten for prediction
            combined_flat = combined.view(-1, combined.shape[-1])

            if self.use_invariant_prediction:
                sl_scores_flat = self.sl_predictor(combined_flat)
            else:
                sl_scores_flat = self.sl_predictor(combined_flat)

            sl_scores = sl_scores_flat.view(batch_size, gene_ids.shape[1])
            results["synthetic_lethality_scores"] = sl_scores

        # === Step 3: Learn Causal Graph ===
        if self.use_causal_graph:
            # Create pathway-level features
            # (simplified: use top variance genes as proxies for pathways)
            pathway_features = self._extract_pathway_features(expression)
            graph_features = torch.cat([
                ecdna_factor,
                pathway_features,
                crispr_scores[:, :30],  # Top dependencies
            ], dim=-1)

            _, adj_matrix = self.causal_graph(graph_features)
            results["causal_graph"] = adj_matrix

        # === Step 4: Compute Causal Effects ===
        if gene_ids is not None:
            causal_effects = []
            for gene_idx in range(gene_ids.shape[1]):
                gene_id = gene_ids[0, gene_idx].item()  # Assume same genes for batch
                effects = self.intervention_model.estimate_causal_effect(
                    treatment_id=gene_id,
                    covariates=causal_rep,
                    condition=ecdna_labels,
                )
                causal_effects.append(effects["ate"])

            results["causal_effects"] = torch.stack(causal_effects)

        if return_all:
            results["reconstruction"] = causal_outputs["reconstruction"]
            results["factors"] = factors

        return results

    def _extract_pathway_features(
        self,
        expression: torch.Tensor,
        num_pathways: int = 20,
    ) -> torch.Tensor:
        """
        Extract pathway-level features from expression.

        Simplified: uses variance-based aggregation.
        In production, would use gene sets from MSigDB.
        """
        # Split genes into pseudo-pathways
        genes_per_pathway = expression.shape[1] // num_pathways

        pathway_features = []
        for i in range(num_pathways):
            start = i * genes_per_pathway
            end = (i + 1) * genes_per_pathway
            pathway_expr = expression[:, start:end]

            # Mean and variance as features
            pathway_features.append(pathway_expr.mean(dim=1))
            pathway_features.append(pathway_expr.var(dim=1))

        return torch.stack(pathway_features, dim=-1)

    def discover_vulnerabilities(
        self,
        expression: torch.Tensor,
        crispr_scores: torch.Tensor,
        ecdna_labels: torch.Tensor,
        environments: torch.Tensor,
        top_k: int = 50,
    ) -> List[Dict]:
        """
        Systematically discover ecDNA-specific vulnerabilities.

        Args:
            expression: Expression data for all samples
            crispr_scores: CRISPR scores for all samples
            ecdna_labels: ecDNA status for all samples
            environments: Environment IDs for all samples
            top_k: Number of top vulnerabilities to return

        Returns:
            Ranked list of therapeutic vulnerabilities
        """
        device = expression.device

        # Get causal representation
        with torch.no_grad():
            causal_outputs = self.causal_encoder(expression, return_factors=True)
            causal_rep = causal_outputs["latent"]

        # Score all genes
        all_gene_ids = torch.arange(min(self.num_genes, crispr_scores.shape[1]),
                                    device=device)

        results = self.vulnerability_scorer.rank_genes(
            all_gene_ids=all_gene_ids,
            do_network=self.intervention_model,
            covariates=causal_rep,
            ecdna_labels=ecdna_labels,
            top_k=top_k,
        )

        return results

    def get_loss(
        self,
        expression: torch.Tensor,
        crispr_scores: torch.Tensor,
        ecdna_labels: torch.Tensor,
        environments: torch.Tensor,
        gene_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            expression: Expression data
            crispr_scores: True CRISPR scores
            ecdna_labels: ecDNA labels
            environments: Environment IDs
            gene_ids: Genes to predict

        Returns:
            Loss dictionary
        """
        losses = {}

        # Causal encoder loss
        causal_outputs = self.causal_encoder(expression, return_factors=True)
        encoder_losses = self.causal_encoder.get_loss(
            expression,
            causal_outputs,
            factor_labels={"ecdna_status": ecdna_labels}
        )
        losses.update({f"encoder_{k}": v for k, v in encoder_losses.items()
                       if k != "total_loss"})

        # Get ecDNA factor
        ecdna_factor = causal_outputs["factors"]["ecdna_status"]

        # Synthetic lethality prediction loss
        if gene_ids is not None and self.use_invariant_prediction:
            gene_emb = self.gene_embedding(gene_ids)
            ecdna_expanded = ecdna_factor.unsqueeze(1).expand(-1, gene_ids.shape[1], -1)
            combined = torch.cat([ecdna_expanded, gene_emb], dim=-1).view(-1, ecdna_factor.shape[-1] + 64)

            # Get corresponding CRISPR scores
            batch_size = expression.shape[0]
            target_scores = torch.gather(crispr_scores, 1, gene_ids)
            target_flat = target_scores.view(-1)

            # Expand environments
            envs_expanded = environments.unsqueeze(1).expand(-1, gene_ids.shape[1]).reshape(-1)

            # IRM loss
            irm_losses = self.sl_predictor.get_loss(
                combined,
                (target_flat < -0.5).float(),  # Dependency threshold
                envs_expanded,
            )
            losses.update({f"irm_{k}": v for k, v in irm_losses.items()
                           if k != "total_loss"})

        # Causal graph loss
        if self.use_causal_graph:
            pathway_features = self._extract_pathway_features(expression)
            graph_features = torch.cat([
                ecdna_factor,
                pathway_features,
                crispr_scores[:, :30],
            ], dim=-1)

            graph_losses = self.causal_graph.get_loss(graph_features)
            losses.update({f"graph_{k}": v for k, v in graph_losses.items()
                           if k != "total_loss"})

        # Total loss
        losses["total_loss"] = sum(losses.values())

        return losses

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs) -> "VulnCausal":
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", {})
        config.update(kwargs)

        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])

        return model

    def save_pretrained(self, path: str, config: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": config or {},
        }
        torch.save(checkpoint, path)
