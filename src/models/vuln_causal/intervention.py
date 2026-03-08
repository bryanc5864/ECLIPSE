"""
Do-Calculus Network for VulnCausal.

Estimates causal intervention effects using neural networks
that implement do-calculus operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List


class DoCalculusNetwork(nn.Module):
    """
    Neural network for computing intervention effects.

    Estimates P(outcome | do(treatment)) rather than
    P(outcome | treatment) by adjusting for confounders.

    Key innovation: First synthetic lethality model to apply
    formal causal inference using do-calculus.
    """

    def __init__(
        self,
        treatment_dim: int,
        outcome_dim: int = 1,
        covariate_dim: int = 64,
        hidden_dim: int = 128,
        num_treatments: int = 1000,
    ):
        """
        Initialize do-calculus network.

        Args:
            treatment_dim: Treatment embedding dimension
            outcome_dim: Outcome dimension (1 for viability)
            covariate_dim: Covariate embedding dimension
            hidden_dim: Hidden dimension
            num_treatments: Number of possible treatments (genes)
        """
        super().__init__()

        self.treatment_dim = treatment_dim
        self.num_treatments = num_treatments

        # Treatment embedding
        self.treatment_embedding = nn.Embedding(num_treatments, treatment_dim)

        # Covariate encoder
        self.covariate_encoder = nn.Sequential(
            nn.Linear(covariate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Outcome model: P(Y | do(T), X)
        self.outcome_model = nn.Sequential(
            nn.Linear(treatment_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, outcome_dim),
        )

        # Propensity model: P(T | X)
        self.propensity_model = nn.Sequential(
            nn.Linear(covariate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_treatments),
        )

        # Doubly robust estimator weights
        self.dr_alpha = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        treatment_ids: torch.Tensor,
        covariates: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict outcome under intervention.

        Args:
            treatment_ids: Treatment (gene knockout) IDs [batch]
            covariates: Covariate features [batch, covariate_dim]

        Returns:
            Predicted outcome [batch, outcome_dim]
        """
        # Encode treatment
        treatment_emb = self.treatment_embedding(treatment_ids)

        # Encode covariates
        cov_emb = self.covariate_encoder(covariates)

        # Predict outcome
        combined = torch.cat([treatment_emb, cov_emb], dim=-1)
        outcome = self.outcome_model(combined)

        return outcome

    def estimate_causal_effect(
        self,
        treatment_id: int,
        covariates: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        condition_name: str = "ecdna_status",
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate causal effect of treatment on outcome.

        Computes: E[Y | do(T=t), C=c] for a given condition.

        Args:
            treatment_id: Treatment ID
            covariates: Sample covariates [batch, covariate_dim]
            condition: Conditioning variable (e.g., ecDNA status) [batch]
            condition_name: Name of condition variable

        Returns:
            Dictionary with causal effect estimates
        """
        batch_size = covariates.shape[0]
        device = covariates.device

        treatment_ids = torch.full((batch_size,), treatment_id, device=device)

        # Predict outcome under treatment
        outcome_treated = self.forward(treatment_ids, covariates)

        # Predict outcome under control (no treatment = gene not knocked out)
        # Use a special "no knockout" treatment
        control_ids = torch.zeros_like(treatment_ids)
        outcome_control = self.forward(control_ids, covariates)

        # Average Treatment Effect (ATE)
        ate = (outcome_treated - outcome_control).mean()

        results = {
            "ate": ate,
            "outcome_treated": outcome_treated,
            "outcome_control": outcome_control,
        }

        # Conditional ATE if condition provided
        if condition is not None:
            # Effect in condition=1 group
            mask_1 = condition > 0.5
            if mask_1.any():
                cate_1 = (outcome_treated[mask_1] - outcome_control[mask_1]).mean()
                results[f"cate_{condition_name}_positive"] = cate_1

            # Effect in condition=0 group
            mask_0 = condition <= 0.5
            if mask_0.any():
                cate_0 = (outcome_treated[mask_0] - outcome_control[mask_0]).mean()
                results[f"cate_{condition_name}_negative"] = cate_0

            # Difference in effects (interaction)
            if mask_1.any() and mask_0.any():
                results["effect_difference"] = cate_1 - cate_0

        return results

    def compute_ipw_estimate(
        self,
        treatment_ids: torch.Tensor,
        covariates: torch.Tensor,
        outcomes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inverse Probability Weighting (IPW) estimate.

        Adjusts for treatment selection bias using propensity scores.

        Args:
            treatment_ids: Observed treatments [batch]
            covariates: Covariates [batch, covariate_dim]
            outcomes: Observed outcomes [batch]

        Returns:
            IPW-adjusted outcome estimate
        """
        # Compute propensity scores
        propensity_logits = self.propensity_model(covariates)
        propensity = F.softmax(propensity_logits, dim=-1)

        # Get propensity for observed treatment
        batch_idx = torch.arange(len(treatment_ids), device=treatment_ids.device)
        prop_observed = propensity[batch_idx, treatment_ids]

        # IPW weights (stabilized)
        weights = 1.0 / (prop_observed + 1e-6)
        weights = weights / weights.sum() * len(weights)  # Normalize

        # Weighted outcome
        ipw_outcome = (outcomes.squeeze() * weights).mean()

        return ipw_outcome

    def doubly_robust_estimate(
        self,
        treatment_ids: torch.Tensor,
        covariates: torch.Tensor,
        outcomes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Doubly robust estimation combining IPW and outcome modeling.

        More robust than either alone - consistent if either
        propensity or outcome model is correct.

        Args:
            treatment_ids: Observed treatments [batch]
            covariates: Covariates [batch, covariate_dim]
            outcomes: Observed outcomes [batch]

        Returns:
            Doubly robust estimate
        """
        # Outcome model prediction
        outcome_pred = self.forward(treatment_ids, covariates).squeeze()

        # IPW estimate
        ipw_est = self.compute_ipw_estimate(treatment_ids, covariates, outcomes)

        # Combine with learned weight
        alpha = torch.sigmoid(self.dr_alpha)
        dr_estimate = alpha * outcome_pred.mean() + (1 - alpha) * ipw_est

        return dr_estimate


class VulnerabilityScoringNetwork(nn.Module):
    """
    Scores therapeutic vulnerabilities for ecDNA-positive cells.

    Combines:
    1. Causal effect size
    2. Specificity to ecDNA
    3. Druggability
    4. Clinical feasibility
    """

    def __init__(
        self,
        num_genes: int,
        gene_feature_dim: int = 64,
        hidden_dim: int = 128,
    ):
        """
        Initialize vulnerability scoring network.

        Args:
            num_genes: Number of genes to score
            gene_feature_dim: Gene feature dimension
            hidden_dim: Hidden dimension
        """
        super().__init__()

        self.num_genes = num_genes

        # Gene feature encoder
        self.gene_encoder = nn.Embedding(num_genes, gene_feature_dim)

        # Vulnerability scorer
        self.scorer = nn.Sequential(
            nn.Linear(gene_feature_dim + 4, hidden_dim),  # +4 for causal features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Druggability predictor
        self.druggability = nn.Sequential(
            nn.Linear(gene_feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        gene_ids: torch.Tensor,
        causal_effects: torch.Tensor,
        specificity: torch.Tensor,
        expression_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Score vulnerability of genes.

        Args:
            gene_ids: Gene IDs [batch]
            causal_effects: Estimated causal effects [batch]
            specificity: ecDNA specificity scores [batch]
            expression_features: Optional expression context

        Returns:
            Dictionary with scores
        """
        # Gene embeddings
        gene_emb = self.gene_encoder(gene_ids)

        # Causal features
        causal_features = torch.stack([
            causal_effects,
            specificity,
            torch.abs(causal_effects),  # Magnitude
            causal_effects * specificity,  # Interaction
        ], dim=-1)

        # Combined features
        combined = torch.cat([gene_emb, causal_features], dim=-1)

        # Vulnerability score
        vuln_score = self.scorer(combined)

        # Druggability
        drug_score = self.druggability(gene_emb)

        # Final score (vulnerability * druggability)
        final_score = vuln_score * drug_score

        return {
            "vulnerability_score": vuln_score.squeeze(-1),
            "druggability_score": drug_score.squeeze(-1),
            "final_score": final_score.squeeze(-1),
        }

    def rank_genes(
        self,
        all_gene_ids: torch.Tensor,
        do_network: DoCalculusNetwork,
        covariates: torch.Tensor,
        ecdna_labels: torch.Tensor,
        top_k: int = 50,
    ) -> List[Dict]:
        """
        Rank all genes by vulnerability score.

        Args:
            all_gene_ids: All gene IDs to evaluate
            do_network: Do-calculus network for causal effects
            covariates: Sample covariates
            ecdna_labels: ecDNA status labels
            top_k: Number of top genes to return

        Returns:
            List of dictionaries with gene info and scores
        """
        results = []

        for gene_id in all_gene_ids:
            gene_id_int = gene_id.item() if isinstance(gene_id, torch.Tensor) else gene_id

            # Estimate causal effect
            effects = do_network.estimate_causal_effect(
                treatment_id=gene_id_int,
                covariates=covariates,
                condition=ecdna_labels,
            )

            # Specificity: effect_positive - effect_negative
            if "cate_ecdna_status_positive" in effects and "cate_ecdna_status_negative" in effects:
                specificity = (effects["cate_ecdna_status_positive"] -
                              effects["cate_ecdna_status_negative"])
            else:
                specificity = torch.tensor(0.0)

            # Score
            gene_tensor = torch.tensor([gene_id_int], device=covariates.device)
            scores = self.forward(
                gene_ids=gene_tensor,
                causal_effects=effects["ate"].unsqueeze(0),
                specificity=specificity.unsqueeze(0),
            )

            results.append({
                "gene_id": gene_id_int,
                "causal_effect": effects["ate"].item(),
                "specificity": specificity.item(),
                "vulnerability_score": scores["vulnerability_score"].item(),
                "druggability_score": scores["druggability_score"].item(),
                "final_score": scores["final_score"].item(),
            })

        # Sort by final score
        results = sorted(results, key=lambda x: x["final_score"], reverse=True)

        return results[:top_k]
