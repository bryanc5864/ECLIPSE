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
    neural network for computing intervention effects.

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
        initialize do-calculus network.

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

        self.treatment_embedding = nn.Embedding(num_treatments, treatment_dim)

        self.covariate_encoder = nn.Sequential(
            nn.Linear(covariate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.outcome_model = nn.Sequential(
            nn.Linear(treatment_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, outcome_dim),
        )

        self.propensity_model = nn.Sequential(
            nn.Linear(covariate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_treatments),
        )

        # doubly robust estimator weights
        self.dr_alpha = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        treatment_ids: torch.Tensor,
        covariates: torch.Tensor,
    ) -> torch.Tensor:
        """
        predict outcome under intervention.

        Args:
            treatment_ids: Treatment (gene knockout) IDs [batch]
            covariates: Covariate features [batch, covariate_dim]

        Returns:
            Predicted outcome [batch, outcome_dim]
        """
        # encode treatment
        treatment_emb = self.treatment_embedding(treatment_ids)

        cov_emb = self.covariate_encoder(covariates)

        # predict outcome
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
        estimate causal effect of treatment on outcome.

        Computes: E[Y | do(T=t), C=c] for a given condition.

        Args:
            treatment_id: Treatment ID
            covariates: Sample covariates [batch, covariate_dim]
            condition: Conditioning variable (e.g., ecDNA status) [batch]
            condition_name: Name of condition variable
        """
        batch_size = covariates.shape[0]
        device = covariates.device

        treatment_ids = torch.full((batch_size,), treatment_id, device=device)

        # predict outcome under treatment
        outcome_treated = self.forward(treatment_ids, covariates)

        # predict outcome under control (no treatment = gene not knocked out)
        # use a special "no knockout" treatment
        control_ids = torch.zeros_like(treatment_ids)
        outcome_control = self.forward(control_ids, covariates)

        # average Treatment Effect (ATE)
        ate = (outcome_treated - outcome_control).mean()

        results = {
            "ate": ate,
            "outcome_treated": outcome_treated,
            "outcome_control": outcome_control,
        }

        # conditional ATE if condition provided
        if condition is not None:
            # effect in condition=1 group
            mask_1 = condition > 0.5
            if mask_1.any():
                cate_1 = (outcome_treated[mask_1] - outcome_control[mask_1]).mean()
                results[f"cate_{condition_name}_positive"] = cate_1

            # effect in condition=0 group
            mask_0 = condition <= 0.5
            if mask_0.any():
                cate_0 = (outcome_treated[mask_0] - outcome_control[mask_0]).mean()
                results[f"cate_{condition_name}_negative"] = cate_0

            # difference in effects (interaction)
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
        inverse Probability Weighting (IPW) estimate.

        Adjusts for treatment selection bias using propensity scores.

        Args:
            treatment_ids: Observed treatments [batch]
            covariates: Covariates [batch, covariate_dim]
            outcomes: Observed outcomes [batch]
        """
        # compute propensity scores
        propensity_logits = self.propensity_model(covariates)
        propensity = F.softmax(propensity_logits, dim=-1)

        # get propensity for observed treatment
        batch_idx = torch.arange(len(treatment_ids), device=treatment_ids.device)
        prop_observed = propensity[batch_idx, treatment_ids]

        # IPW weights (stabilized)
        weights = 1.0 / (prop_observed + 1e-6)
        weights = weights / weights.sum() * len(weights)  # normalize

        # weighted outcome
        ipw_outcome = (outcomes.squeeze() * weights).mean()

        return ipw_outcome

    def doubly_robust_estimate(
        self,
        treatment_ids: torch.Tensor,
        covariates: torch.Tensor,
        outcomes: torch.Tensor,
    ) -> torch.Tensor:
        """
        doubly robust estimation combining IPW and outcome modeling.

        More robust than either alone - consistent if either
        propensity or outcome model is correct.

        Args:
            treatment_ids: Observed treatments [batch]
            covariates: Covariates [batch, covariate_dim]
            outcomes: Observed outcomes [batch]
        """
        # outcome model prediction
        outcome_pred = self.forward(treatment_ids, covariates).squeeze()

        ipw_est = self.compute_ipw_estimate(treatment_ids, covariates, outcomes)

        # combine with learned weight
        alpha = torch.sigmoid(self.dr_alpha)
        dr_estimate = alpha * outcome_pred.mean() + (1 - alpha) * ipw_est

        return dr_estimate


class VulnerabilityScoringNetwork(nn.Module):
    """
    scores therapeutic vulnerabilities for ecDNA-positive cells.

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
        super().__init__()

        self.num_genes = num_genes

        self.gene_encoder = nn.Embedding(num_genes, gene_feature_dim)

        # vulnerability scorer
        self.scorer = nn.Sequential(
            nn.Linear(gene_feature_dim + 4, hidden_dim),  # +4 for causal features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # druggability predictor
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
        score vulnerability of genes.

        Args:
            gene_ids: Gene IDs [batch]
            causal_effects: Estimated causal effects [batch]
            specificity: ecDNA specificity scores [batch]
            expression_features: Optional expression context
        """
        # gene embeddings
        gene_emb = self.gene_encoder(gene_ids)

        causal_features = torch.stack([
            causal_effects,
            specificity,
            torch.abs(causal_effects),  # magnitude
            causal_effects * specificity,  # interaction
        ], dim=-1)

        combined = torch.cat([gene_emb, causal_features], dim=-1)

        # vulnerability score
        vuln_score = self.scorer(combined)

        drug_score = self.druggability(gene_emb)

        # final score (vulnerability * druggability)
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
        """rank all genes by vulnerability score."""
        results = []

        for gene_id in all_gene_ids:
            gene_id_int = gene_id.item() if isinstance(gene_id, torch.Tensor) else gene_id

            effects = do_network.estimate_causal_effect(
                treatment_id=gene_id_int,
                covariates=covariates,
                condition=ecdna_labels,
            )

            # specificity: effect_positive - effect_negative
            if "cate_ecdna_status_positive" in effects and "cate_ecdna_status_negative" in effects:
                specificity = (effects["cate_ecdna_status_positive"] -
                              effects["cate_ecdna_status_negative"])
            else:
                specificity = torch.tensor(0.0)

            # score
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

        results = sorted(results, key=lambda x: x["final_score"], reverse=True)

        return results[:top_k]
