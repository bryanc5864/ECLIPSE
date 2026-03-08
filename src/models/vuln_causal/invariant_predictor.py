"""
Invariant Risk Minimization for VulnCausal.

Finds ecDNA-specific vulnerabilities that hold across
different cellular environments (lineages, tissues).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List


class InvariantRiskMinimization(nn.Module):
    """
    Invariant Risk Minimization (IRM) predictor.

    Finds predictors that are invariant across different environments
    (cell lineages, cancer types). This ensures vulnerabilities
    are ecDNA-specific, not context-specific.

    Based on: Arjovsky et al., "Invariant Risk Minimization" (2019)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 2,
        irm_penalty_weight: float = 1.0,
    ):
        """
        Initialize IRM predictor.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (1 for binary)
            num_layers: Number of hidden layers
            irm_penalty_weight: Weight for IRM penalty
        """
        super().__init__()

        self.irm_penalty_weight = irm_penalty_weight

        # Feature extractor (Phi)
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            in_dim = out_dim
        self.phi = nn.Sequential(*layers)

        # Classifier (w)
        self.classifier = nn.Linear(hidden_dim, output_dim)

        # Dummy classifier for IRM penalty computation
        self.dummy_w = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [batch, input_dim]
            return_features: Whether to return intermediate features

        Returns:
            Predictions [batch, output_dim]
        """
        features = self.phi(x)
        logits = self.classifier(features) * self.dummy_w

        if return_features:
            return logits, features

        return logits

    def compute_irm_penalty(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute IRM penalty.

        The IRM penalty measures how much the optimal classifier
        differs from a scalar (1.0). If invariant, scaling by 1.0
        should be optimal.

        Args:
            logits: Model logits [batch, output_dim]
            labels: True labels [batch]

        Returns:
            IRM penalty scalar
        """
        # Scale by dummy parameter
        scaled_logits = logits * self.dummy_w

        # Compute loss
        loss = F.binary_cross_entropy_with_logits(
            scaled_logits.squeeze(-1), labels.float()
        )

        # Gradient of loss w.r.t. dummy_w
        grad = torch.autograd.grad(
            loss, self.dummy_w, create_graph=True
        )[0]

        # IRM penalty: squared gradient norm
        return grad.pow(2).mean()

    def get_loss(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        environments: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute IRM loss across environments.

        Args:
            x: Input features [batch, input_dim]
            labels: Binary labels [batch]
            environments: Environment IDs [batch]

        Returns:
            Dictionary of loss components
        """
        unique_envs = environments.unique()

        total_erm_loss = 0
        total_irm_penalty = 0

        for env in unique_envs:
            mask = environments == env
            x_env = x[mask]
            labels_env = labels[mask]

            if len(x_env) < 2:
                continue

            # Forward pass for this environment
            logits = self.forward(x_env)

            # ERM loss
            erm_loss = F.binary_cross_entropy_with_logits(
                logits.squeeze(-1), labels_env.float()
            )
            total_erm_loss += erm_loss

            # IRM penalty
            irm_penalty = self.compute_irm_penalty(logits, labels_env)
            total_irm_penalty += irm_penalty

        n_envs = len(unique_envs)

        return {
            "erm_loss": total_erm_loss / n_envs,
            "irm_penalty": self.irm_penalty_weight * total_irm_penalty / n_envs,
            "total_loss": total_erm_loss / n_envs + self.irm_penalty_weight * total_irm_penalty / n_envs,
        }


class MultiEnvironmentPredictor(nn.Module):
    """
    Predictor that explicitly models environment-specific effects.

    Separates:
    - Invariant features (shared across environments)
    - Environment-specific features
    """

    def __init__(
        self,
        input_dim: int,
        num_environments: int,
        invariant_dim: int = 64,
        specific_dim: int = 32,
        hidden_dim: int = 128,
    ):
        """
        Initialize multi-environment predictor.

        Args:
            input_dim: Input dimension
            num_environments: Number of environments
            invariant_dim: Invariant feature dimension
            specific_dim: Environment-specific dimension
            hidden_dim: Hidden dimension
        """
        super().__init__()

        # Invariant encoder (shared)
        self.invariant_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, invariant_dim),
        )

        # Environment-specific encoders
        self.specific_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, specific_dim),
            )
            for _ in range(num_environments)
        ])

        # Predictor from invariant features only
        self.invariant_predictor = nn.Linear(invariant_dim, 1)

        # Full predictor (for comparison)
        self.full_predictor = nn.Linear(invariant_dim + specific_dim, 1)

        # Domain discriminator (for adversarial training)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(invariant_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_environments),
        )

    def forward(
        self,
        x: torch.Tensor,
        environments: torch.Tensor,
        use_invariant_only: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input features [batch, input_dim]
            environments: Environment IDs [batch]
            use_invariant_only: Use only invariant features for prediction

        Returns:
            Tuple of (predictions, intermediate_outputs)
        """
        # Invariant features
        invariant = self.invariant_encoder(x)

        # Environment-specific features
        specific_list = []
        for i, enc in enumerate(self.specific_encoders):
            mask = environments == i
            if mask.any():
                specific = enc(x[mask])
                full_specific = torch.zeros(x.shape[0], specific.shape[-1], device=x.device)
                full_specific[mask] = specific
                specific_list.append(full_specific)

        if specific_list:
            specific = torch.stack(specific_list, dim=-1).sum(dim=-1)
        else:
            specific = torch.zeros(x.shape[0], self.specific_encoders[0][-1].out_features, device=x.device)

        # Prediction
        if use_invariant_only:
            logits = self.invariant_predictor(invariant)
        else:
            combined = torch.cat([invariant, specific], dim=-1)
            logits = self.full_predictor(combined)

        # Domain prediction (for adversarial)
        domain_logits = self.domain_discriminator(invariant)

        return logits, {
            "invariant_features": invariant,
            "specific_features": specific,
            "domain_logits": domain_logits,
        }

    def get_loss(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        environments: torch.Tensor,
        adversarial_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss with adversarial domain confusion.

        Args:
            x: Input features
            labels: Binary labels
            environments: Environment IDs
            adversarial_weight: Weight for adversarial loss

        Returns:
            Loss dictionary
        """
        logits, outputs = self.forward(x, environments, use_invariant_only=True)

        # Prediction loss
        pred_loss = F.binary_cross_entropy_with_logits(
            logits.squeeze(-1), labels.float()
        )

        # Domain confusion loss (invariant should not predict environment)
        domain_logits = outputs["domain_logits"]
        # Uniform distribution as target (maximize confusion)
        n_envs = domain_logits.shape[-1]
        uniform = torch.ones_like(domain_logits) / n_envs
        domain_loss = F.kl_div(
            F.log_softmax(domain_logits, dim=-1),
            uniform,
            reduction='batchmean'
        )

        return {
            "prediction_loss": pred_loss,
            "domain_confusion_loss": adversarial_weight * domain_loss,
            "total_loss": pred_loss + adversarial_weight * domain_loss,
        }
