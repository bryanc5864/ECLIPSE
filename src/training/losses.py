"""
Loss Functions for ECLIPSE.

Specialized loss functions for each module:
- FocalLoss: Handles class imbalance in ecDNA prediction
- PhysicsInformedLoss: Incorporates ecDNA biology constraints
- CausalLoss: Supports causal inference training
- MultiTaskLoss: Combines multiple objectives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    ecDNA is present in ~30% of cancers, creating class imbalance.
    Focal loss down-weights easy examples to focus on hard ones.

    Based on: Lin et al., "Focal Loss for Dense Object Detection" (2017)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Initialize focal loss.

        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: Reduction method ("mean", "sum", "none")
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predicted logits [batch, ...]
            targets: Binary targets [batch, ...]

        Returns:
            Focal loss value
        """
        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction="none"
        )

        # Probability
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Combined loss
        loss = alpha_t * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class PhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Loss for CircularODE.

    Incorporates biological constraints of ecDNA dynamics:
    1. Non-negativity of copy number
    2. Binomial segregation statistics
    3. Selection-drift balance
    4. Fitness landscape constraints
    """

    def __init__(
        self,
        data_weight: float = 1.0,
        segregation_weight: float = 0.1,
        nonnegativity_weight: float = 0.01,
        smoothness_weight: float = 0.05,
    ):
        """
        Initialize physics-informed loss.

        Args:
            data_weight: Weight for data fitting term
            segregation_weight: Weight for segregation constraint
            nonnegativity_weight: Weight for non-negativity constraint
            smoothness_weight: Weight for trajectory smoothness
        """
        super().__init__()
        self.data_weight = data_weight
        self.segregation_weight = segregation_weight
        self.nonnegativity_weight = nonnegativity_weight
        self.smoothness_weight = smoothness_weight

    def forward(
        self,
        predicted_trajectory: torch.Tensor,
        observed_trajectory: Optional[torch.Tensor] = None,
        predicted_variance: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute physics-informed loss.

        Args:
            predicted_trajectory: Predicted copy numbers [batch, time, ...]
            observed_trajectory: Observed copy numbers [batch, time, ...]
            predicted_variance: Predicted variance [batch, time]
            mask: Mask for valid time points [batch, time]

        Returns:
            Dictionary of loss components
        """
        losses = {}

        # Data fitting loss
        if observed_trajectory is not None:
            if mask is not None:
                data_loss = F.mse_loss(
                    predicted_trajectory[mask],
                    observed_trajectory[mask],
                )
            else:
                data_loss = F.mse_loss(predicted_trajectory, observed_trajectory)
            losses["data_loss"] = self.data_weight * data_loss

        # Non-negativity constraint
        nonnegativity = F.relu(-predicted_trajectory).mean()
        losses["nonnegativity_loss"] = self.nonnegativity_weight * nonnegativity

        # Segregation constraint: variance should scale with mean
        if predicted_variance is not None:
            mean_cn = predicted_trajectory.mean(dim=1)
            expected_variance = mean_cn * 0.25  # Binomial variance
            segregation_loss = F.mse_loss(predicted_variance.mean(dim=1), expected_variance)
            losses["segregation_loss"] = self.segregation_weight * segregation_loss

        # Smoothness constraint (penalize rapid changes)
        if predicted_trajectory.shape[1] > 1:
            diff = predicted_trajectory[:, 1:] - predicted_trajectory[:, :-1]
            smoothness_loss = (diff ** 2).mean()
            losses["smoothness_loss"] = self.smoothness_weight * smoothness_loss

        # Total
        losses["total_loss"] = sum(losses.values())

        return losses


class CausalLoss(nn.Module):
    """
    Causal Loss for VulnCausal.

    Combines multiple objectives for causal inference:
    1. Reconstruction loss
    2. KL divergence
    3. Independence penalty
    4. IRM penalty
    5. DAG constraint
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        kl_weight: float = 4.0,
        independence_weight: float = 1.0,
        irm_weight: float = 1.0,
        dag_weight: float = 1.0,
    ):
        """
        Initialize causal loss.

        Args:
            recon_weight: Reconstruction loss weight
            kl_weight: KL divergence weight (beta-VAE)
            independence_weight: Independence penalty weight
            irm_weight: IRM penalty weight
            dag_weight: DAG constraint weight
        """
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.independence_weight = independence_weight
        self.irm_weight = irm_weight
        self.dag_weight = dag_weight

    def forward(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        independence_loss: Optional[torch.Tensor] = None,
        irm_penalty: Optional[torch.Tensor] = None,
        dag_constraint: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute causal loss.

        Args:
            x: Original input
            reconstruction: Reconstructed input
            mu: Latent mean
            logvar: Latent log variance
            independence_loss: Pre-computed independence loss
            irm_penalty: Pre-computed IRM penalty
            dag_constraint: Pre-computed DAG constraint

        Returns:
            Dictionary of loss components
        """
        losses = {}

        # Reconstruction loss
        losses["recon_loss"] = self.recon_weight * F.mse_loss(reconstruction, x)

        # KL divergence
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        losses["kl_loss"] = self.kl_weight * kl.mean()

        # Independence penalty
        if independence_loss is not None:
            losses["independence_loss"] = self.independence_weight * independence_loss

        # IRM penalty
        if irm_penalty is not None:
            losses["irm_loss"] = self.irm_weight * irm_penalty

        # DAG constraint
        if dag_constraint is not None:
            losses["dag_loss"] = self.dag_weight * dag_constraint

        # Total
        losses["total_loss"] = sum(losses.values())

        return losses


class MultiTaskLoss(nn.Module):
    """
    Multi-Task Loss with automatic weighting.

    Uses uncertainty-based weighting or learned task weights
    to balance multiple objectives.
    """

    def __init__(
        self,
        task_names: List[str],
        initial_weights: Optional[Dict[str, float]] = None,
        learn_weights: bool = True,
    ):
        """
        Initialize multi-task loss.

        Args:
            task_names: Names of tasks
            initial_weights: Initial weights for each task
            learn_weights: Whether to learn task weights
        """
        super().__init__()

        self.task_names = task_names
        self.learn_weights = learn_weights

        # Initialize weights
        if initial_weights is None:
            initial_weights = {name: 1.0 for name in task_names}

        if learn_weights:
            # Learnable log variances for uncertainty weighting
            self.log_vars = nn.ParameterDict({
                name: nn.Parameter(torch.tensor(0.0))
                for name in task_names
            })
        else:
            self.weights = initial_weights

    def forward(
        self,
        task_losses: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted multi-task loss.

        Args:
            task_losses: Dictionary of task losses

        Returns:
            Dictionary with weighted losses and total
        """
        result = {}
        total = torch.tensor(0.0, device=next(iter(task_losses.values())).device)

        for name in self.task_names:
            if name not in task_losses:
                continue

            loss = task_losses[name]

            if self.learn_weights:
                # Uncertainty weighting: L = 1/(2*sigma^2) * loss + log(sigma)
                precision = torch.exp(-self.log_vars[name])
                weighted = precision * loss + self.log_vars[name]
                result[f"{name}_weight"] = precision
            else:
                weighted = self.weights[name] * loss

            result[name] = weighted
            total = total + weighted

        result["total_loss"] = total

        return result


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for representation learning.

    Useful for learning embeddings where ecDNA+ samples
    should be similar to each other and different from ecDNA-.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        margin: float = 1.0,
    ):
        """
        Initialize contrastive loss.

        Args:
            temperature: Temperature for softmax
            margin: Margin for triplet loss variant
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            embeddings: Sample embeddings [batch, dim]
            labels: Binary labels [batch]

        Returns:
            Contrastive loss value
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix
        similarity = torch.mm(embeddings, embeddings.t()) / self.temperature

        # Create label mask
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.t()).float()

        # Mask out self-similarity
        batch_size = embeddings.shape[0]
        mask_self = 1 - torch.eye(batch_size, device=embeddings.device)
        mask = mask * mask_self

        # Positive pairs
        pos_mask = mask
        # Negative pairs
        neg_mask = (1 - mask) * mask_self

        # InfoNCE-style loss
        exp_sim = torch.exp(similarity) * mask_self
        pos_sum = (exp_sim * pos_mask).sum(dim=1)
        neg_sum = (exp_sim * neg_mask).sum(dim=1)

        loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8) + 1e-8)

        return loss.mean()
