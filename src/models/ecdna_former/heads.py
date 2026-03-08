"""
Prediction Heads for ecDNA-Former.

Provides:
- FormationHead: Predicts ecDNA formation probability
- OncogeneHead: Predicts which oncogenes will be on ecDNA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List


class FormationHead(nn.Module):
    """
    Predicts ecDNA formation probability.

    Uses the fused representation to predict whether a genomic region
    will form ecDNA. Includes calibration for reliable probability estimates.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        use_temperature_scaling: bool = True,
    ):
        """
        Initialize formation prediction head.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            use_temperature_scaling: Whether to use temperature scaling for calibration
        """
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Temperature for calibration
        self.use_temperature_scaling = use_temperature_scaling
        if use_temperature_scaling:
            self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Predict ecDNA formation probability.

        Args:
            x: Fused features [batch, input_dim]
            return_logits: If True, return logits instead of probabilities

        Returns:
            Formation probability [batch, 1] or logits if return_logits=True
        """
        logits = self.classifier(x)

        if return_logits:
            return logits

        # Apply temperature scaling
        if self.use_temperature_scaling:
            logits = logits / self.temperature

        return torch.sigmoid(logits)

    def calibrate(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """
        Calibrate temperature on validation set.

        Args:
            val_logits: Logits from validation set
            val_labels: True labels
            lr: Learning rate for optimization
            max_iter: Maximum iterations

        Returns:
            Optimal temperature value
        """
        if not self.use_temperature_scaling:
            return 1.0

        # Optimize temperature using NLL loss
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = val_logits / self.temperature
            loss = F.binary_cross_entropy_with_logits(
                scaled_logits, val_labels
            )
            loss.backward()
            return loss

        optimizer.step(closure)

        return self.temperature.item()


class OncogeneHead(nn.Module):
    """
    Predicts which oncogenes will be amplified on ecDNA.

    Multi-label classification head for predicting oncogene content.
    Common ecDNA-associated oncogenes include: MYC, MYCN, EGFR, CDK4, MDM2.
    """

    # Common ecDNA-associated oncogenes
    ONCOGENES = [
        "MYC", "MYCN", "EGFR", "ERBB2", "CDK4", "MDM2", "CCND1",
        "FGFR1", "FGFR2", "MET", "PDGFRA", "KIT", "KRAS", "BRAF",
        "PIK3CA", "CDK6", "AKT1", "AKT2", "TERT", "AR",
    ]

    def __init__(
        self,
        input_dim: int,
        num_oncogenes: int = 20,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        use_label_smoothing: bool = True,
        smoothing: float = 0.1,
    ):
        """
        Initialize oncogene prediction head.

        Args:
            input_dim: Input feature dimension
            num_oncogenes: Number of oncogenes to predict
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            use_label_smoothing: Whether to use label smoothing
            smoothing: Label smoothing factor
        """
        super().__init__()

        self.num_oncogenes = num_oncogenes
        self.use_label_smoothing = use_label_smoothing
        self.smoothing = smoothing

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Per-oncogene classifiers (allows learning gene-specific patterns)
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, 1)
            for _ in range(num_oncogenes)
        ])

        # Oncogene embeddings (for co-occurrence modeling)
        self.oncogene_embeddings = nn.Embedding(num_oncogenes, hidden_dim // 4)

        # Co-occurrence layer
        self.cooccurrence = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
        model_cooccurrence: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict oncogene content.

        Args:
            x: Fused features [batch, input_dim]
            return_logits: If True, return logits instead of probabilities
            model_cooccurrence: Whether to model co-occurrence

        Returns:
            Tuple of:
                - Oncogene probabilities [batch, num_oncogenes]
                - Co-occurrence scores [batch, num_oncogenes, num_oncogenes] (optional)
        """
        batch_size = x.shape[0]

        # Encode
        encoded = self.encoder(x)

        # Per-oncogene predictions
        logits = torch.cat([
            clf(encoded) for clf in self.classifiers
        ], dim=-1)  # [B, num_oncogenes]

        # Model co-occurrence (which oncogenes tend to appear together)
        cooccurrence_scores = None
        if model_cooccurrence:
            # Get oncogene embeddings
            oncogene_idx = torch.arange(self.num_oncogenes, device=x.device)
            oncogene_emb = self.oncogene_embeddings(oncogene_idx)  # [O, H/4]

            # Compute pairwise scores
            cooccurrence_scores = torch.zeros(
                batch_size, self.num_oncogenes, self.num_oncogenes,
                device=x.device
            )

            for i in range(self.num_oncogenes):
                for j in range(i + 1, self.num_oncogenes):
                    pair_emb = torch.cat([
                        encoded,
                        oncogene_emb[i].unsqueeze(0).expand(batch_size, -1),
                    ], dim=-1)
                    score = self.cooccurrence(pair_emb)
                    cooccurrence_scores[:, i, j] = score.squeeze(-1)
                    cooccurrence_scores[:, j, i] = score.squeeze(-1)

        if return_logits:
            return logits, cooccurrence_scores

        return torch.sigmoid(logits), cooccurrence_scores

    def get_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        formation_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute oncogene prediction loss.

        Args:
            predictions: Predicted logits [batch, num_oncogenes]
            targets: True labels [batch, num_oncogenes]
            formation_mask: Mask for ecDNA-positive samples

        Returns:
            Loss value
        """
        # Apply label smoothing
        if self.use_label_smoothing:
            targets = targets * (1 - self.smoothing) + self.smoothing / 2

        # BCE loss
        loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )

        # Only compute loss for ecDNA-positive samples (oncogenes only matter if ecDNA forms)
        if formation_mask is not None:
            loss = loss * formation_mask.unsqueeze(-1)
            return loss.sum() / (formation_mask.sum() * self.num_oncogenes + 1e-8)

        return loss.mean()

    @classmethod
    def get_oncogene_names(cls) -> List[str]:
        """Get list of oncogene names."""
        return cls.ONCOGENES


class UncertaintyHead(nn.Module):
    """
    Uncertainty estimation head.

    Predicts both the mean prediction and its uncertainty
    using a heteroscedastic model.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        """
        Initialize uncertainty head.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Mean prediction
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Log variance prediction
        self.logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict mean and uncertainty.

        Args:
            x: Input features [batch, input_dim]

        Returns:
            Tuple of:
                - Mean prediction [batch, 1]
                - Uncertainty (std) [batch, 1]
        """
        shared = self.shared(x)

        mean = torch.sigmoid(self.mean_head(shared))
        logvar = self.logvar_head(shared)

        # Convert log variance to standard deviation
        std = torch.exp(0.5 * logvar)

        return mean, std

    def get_loss(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute heteroscedastic loss.

        Args:
            x: Input features
            targets: True labels

        Returns:
            Negative log likelihood loss
        """
        mean, std = self.forward(x)

        # Gaussian negative log likelihood
        nll = 0.5 * (torch.log(std**2 + 1e-8) + (targets - mean)**2 / (std**2 + 1e-8))

        return nll.mean()
