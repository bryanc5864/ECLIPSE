"""
Causal Representation Learning for VulnCausal.

Learns disentangled representations that separate:
- ecDNA status
- Oncogene dosage
- Cell lineage
- Mutation burden
- Cell cycle state
- Metabolic state
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List


class CausalRepresentationLearner(nn.Module):
    """
    Variational autoencoder for causal representation learning.

    Learns disentangled latent factors that correspond to
    biologically meaningful sources of variation.

    Key innovation: Independence penalty to encourage disentanglement.
    """

    def __init__(
        self,
        input_dim: int,
        latent_factors: List[str] = None,
        factor_dim: int = 16,
        hidden_dim: int = 256,
        num_layers: int = 3,
        beta: float = 4.0,
        independence_penalty: float = 1.0,
    ):
        """
        Initialize causal representation learner.

        Args:
            input_dim: Input dimension (expression + other features)
            latent_factors: Names of latent factors to learn
            factor_dim: Dimension per factor
            hidden_dim: Hidden layer dimension
            num_layers: Number of encoder/decoder layers
            beta: Beta-VAE weight
            independence_penalty: Weight for independence loss
        """
        super().__init__()

        if latent_factors is None:
            latent_factors = [
                "ecdna_status", "oncogene_dosage", "lineage",
                "mutation_burden", "cell_cycle", "metabolic_state"
            ]

        self.latent_factors = latent_factors
        self.num_factors = len(latent_factors)
        self.factor_dim = factor_dim
        self.latent_dim = self.num_factors * factor_dim
        self.beta = beta
        self.independence_penalty = independence_penalty

        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim
            encoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.LeakyReLU(0.2),
            ])
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent parameters (mean and logvar for each factor)
        self.fc_mu = nn.Linear(hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, self.latent_dim)

        # Decoder
        decoder_layers = []
        in_dim = self.latent_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else input_dim
            decoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim) if i < num_layers - 1 else nn.Identity(),
                nn.LeakyReLU(0.2) if i < num_layers - 1 else nn.Identity(),
            ])
            in_dim = out_dim
        self.decoder = nn.Sequential(*decoder_layers)

        # Factor-specific heads (for supervision if available)
        self.factor_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(factor_dim, factor_dim),
                nn.ReLU(),
                nn.Linear(factor_dim, 1),
            )
            for name in latent_factors
        })

        # Independence discriminator (for adversarial independence)
        self.independence_disc = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_factors * (self.num_factors - 1) // 2),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: Input features [batch, input_dim]

        Returns:
            Tuple of (mu, logvar) each [batch, latent_dim]
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling.

        Args:
            mu: Mean [batch, latent_dim]
            logvar: Log variance [batch, latent_dim]

        Returns:
            Sampled latent [batch, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to reconstruction.

        Args:
            z: Latent representation [batch, latent_dim]

        Returns:
            Reconstruction [batch, input_dim]
        """
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        return_factors: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VAE.

        Args:
            x: Input features [batch, input_dim]
            return_factors: Whether to return individual factors

        Returns:
            Dictionary with reconstruction, latent, and optionally factors
        """
        # Encode
        mu, logvar = self.encode(x)

        # Sample
        z = self.reparameterize(mu, logvar)

        # Decode
        recon = self.decode(z)

        results = {
            "reconstruction": recon,
            "latent": z,
            "mu": mu,
            "logvar": logvar,
        }

        if return_factors:
            # Split latent into factors
            factors = {}
            for i, name in enumerate(self.latent_factors):
                start = i * self.factor_dim
                end = (i + 1) * self.factor_dim
                factors[name] = z[:, start:end]
            results["factors"] = factors

        return results

    def get_factor(self, z: torch.Tensor, factor_name: str) -> torch.Tensor:
        """
        Extract a specific factor from latent representation.

        Args:
            z: Full latent [batch, latent_dim]
            factor_name: Name of factor to extract

        Returns:
            Factor representation [batch, factor_dim]
        """
        idx = self.latent_factors.index(factor_name)
        start = idx * self.factor_dim
        end = (idx + 1) * self.factor_dim
        return z[:, start:end]

    def independence_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute independence penalty between factors.

        Uses total correlation as measure of dependency.

        Args:
            z: Latent representation [batch, latent_dim]

        Returns:
            Independence loss (lower = more independent)
        """
        batch_size = z.shape[0]

        # Split into factors
        factors = []
        for i in range(self.num_factors):
            start = i * self.factor_dim
            end = (i + 1) * self.factor_dim
            factors.append(z[:, start:end])

        # Compute pairwise correlations
        total_corr = 0
        num_pairs = 0

        for i in range(self.num_factors):
            for j in range(i + 1, self.num_factors):
                # Correlation between factors
                f_i = factors[i]
                f_j = factors[j]

                # Center
                f_i_centered = f_i - f_i.mean(dim=0)
                f_j_centered = f_j - f_j.mean(dim=0)

                # Cross-covariance
                cov = torch.mm(f_i_centered.t(), f_j_centered) / batch_size

                # Frobenius norm as correlation measure
                total_corr += torch.norm(cov, p='fro')
                num_pairs += 1

        return total_corr / num_pairs if num_pairs > 0 else torch.tensor(0.0)

    def get_loss(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        factor_labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            x: Original input
            outputs: Forward pass outputs
            factor_labels: Optional supervision for factors

        Returns:
            Dictionary of loss components
        """
        losses = {}

        # Reconstruction loss
        recon = outputs["reconstruction"]
        losses["recon_loss"] = F.mse_loss(recon, x)

        # KL divergence
        mu = outputs["mu"]
        logvar = outputs["logvar"]
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        losses["kl_loss"] = self.beta * kl.mean()

        # Independence loss
        z = outputs["latent"]
        losses["independence_loss"] = self.independence_penalty * self.independence_loss(z)

        # Factor supervision (if available)
        if factor_labels is not None:
            for name, label in factor_labels.items():
                if name in outputs.get("factors", {}):
                    factor = outputs["factors"][name]
                    pred = self.factor_heads[name](factor)
                    losses[f"{name}_loss"] = F.binary_cross_entropy_with_logits(
                        pred.squeeze(-1), label.float()
                    )

        # Total
        losses["total_loss"] = sum(losses.values())

        return losses


class FactorPredictor(nn.Module):
    """
    Predicts biological factors from expression data.

    Used for weak supervision of causal representation learning.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
    ):
        """
        Initialize factor predictor.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
        """
        super().__init__()

        self.predictors = nn.ModuleDict({
            "ecdna_status": self._make_predictor(input_dim, hidden_dim, 1),
            "cell_cycle": self._make_predictor(input_dim, hidden_dim, 4),  # G1, S, G2, M
            "lineage": self._make_predictor(input_dim, hidden_dim, 20),
        })

    def _make_predictor(self, input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict all factors."""
        return {name: pred(x) for name, pred in self.predictors.items()}
