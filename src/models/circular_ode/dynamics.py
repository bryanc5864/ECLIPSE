"""
dynamics components for CircularODE.

Provides:
- DriftNetwork: Deterministic dynamics (growth, selection)
- DiffusionNetwork: Stochastic component (segregation noise)
- SegregationPhysics: Physics-informed constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class DriftNetwork(nn.Module):
    """
    neural network for drift (deterministic dynamics).

    Models dz/dt = f(z, t, treatment) where:
    - Growth from oncogene fitness advantage
    - Treatment-induced negative selection
    - Carrying capacity effects
    """

    def __init__(
        self,
        latent_dim: int,
        treatment_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 3,
        time_embedding_dim: int = 32,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.time_embedding_dim = time_embedding_dim

        # time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        # main network
        input_dim = latent_dim + treatment_dim + time_embedding_dim
        layers = []
        for i in range(num_layers):
            in_features = input_dim if i == 0 else hidden_dim
            out_features = hidden_dim if i < num_layers - 1 else latent_dim
            layers.extend([
                nn.Linear(in_features, out_features),
                nn.LayerNorm(out_features) if i < num_layers - 1 else nn.Identity(),
                nn.SiLU() if i < num_layers - 1 else nn.Identity(),
            ])
        self.network = nn.Sequential(*layers)

        # fitness landscape parameters
        self.fitness_scale = nn.Parameter(torch.tensor(0.1))
        self.carrying_capacity = nn.Parameter(torch.tensor(100.0))

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        treatment_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        compute drift.

        Args:
            z: Latent state [batch, latent_dim]
            t: Time [batch, 1] or scalar
            treatment_emb: Treatment embedding [batch, treatment_dim]

        Returns:
            Drift [batch, latent_dim]
        """
        batch_size = z.shape[0]

        # time embedding
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)
        time_emb = self.time_mlp(t)

        # prepare treatment
        if treatment_emb is None:
            treatment_emb = torch.zeros(batch_size, 16, device=z.device)

        # concatenate inputs
        x = torch.cat([z, treatment_emb, time_emb], dim=-1)

        # base drift from network
        drift = self.network(x)

        # add physics-informed components
        # fitness advantage (growth term)
        copy_number = z[:, 0:1]  # first dimension is copy number
        fitness_growth = self.fitness_scale * copy_number

        # carrying capacity (logistic growth)
        capacity_term = 1 - copy_number / self.carrying_capacity
        fitness_growth = fitness_growth * F.relu(capacity_term)

        # combine learned and physics-based drift (out-of-place to avoid autograd issues)
        drift = torch.cat([drift[:, 0:1] + fitness_growth, drift[:, 1:]], dim=1)

        return drift


class DiffusionNetwork(nn.Module):
    """
    neural network for diffusion (stochastic dynamics).

    Models the noise structure that captures:
    - Cell-to-cell heterogeneity
    - Binomial segregation noise (scales with sqrt(CN))
    - Measurement noise
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        output_type: str = "diagonal",
        min_diffusion: float = 0.01,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.output_type = output_type
        self.min_diffusion = min_diffusion

        if output_type == "diagonal":
            output_dim = latent_dim
        else:
            output_dim = latent_dim * latent_dim

        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus(),  # ensure positive diffusion
        )

        self.segregation_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        compute diffusion coefficient.

        Args:
            z: Latent state [batch, latent_dim]

        Returns:
            Diffusion [batch, latent_dim] for diagonal
            or [batch, latent_dim, latent_dim] for full
        """
        # base diffusion from network
        base_diff = self.network(z)

        if self.output_type == "diagonal":
            diffusion = base_diff + self.min_diffusion
        else:
            # reshape to matrix
            diffusion = base_diff.view(-1, self.latent_dim, self.latent_dim)
            # make positive semi-definite
            diffusion = torch.bmm(diffusion, diffusion.transpose(1, 2))
            diffusion = diffusion + self.min_diffusion * torch.eye(
                self.latent_dim, device=z.device
            )

        return diffusion

    def get_segregation_noise(self, copy_number: torch.Tensor) -> torch.Tensor:
        """
        compute segregation noise that scales with sqrt(copy number).

        Based on binomial segregation statistics:
        Var(CN') ~ CN * p * (1-p) where p = 0.5 for random segregation.
        So std ~ sqrt(CN/4) ~ sqrt(CN) * 0.5
        """
        return self.segregation_scale * torch.sqrt(copy_number.clamp(min=1))


class SegregationPhysics(nn.Module):
    """
    Physics-informed constraints for ecDNA segregation.

    Encodes biological knowledge about ecDNA inheritance:
    - Lack of centromeres → random segregation
    - Binomial distribution of copies to daughter cells
    - Copy number variance scales with mean
    """

    def __init__(
        self,
        inheritance_model: str = "binomial_with_selection",
        selection_strength: float = 0.1,
    ):
        super().__init__()

        self.inheritance_model = inheritance_model
        self.selection_strength = nn.Parameter(torch.tensor(selection_strength))

        # learned segregation bias (slight preference for ecDNA)
        self.segregation_bias = nn.Parameter(torch.tensor(0.5))

    def expected_variance(self, copy_number: torch.Tensor) -> torch.Tensor:
        """
        compute expected variance under binomial segregation.

        For binomial(n, p) where n = CN and p = 0.5:
        Var = n * p * (1-p) = CN * 0.25
        """
        return copy_number * 0.25

    def segregation_probability(self, copy_number: torch.Tensor) -> torch.Tensor:
        """
        compute probability of ecDNA inheritance to a daughter cell.

        Under pure random segregation, p = 0.5.
        With selection, cells with more ecDNA may have slight advantage.
        """
        base_prob = torch.sigmoid(self.segregation_bias)

        # selection: higher CN slightly increases inheritance probability
        selection_term = self.selection_strength * torch.tanh(copy_number / 50)

        return (base_prob + selection_term).clamp(0.01, 0.99)

    def constraint_loss(
        self,
        predicted_variance: torch.Tensor,
        copy_number: torch.Tensor,
    ) -> torch.Tensor:
        """
        compute physics constraint loss.

        Penalizes deviations from expected binomial variance.
        """
        expected_var = self.expected_variance(copy_number)
        return F.mse_loss(predicted_variance, expected_var)

    def sample_division(
        self,
        copy_number: torch.Tensor,
        n_samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        sample ecDNA segregation during cell division.

        Args:
            copy_number: Current copy number [batch]
            n_samples: Number of division samples
        """
        batch_size = copy_number.shape[0]
        cn_int = copy_number.round().long()

        p = self.segregation_probability(copy_number)

        # sample binomial for daughter 1
        daughter1 = torch.zeros(batch_size, n_samples, device=copy_number.device)
        for i in range(batch_size):
            n = cn_int[i].item()
            prob = p[i].item()
            if n > 0:
                daughter1[i] = torch.binomial(
                    torch.tensor([n] * n_samples, dtype=torch.float),
                    torch.tensor([prob] * n_samples)
                )

        # daughter 2 gets the rest
        daughter2 = copy_number.unsqueeze(1) - daughter1

        return daughter1, daughter2


class FitnessLandscape(nn.Module):
    """
    learned fitness landscape for ecDNA.

    Models how fitness depends on:
    - Oncogene copy number (dosage effect)
    - Treatment presence
    - Cellular context
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # optimal copy number (learned)
        self.optimal_cn = nn.Parameter(torch.tensor(20.0))
        self.fitness_width = nn.Parameter(torch.tensor(50.0))

    def forward(self, copy_number: torch.Tensor) -> torch.Tensor:
        """
        compute fitness from copy number.

        Args:
            copy_number: ecDNA copy number [batch, 1]

        Returns:
            Fitness value [batch, 1]
        """
        # learned component
        learned_fitness = self.network(copy_number)

        # prior: Gaussian centered at optimal CN
        distance_from_optimal = (copy_number - self.optimal_cn) ** 2
        gaussian_fitness = torch.exp(-distance_from_optimal / (2 * self.fitness_width ** 2))

        # combine
        return learned_fitness + gaussian_fitness
