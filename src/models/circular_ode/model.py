"""
CircularODE: Physics-Informed Neural SDE for ecDNA Dynamics.

Main model that integrates all dynamics components to predict
ecDNA copy number evolution over time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import math

try:
    import torchsde
    HAS_TORCHSDE = True
except ImportError:
    HAS_TORCHSDE = False

from .dynamics import DriftNetwork, DiffusionNetwork, SegregationPhysics, FitnessLandscape
from .treatment import TreatmentEncoder, TreatmentEffectModel


class CircularODE(nn.Module):
    """
    Physics-Informed Neural SDE for ecDNA Copy Number Dynamics.

    Models ecDNA evolution as:
        dz(t) = f(z, t, treatment) dt + g(z) dW(t)

    Where:
    - f is the drift (deterministic dynamics)
    - g is the diffusion (stochastic component)
    - W(t) is a Wiener process

    Physics-informed constraints:
    1. Binomial segregation statistics
    2. Oncogene-driven fitness landscape
    3. Treatment-conditioned dynamics
    4. Non-negativity of copy number
    """

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
        self,
        latent_dim: int = 8,
        treatment_dim: int = 16,
        hidden_dim: int = 128,
        num_drift_layers: int = 3,
        use_physics_constraints: bool = True,
        segregation_scale: float = 0.5,
        min_diffusion: float = 0.01,
    ):
        """
        Initialize CircularODE.

        Args:
            latent_dim: Dimension of latent state
            treatment_dim: Treatment embedding dimension
            hidden_dim: Hidden layer dimension
            num_drift_layers: Number of layers in drift network
            use_physics_constraints: Whether to use physics-informed constraints
            segregation_scale: Scale of segregation noise
            min_diffusion: Minimum diffusion for stability
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.treatment_dim = treatment_dim
        self.use_physics_constraints = use_physics_constraints

        # State encoder (maps observed state to latent)
        self.state_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),  # [copy_number, time, activity]
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # State decoder (maps latent to observed)
        self.state_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

        # Copy number decoder (ensures positivity)
        self.cn_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Ensures positive output
        )

        # Drift network
        self.drift_net = DriftNetwork(
            latent_dim=latent_dim,
            treatment_dim=treatment_dim,
            hidden_dim=hidden_dim,
            num_layers=num_drift_layers,
        )

        # Diffusion network
        self.diffusion_net = DiffusionNetwork(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim // 2,
            output_type="diagonal",
            min_diffusion=min_diffusion,
        )

        # Treatment encoder
        self.treatment_encoder = TreatmentEncoder(
            embedding_dim=64,
            hidden_dim=hidden_dim,
            output_dim=treatment_dim,
        )

        # Treatment effect model
        self.treatment_effect = TreatmentEffectModel(
            treatment_dim=treatment_dim,
            hidden_dim=hidden_dim // 2,
        )

        # Physics constraints
        if use_physics_constraints:
            self.segregation_physics = SegregationPhysics(
                inheritance_model="binomial_with_selection",
            )
            self.fitness_landscape = FitnessLandscape(
                input_dim=1,
                hidden_dim=hidden_dim // 2,
            )

        # Store current treatment embedding for SDE interface
        self._current_treatment = None

    def f(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Drift function for SDE.

        Args:
            t: Time
            z: Latent state [batch, latent_dim]

        Returns:
            Drift [batch, latent_dim]
        """
        treatment_emb = self._current_treatment
        return self.drift(z, t, treatment_emb)

    def g(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Diffusion function for SDE.

        Args:
            t: Time
            z: Latent state [batch, latent_dim]

        Returns:
            Diffusion [batch, latent_dim]
        """
        return self.diffusion(z)

    def drift(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        treatment_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute drift with physics constraints.

        Args:
            z: Latent state [batch, latent_dim]
            t: Time
            treatment_emb: Treatment embedding

        Returns:
            Drift [batch, latent_dim]
        """
        # Base drift from network
        drift = self.drift_net(z, t, treatment_emb)

        if self.use_physics_constraints:
            # Decode copy number
            copy_number = self.cn_decoder(z)

            # Fitness-based growth
            fitness = self.fitness_landscape(copy_number)
            fitness_drift = fitness * 0.1  # Scale

            # Add fitness contribution to first latent dimension (out-of-place)
            first_dim = drift[:, 0:1] + fitness_drift

            # Treatment effects
            if treatment_emb is not None:
                effects = self.treatment_effect(treatment_emb)
                # Reduce growth under treatment
                first_dim = first_dim * (1 + effects["growth_effect"])

            drift = torch.cat([first_dim, drift[:, 1:]], dim=1)

        return drift

    def diffusion(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute diffusion with segregation physics.

        Args:
            z: Latent state [batch, latent_dim]

        Returns:
            Diffusion [batch, latent_dim]
        """
        # Base diffusion
        base_diff = self.diffusion_net(z)

        if self.use_physics_constraints:
            # Decode copy number
            copy_number = self.cn_decoder(z)

            # Segregation noise (scales with sqrt(CN))
            seg_noise = self.diffusion_net.get_segregation_noise(copy_number)

            # First dimension gets segregation-scaled noise (out-of-place)
            base_diff = torch.cat([base_diff[:, 0:1] * seg_noise, base_diff[:, 1:]], dim=1)

        return base_diff

    def forward(
        self,
        initial_state: torch.Tensor,
        time_points: torch.Tensor,
        treatment_info: Optional[Dict] = None,
        n_samples: int = 1,
        return_trajectories: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Simulate ecDNA evolution trajectory.

        Args:
            initial_state: Initial ecDNA state [batch, 3] (CN, time, activity)
            time_points: Times to evaluate [num_times]
            treatment_info: Treatment information dictionary
            n_samples: Number of trajectory samples
            return_trajectories: Whether to return full trajectories

        Returns:
            Dictionary with predictions
        """
        batch_size = initial_state.shape[0]
        device = initial_state.device

        # Encode initial state
        z0 = self.state_encoder(initial_state)

        # Encode treatment
        if treatment_info is not None:
            self._current_treatment = self.treatment_encoder(
                drug_ids=treatment_info.get("drug_ids"),
                categories=treatment_info.get("categories"),
                doses=treatment_info.get("doses"),
                durations=treatment_info.get("durations"),
            )
        else:
            self._current_treatment = torch.zeros(batch_size, self.treatment_dim, device=device)

        # Solve SDE
        if HAS_TORCHSDE:
            # Use torchsde for proper SDE solving
            trajectory = torchsde.sdeint(
                sde=self,
                y0=z0,
                ts=time_points,
                method='milstein',
                dt=0.1,
            )  # [num_times, batch, latent_dim]
            trajectory = trajectory.permute(1, 0, 2)  # [batch, num_times, latent_dim]
        else:
            # Fallback: Euler-Maruyama
            trajectory = self._euler_maruyama(z0, time_points)

        # Decode trajectories
        copy_numbers = self.cn_decoder(trajectory)  # [batch, num_times, 1]

        # Compute derived quantities
        results = {
            "latent_trajectory": trajectory,
            "copy_number_trajectory": copy_numbers.squeeze(-1),
            "final_copy_number": copy_numbers[:, -1, 0],
        }

        # Compute extinction and resistance probabilities
        results["extinction_probability"] = self._compute_extinction_prob(copy_numbers)
        results["resistance_probability"] = self._compute_resistance_prob(
            copy_numbers, treatment_info
        )

        # Variance (for uncertainty)
        if self.use_physics_constraints:
            expected_variance = self.segregation_physics.expected_variance(copy_numbers.mean(dim=1))
            results["expected_variance"] = expected_variance

        return results

    def _euler_maruyama(
        self,
        z0: torch.Tensor,
        time_points: torch.Tensor,
        dt: float = 0.1,
    ) -> torch.Tensor:
        """
        Euler-Maruyama SDE solver (fallback).

        Args:
            z0: Initial latent state [batch, latent_dim]
            time_points: Time points [num_times]
            dt: Time step

        Returns:
            Trajectory [batch, num_times, latent_dim]
        """
        batch_size = z0.shape[0]
        device = z0.device
        num_times = len(time_points)

        trajectory = torch.zeros(batch_size, num_times, self.latent_dim, device=device)
        z = z0.clone()

        for i, t in enumerate(time_points):
            trajectory[:, i] = z

            if i < num_times - 1:
                # Compute drift and diffusion
                t_tensor = torch.full((batch_size,), t.item() if isinstance(t, torch.Tensor) else t, device=device)
                drift = self.drift(z, t_tensor, self._current_treatment)
                diffusion = self.diffusion(z)

                # Time step
                actual_dt = (time_points[i + 1] - t).item()
                sqrt_dt = math.sqrt(abs(actual_dt))

                # Euler-Maruyama update
                noise = torch.randn_like(z)
                z = z + drift * actual_dt + diffusion * noise * sqrt_dt

        return trajectory

    def _compute_extinction_prob(
        self,
        copy_numbers: torch.Tensor,
        threshold: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute probability of ecDNA extinction.

        Args:
            copy_numbers: Copy number trajectory [batch, num_times, 1]
            threshold: Extinction threshold

        Returns:
            Extinction probability [batch]
        """
        # Probability that CN drops below threshold
        below_threshold = (copy_numbers < threshold).float()
        # Use minimum CN as indicator
        min_cn = copy_numbers.min(dim=1)[0]
        return torch.sigmoid(-min_cn.squeeze(-1) + threshold)

    def _compute_resistance_prob(
        self,
        copy_numbers: torch.Tensor,
        treatment_info: Optional[Dict],
        threshold_increase: float = 2.0,
    ) -> torch.Tensor:
        """
        Compute probability of treatment resistance.

        Args:
            copy_numbers: Copy number trajectory [batch, num_times, 1]
            treatment_info: Treatment information
            threshold_increase: Fold-increase indicating resistance

        Returns:
            Resistance probability [batch]
        """
        if treatment_info is None:
            # No treatment, no resistance concept
            return torch.zeros(copy_numbers.shape[0], device=copy_numbers.device)

        # Look for CN rebound after initial decrease
        initial_cn = copy_numbers[:, 0, 0]
        min_cn = copy_numbers.min(dim=1)[0].squeeze(-1)
        final_cn = copy_numbers[:, -1, 0]

        # Resistance: initial drop followed by rebound
        had_response = min_cn < initial_cn * 0.5  # 50% reduction
        rebounded = final_cn > min_cn * threshold_increase

        return (had_response & rebounded).float()

    def get_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        observations: Dict[str, torch.Tensor],
        physics_weight: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss with physics constraints.

        Args:
            predictions: Model predictions
            observations: Observed values
            physics_weight: Weight for physics constraint loss

        Returns:
            Dictionary of loss components
        """
        losses = {}

        # Data loss (MSE on copy numbers)
        if "copy_numbers" in observations:
            pred_cn = predictions["copy_number_trajectory"]
            obs_cn = observations["copy_numbers"]

            # Log-space for better scale handling
            losses["data_loss"] = F.mse_loss(
                torch.log1p(pred_cn),
                torch.log1p(obs_cn),
            )

        # Physics constraint: segregation variance
        if self.use_physics_constraints and "copy_number_trajectory" in predictions:
            cn_traj = predictions["copy_number_trajectory"]

            # Compute trajectory variance
            traj_var = cn_traj.var(dim=1)

            # Expected variance from segregation
            mean_cn = cn_traj.mean(dim=1)
            expected_var = self.segregation_physics.expected_variance(mean_cn)

            losses["physics_loss"] = physics_weight * F.mse_loss(traj_var, expected_var)

        # Non-negativity constraint (soft)
        if "copy_number_trajectory" in predictions:
            cn_traj = predictions["copy_number_trajectory"]
            losses["nonnegativity_loss"] = 0.01 * F.relu(-cn_traj).mean()

        # Total loss
        losses["total_loss"] = sum(losses.values())

        return losses

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs) -> "CircularODE":
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
