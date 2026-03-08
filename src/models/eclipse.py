"""
ECLIPSE: Unified Framework for ecDNA Analysis.

Integrates all three modules:
- Module 1 (ecDNA-Former): Predicts ecDNA formation
- Module 2 (CircularODE): Models ecDNA dynamics
- Module 3 (VulnCausal): Discovers therapeutic vulnerabilities

Provides unified patient stratification and treatment recommendation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List, Union
from dataclasses import dataclass
from enum import Enum

from .ecdna_former import ECDNAFormer
from .circular_ode import CircularODE
from .vuln_causal import VulnCausal


class RiskLevel(Enum):
    """Patient risk levels based on ecDNA status and dynamics."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class PatientStratification:
    """Patient stratification result from ECLIPSE."""
    patient_id: str
    ecdna_formation_probability: float
    predicted_oncogenes: List[str]
    evolution_trajectory: Optional[torch.Tensor]
    resistance_probability: float
    therapeutic_vulnerabilities: List[Dict]
    risk_level: RiskLevel
    recommended_monitoring: str
    treatment_considerations: List[str]


class ECLIPSE(nn.Module):
    """
    ECLIPSE: Extrachromosomal Circular DNA Learning for Integrated
    Prediction of Synthetic-lethality and Expression.

    A unified computational framework that combines:
    1. ecDNA formation prediction (ecDNA-Former)
    2. ecDNA evolutionary dynamics (CircularODE)
    3. Therapeutic vulnerability discovery (VulnCausal)

    Provides comprehensive patient stratification for ecDNA-related
    cancer treatment planning.
    """

    def __init__(
        self,
        # ecDNA-Former config
        sequence_model: str = "cnn",
        sequence_dim: int = 256,
        topology_dim: int = 256,
        # CircularODE config
        dynamics_latent_dim: int = 8,
        dynamics_hidden_dim: int = 128,
        # VulnCausal config
        num_genes: int = 18000,
        expression_dim: int = 20000,
        num_environments: int = 20,
        # Integration config
        integration_hidden_dim: int = 256,
        use_all_modules: bool = True,
    ):
        """
        Initialize ECLIPSE framework.

        Args:
            sequence_model: Type of sequence encoder for ecDNA-Former
            sequence_dim: Sequence embedding dimension
            topology_dim: Topology embedding dimension
            dynamics_latent_dim: CircularODE latent dimension
            dynamics_hidden_dim: CircularODE hidden dimension
            num_genes: Number of genes for VulnCausal
            expression_dim: Expression dimension
            num_environments: Number of cellular environments
            integration_hidden_dim: Integration layer hidden dimension
            use_all_modules: Whether to use all three modules
        """
        super().__init__()

        self.use_all_modules = use_all_modules

        # === Module 1: ecDNA-Former ===
        self.ecdna_former = ECDNAFormer(
            sequence_model=sequence_model,
            sequence_dim=sequence_dim,
            topology_output_dim=topology_dim,
            fusion_dim=integration_hidden_dim,
        )

        # === Module 2: CircularODE ===
        self.circular_ode = CircularODE(
            latent_dim=dynamics_latent_dim,
            hidden_dim=dynamics_hidden_dim,
        )

        # === Module 3: VulnCausal ===
        self.vuln_causal = VulnCausal(
            num_genes=num_genes,
            expression_dim=expression_dim,
            num_environments=num_environments,
            hidden_dim=integration_hidden_dim,
        )

        # === Integration Layer ===
        # Combines outputs from all modules for patient stratification
        module1_dim = integration_hidden_dim  # From ecDNA-Former
        module2_dim = dynamics_latent_dim + 2  # Latent + resistance + extinction
        module3_dim = 128  # From VulnCausal causal representation

        self.integration_network = nn.Sequential(
            nn.Linear(module1_dim + module2_dim + module3_dim, integration_hidden_dim),
            nn.LayerNorm(integration_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(integration_hidden_dim, integration_hidden_dim // 2),
            nn.GELU(),
        )

        # Risk classification head
        self.risk_classifier = nn.Sequential(
            nn.Linear(integration_hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # 4 risk levels
        )

        # Treatment response prediction
        self.treatment_response = nn.Sequential(
            nn.Linear(integration_hidden_dim // 2 + 16, 64),  # + treatment embedding
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        # ecDNA-Former inputs
        sequence_features: Optional[torch.Tensor] = None,
        topology_features: Optional[torch.Tensor] = None,
        fragile_site_features: Optional[torch.Tensor] = None,
        copy_number_features: Optional[torch.Tensor] = None,
        # CircularODE inputs
        initial_state: Optional[torch.Tensor] = None,
        time_points: Optional[torch.Tensor] = None,
        treatment_info: Optional[Dict] = None,
        # VulnCausal inputs
        expression: Optional[torch.Tensor] = None,
        crispr_scores: Optional[torch.Tensor] = None,
        ecdna_labels: Optional[torch.Tensor] = None,
        environments: Optional[torch.Tensor] = None,
        # Control
        run_all_modules: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ECLIPSE.

        Can run individual modules or full pipeline.

        Args:
            sequence_features: Pre-computed sequence features
            topology_features: Pre-computed topology features
            fragile_site_features: Pre-computed fragile site features
            copy_number_features: Copy number features
            initial_state: Initial ecDNA state for dynamics
            time_points: Time points for trajectory prediction
            treatment_info: Treatment information dictionary
            expression: Gene expression data
            crispr_scores: CRISPR dependency scores
            ecdna_labels: ecDNA status labels
            environments: Environment (lineage) IDs
            run_all_modules: Whether to run all three modules

        Returns:
            Dictionary with outputs from all modules
        """
        results = {}
        batch_size = self._infer_batch_size(
            sequence_features, expression, initial_state
        )
        device = self._infer_device(sequence_features, expression, initial_state)

        # === Module 1: ecDNA Formation Prediction ===
        if sequence_features is not None or topology_features is not None:
            former_outputs = self.ecdna_former(
                sequence_features=sequence_features,
                topology_features=topology_features,
                fragile_site_features=fragile_site_features,
                copy_number_features=copy_number_features,
                return_embeddings=True,
            )
            results["formation_probability"] = former_outputs["formation_probability"]
            results["oncogene_probabilities"] = former_outputs["oncogene_probabilities"]
            results["former_embedding"] = former_outputs.get("fused_embedding",
                                                            torch.zeros(batch_size, 256, device=device))
        else:
            results["former_embedding"] = torch.zeros(batch_size, 256, device=device)

        # === Module 2: ecDNA Dynamics ===
        if initial_state is not None and time_points is not None:
            dynamics_outputs = self.circular_ode(
                initial_state=initial_state,
                time_points=time_points,
                treatment_info=treatment_info,
            )
            results["copy_number_trajectory"] = dynamics_outputs["copy_number_trajectory"]
            results["resistance_probability"] = dynamics_outputs["resistance_probability"]
            results["extinction_probability"] = dynamics_outputs["extinction_probability"]

            # Dynamics embedding
            final_latent = dynamics_outputs["latent_trajectory"][:, -1, :]
            dynamics_emb = torch.cat([
                final_latent,
                dynamics_outputs["resistance_probability"].unsqueeze(-1),
                dynamics_outputs["extinction_probability"].unsqueeze(-1),
            ], dim=-1)
            results["dynamics_embedding"] = dynamics_emb
        else:
            results["dynamics_embedding"] = torch.zeros(batch_size, 10, device=device)

        # === Module 3: Vulnerability Discovery ===
        if expression is not None and crispr_scores is not None:
            vuln_outputs = self.vuln_causal(
                expression=expression,
                crispr_scores=crispr_scores,
                ecdna_labels=ecdna_labels if ecdna_labels is not None else torch.zeros(batch_size, device=device),
                environments=environments,
                return_all=True,
            )
            results["causal_representation"] = vuln_outputs["causal_representation"]
            results["ecdna_factor"] = vuln_outputs["ecdna_factor"]

            if "synthetic_lethality_scores" in vuln_outputs:
                results["synthetic_lethality_scores"] = vuln_outputs["synthetic_lethality_scores"]

            results["vuln_embedding"] = vuln_outputs["causal_representation"][:, :128]
        else:
            results["vuln_embedding"] = torch.zeros(batch_size, 128, device=device)

        # === Integration ===
        if run_all_modules:
            combined = torch.cat([
                results["former_embedding"],
                results["dynamics_embedding"],
                results["vuln_embedding"],
            ], dim=-1)

            integrated = self.integration_network(combined)
            results["integrated_embedding"] = integrated

            # Risk classification
            risk_logits = self.risk_classifier(integrated)
            results["risk_logits"] = risk_logits
            results["risk_probabilities"] = F.softmax(risk_logits, dim=-1)

        return results

    def stratify_patient(
        self,
        patient_id: str,
        genomic_data: Dict[str, torch.Tensor],
        clinical_data: Optional[Dict] = None,
    ) -> PatientStratification:
        """
        Generate comprehensive patient stratification.

        Args:
            patient_id: Patient identifier
            genomic_data: Dictionary with genomic features
            clinical_data: Optional clinical information

        Returns:
            PatientStratification object with all predictions
        """
        with torch.no_grad():
            # Run full pipeline
            outputs = self.forward(**genomic_data, run_all_modules=True)

        # Extract predictions
        formation_prob = outputs.get("formation_probability", torch.tensor([0.5])).item()

        # Get predicted oncogenes
        oncogene_probs = outputs.get("oncogene_probabilities", torch.zeros(1, 20))
        top_oncogenes = self._get_top_oncogenes(oncogene_probs[0])

        # Get trajectory
        trajectory = outputs.get("copy_number_trajectory", None)

        # Resistance probability
        resistance_prob = outputs.get("resistance_probability", torch.tensor([0.0])).item()

        # Vulnerability ranking (if available)
        vulnerabilities = []
        if "synthetic_lethality_scores" in outputs:
            vulnerabilities = self._format_vulnerabilities(
                outputs["synthetic_lethality_scores"]
            )

        # Risk level
        risk_probs = outputs.get("risk_probabilities", torch.tensor([[0.25, 0.25, 0.25, 0.25]]))
        risk_level = self._determine_risk_level(risk_probs[0], formation_prob)

        # Generate recommendations
        monitoring = self._recommend_monitoring(risk_level, formation_prob)
        treatments = self._recommend_treatments(
            risk_level, vulnerabilities, resistance_prob
        )

        return PatientStratification(
            patient_id=patient_id,
            ecdna_formation_probability=formation_prob,
            predicted_oncogenes=top_oncogenes,
            evolution_trajectory=trajectory,
            resistance_probability=resistance_prob,
            therapeutic_vulnerabilities=vulnerabilities,
            risk_level=risk_level,
            recommended_monitoring=monitoring,
            treatment_considerations=treatments,
        )

    def _get_top_oncogenes(
        self,
        probs: torch.Tensor,
        threshold: float = 0.3,
    ) -> List[str]:
        """Get predicted oncogenes above threshold."""
        oncogene_names = [
            "MYC", "MYCN", "EGFR", "ERBB2", "CDK4", "MDM2", "CCND1",
            "FGFR1", "FGFR2", "MET", "PDGFRA", "KIT", "KRAS", "BRAF",
            "PIK3CA", "CDK6", "AKT1", "AKT2", "TERT", "AR",
        ]

        predicted = []
        for i, prob in enumerate(probs):
            if prob > threshold and i < len(oncogene_names):
                predicted.append(oncogene_names[i])

        return predicted

    def _format_vulnerabilities(
        self,
        scores: torch.Tensor,
        top_k: int = 10,
    ) -> List[Dict]:
        """Format vulnerability scores into list of dicts."""
        values, indices = scores[0].topk(min(top_k, scores.shape[1]))

        return [
            {"gene_id": idx.item(), "score": val.item()}
            for idx, val in zip(indices, values)
        ]

    def _determine_risk_level(
        self,
        risk_probs: torch.Tensor,
        formation_prob: float,
    ) -> RiskLevel:
        """Determine patient risk level."""
        predicted_idx = risk_probs.argmax().item()

        if formation_prob > 0.8 or predicted_idx == 3:
            return RiskLevel.VERY_HIGH
        elif formation_prob > 0.6 or predicted_idx == 2:
            return RiskLevel.HIGH
        elif formation_prob > 0.3 or predicted_idx == 1:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW

    def _recommend_monitoring(
        self,
        risk_level: RiskLevel,
        formation_prob: float,
    ) -> str:
        """Generate monitoring recommendations."""
        recommendations = {
            RiskLevel.LOW: "Standard follow-up schedule",
            RiskLevel.MODERATE: "Enhanced monitoring every 3 months with liquid biopsy",
            RiskLevel.HIGH: "Frequent monitoring monthly with cfDNA analysis",
            RiskLevel.VERY_HIGH: "Intensive monitoring with ctDNA and imaging every 2-4 weeks",
        }
        return recommendations[risk_level]

    def _recommend_treatments(
        self,
        risk_level: RiskLevel,
        vulnerabilities: List[Dict],
        resistance_prob: float,
    ) -> List[str]:
        """Generate treatment considerations."""
        recommendations = []

        if risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
            recommendations.append("Consider early intervention before ecDNA emergence")

        if resistance_prob > 0.5:
            recommendations.append("High resistance risk - consider combination therapy")

        if vulnerabilities:
            top_vuln = vulnerabilities[0]
            recommendations.append(
                f"Potential synthetic lethal target: Gene {top_vuln['gene_id']}"
            )

        recommendations.append("Consult with ecDNA specialist for treatment planning")

        return recommendations

    def _infer_batch_size(self, *tensors) -> int:
        for t in tensors:
            if t is not None:
                return t.shape[0]
        return 1

    def _infer_device(self, *tensors) -> torch.device:
        for t in tensors:
            if t is not None:
                return t.device
        return torch.device("cpu")

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        load_modules: bool = True,
        **kwargs
    ) -> "ECLIPSE":
        """
        Load ECLIPSE from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            load_modules: Whether to load individual module weights
            **kwargs: Override config options
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", {})
        config.update(kwargs)

        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        return model

    def save_pretrained(
        self,
        path: str,
        config: Optional[Dict] = None,
        save_modules: bool = True,
    ):
        """
        Save ECLIPSE checkpoint.

        Args:
            path: Save path
            config: Configuration dictionary
            save_modules: Whether to save individual module checkpoints
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": config or {},
        }
        torch.save(checkpoint, path)

        if save_modules:
            # Save individual modules
            import os
            base_dir = os.path.dirname(path)
            self.ecdna_former.save_pretrained(
                os.path.join(base_dir, "ecdna_former.pt")
            )
            self.circular_ode.save_pretrained(
                os.path.join(base_dir, "circular_ode.pt")
            )
            self.vuln_causal.save_pretrained(
                os.path.join(base_dir, "vuln_causal.pt")
            )
