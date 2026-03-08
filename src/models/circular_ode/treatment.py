"""
Treatment Encoder for CircularODE.

Encodes treatment information (drug type, dose, duration) for
conditioning ecDNA dynamics on therapeutic interventions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import math


class TreatmentEncoder(nn.Module):
    """
    Encodes treatment information for conditioning dynamics.

    Supports various treatment categories:
    - Targeted therapies (EGFR inhibitors, etc.)
    - Chemotherapy (cytotoxic)
    - Immunotherapy
    - ecDNA-specific (transcription inhibitors, etc.)
    """

    # Treatment categories and example drugs
    TREATMENT_CATEGORIES = {
        "targeted": 0,
        "chemo": 1,
        "immuno": 2,
        "ecdna_specific": 3,
        "combination": 4,
        "none": 5,
    }

    # Known drugs with ecDNA effects
    KNOWN_DRUGS = {
        # Targeted therapies
        "erlotinib": ("targeted", "EGFR"),
        "gefitinib": ("targeted", "EGFR"),
        "osimertinib": ("targeted", "EGFR"),
        "trastuzumab": ("targeted", "ERBB2"),
        "lapatinib": ("targeted", "ERBB2"),
        "imatinib": ("targeted", "BCR-ABL"),
        # Chemo
        "cisplatin": ("chemo", "DNA"),
        "carboplatin": ("chemo", "DNA"),
        "paclitaxel": ("chemo", "microtubule"),
        "doxorubicin": ("chemo", "DNA"),
        # ecDNA-specific
        "actinomycin_d": ("ecdna_specific", "transcription"),
        "triptolide": ("ecdna_specific", "transcription"),
        "bes": ("ecdna_specific", "replication"),  # BET inhibitor
    }

    def __init__(
        self,
        drug_vocab_size: int = 1000,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 16,
        num_categories: int = 6,
    ):
        """
        Initialize treatment encoder.

        Args:
            drug_vocab_size: Size of drug vocabulary
            embedding_dim: Drug embedding dimension
            hidden_dim: Hidden dimension
            output_dim: Output embedding dimension
            num_categories: Number of treatment categories
        """
        super().__init__()

        self.output_dim = output_dim

        # Drug embedding
        self.drug_embedding = nn.Embedding(drug_vocab_size, embedding_dim)

        # Category embedding
        self.category_embedding = nn.Embedding(num_categories, embedding_dim // 2)

        # Dose encoder (continuous)
        self.dose_encoder = nn.Sequential(
            nn.Linear(1, embedding_dim // 4),
            nn.SiLU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 4),
        )

        # Duration encoder
        self.duration_encoder = nn.Sequential(
            nn.Linear(1, embedding_dim // 4),
            nn.SiLU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 4),
        )

        # Combination encoder (for multi-drug regimens)
        self.combination_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            batch_first=True,
        )

        # Final projection
        total_dim = embedding_dim + embedding_dim // 2 + embedding_dim // 2
        self.output_projection = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Time-varying treatment effect
        self.time_modulation = nn.Sequential(
            nn.Linear(output_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        drug_ids: Optional[torch.Tensor] = None,
        categories: Optional[torch.Tensor] = None,
        doses: Optional[torch.Tensor] = None,
        durations: Optional[torch.Tensor] = None,
        start_times: Optional[torch.Tensor] = None,
        current_time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode treatment information.

        Args:
            drug_ids: Drug identifier [batch] or [batch, num_drugs]
            categories: Treatment category [batch]
            doses: Dose level [batch] (normalized 0-1)
            durations: Treatment duration [batch] (days)
            start_times: When treatment started [batch]
            current_time: Current simulation time [batch]

        Returns:
            Treatment embedding [batch, output_dim]
        """
        batch_size = self._infer_batch_size(drug_ids, categories, doses)
        device = self._infer_device(drug_ids, categories, doses)

        # Encode drug
        if drug_ids is not None:
            if drug_ids.dim() == 1:
                drug_emb = self.drug_embedding(drug_ids)
            else:
                # Multiple drugs - use attention
                drug_embs = self.drug_embedding(drug_ids)  # [B, N, D]
                drug_emb, _ = self.combination_attention(
                    drug_embs, drug_embs, drug_embs
                )
                drug_emb = drug_emb.mean(dim=1)  # [B, D]
        else:
            drug_emb = torch.zeros(batch_size, self.drug_embedding.embedding_dim,
                                   device=device)

        # Encode category
        if categories is not None:
            cat_emb = self.category_embedding(categories)
        else:
            cat_emb = self.category_embedding(
                torch.full((batch_size,), 5, device=device)  # "none"
            )

        # Encode dose
        if doses is not None:
            dose_emb = self.dose_encoder(doses.unsqueeze(-1) if doses.dim() == 1 else doses)
        else:
            dose_emb = torch.zeros(batch_size, self.dose_encoder[-1].out_features,
                                   device=device)

        # Encode duration
        if durations is not None:
            dur_normalized = durations / 30.0  # Normalize to months
            dur_emb = self.duration_encoder(
                dur_normalized.unsqueeze(-1) if dur_normalized.dim() == 1 else dur_normalized
            )
        else:
            dur_emb = torch.zeros(batch_size, self.duration_encoder[-1].out_features,
                                  device=device)

        # Concatenate and project
        combined = torch.cat([
            drug_emb,
            cat_emb,
            dose_emb,
            dur_emb,
        ], dim=-1)

        treatment_emb = self.output_projection(combined)

        # Time-varying modulation
        if current_time is not None and start_times is not None:
            # Compute time since treatment start
            time_since_start = current_time - start_times
            time_since_start = time_since_start.unsqueeze(-1) if time_since_start.dim() == 1 else time_since_start

            # Modulate based on time (effect may build up or decay)
            time_input = torch.cat([treatment_emb, time_since_start / 30.0], dim=-1)
            treatment_emb = self.time_modulation(time_input)

        return treatment_emb

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

    def encode_treatment_sequence(
        self,
        treatments: List[Dict],
        time_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a sequence of treatments over time.

        Args:
            treatments: List of treatment dictionaries with keys:
                - drug: Drug name or ID
                - category: Treatment category
                - dose: Dose level
                - start: Start time
                - end: End time
            time_points: Time points to evaluate [num_times]

        Returns:
            Treatment embeddings [num_times, output_dim]
        """
        num_times = len(time_points)
        device = time_points.device

        embeddings = []
        for t in time_points:
            # Find active treatments at time t
            active = []
            for treat in treatments:
                if treat.get("start", 0) <= t <= treat.get("end", float('inf')):
                    active.append(treat)

            if not active:
                # No treatment
                emb = torch.zeros(1, self.output_dim, device=device)
            else:
                # Encode active treatments
                embs = []
                for treat in active:
                    single_emb = self.forward(
                        drug_ids=torch.tensor([treat.get("drug_id", 0)], device=device),
                        categories=torch.tensor([self.TREATMENT_CATEGORIES.get(
                            treat.get("category", "none"), 5
                        )], device=device),
                        doses=torch.tensor([treat.get("dose", 0.5)], device=device),
                        durations=torch.tensor([treat.get("end", t) - treat.get("start", 0)], device=device),
                        start_times=torch.tensor([treat.get("start", 0)], device=device),
                        current_time=torch.tensor([t.item()], device=device),
                    )
                    embs.append(single_emb)

                # Average if multiple active treatments
                emb = torch.stack(embs).mean(dim=0)

            embeddings.append(emb)

        return torch.cat(embeddings, dim=0)


class TreatmentEffectModel(nn.Module):
    """
    Models how treatments affect ecDNA dynamics.

    Different treatment types have different effects:
    - Targeted: May reduce ecDNA copy number temporarily
    - Chemo: General cytotoxic effect
    - ecDNA-specific: May disrupt ecDNA replication/transcription
    """

    def __init__(
        self,
        treatment_dim: int = 16,
        hidden_dim: int = 64,
    ):
        """
        Initialize treatment effect model.

        Args:
            treatment_dim: Treatment embedding dimension
            hidden_dim: Hidden dimension
        """
        super().__init__()

        # Effect on growth rate
        self.growth_effect = nn.Sequential(
            nn.Linear(treatment_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # Bounded effect
        )

        # Effect on segregation
        self.segregation_effect = nn.Sequential(
            nn.Linear(treatment_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Multiplicative factor
        )

        # Effect on selection pressure
        self.selection_effect = nn.Sequential(
            nn.Linear(treatment_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        treatment_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute treatment effects.

        Args:
            treatment_emb: Treatment embedding [batch, treatment_dim]

        Returns:
            Dictionary of effect tensors
        """
        return {
            "growth_effect": self.growth_effect(treatment_emb),
            "segregation_effect": self.segregation_effect(treatment_emb),
            "selection_effect": self.selection_effect(treatment_emb),
        }
