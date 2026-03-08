"""
ecDNA-Former: Main Model Architecture.

Integrates sequence, topology, and fragile site encoders
for predicting ecDNA formation probability and oncogene content.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List

from .sequence_encoder import SequenceEncoder
from .topology_encoder import TopologyEncoder
from .fragile_site_encoder import FragileSiteEncoder
from .fusion import CrossModalFusion, HierarchicalFusion
from .heads import FormationHead, OncogeneHead, UncertaintyHead


class ECDNAFormer(nn.Module):
    """
    ecDNA-Former: Topological Deep Learning for ecDNA Formation Prediction.

    A multi-modal transformer that predicts ecDNA formation probability
    from genomic context, including:
    - DNA sequence features (via DNA language model)
    - Chromatin topology (via hierarchical graph transformer)
    - Chromosomal fragile site context
    - Copy number features

    Innovations:
    1. Circular-aware positional encoding for ecDNA
    2. Multi-resolution Hi-C encoding
    3. Fragile site attention
    4. Bottleneck cross-modal fusion
    """

    def __init__(
        self,
        # Sequence encoder config
        sequence_model: str = "cnn",  # "nucleotide_transformer", "dnabert2", or "cnn"
        sequence_dim: int = 256,
        max_sequence_length: int = 6000,
        freeze_sequence_encoder: bool = True,
        # Topology encoder config
        topology_input_dim: int = 16,
        topology_hidden_dim: int = 256,
        topology_output_dim: int = 256,
        num_topology_levels: int = 4,
        # Fragile site encoder config
        num_fragile_sites: int = 100,
        fragile_hidden_dim: int = 128,
        fragile_output_dim: int = 64,
        # Fusion config
        fusion_type: str = "bottleneck",  # "bottleneck", "hierarchical", or "gated"
        fusion_dim: int = 256,
        num_bottleneck_tokens: int = 16,
        # Prediction heads config
        num_oncogenes: int = 20,
        use_uncertainty: bool = False,
        # General config
        dropout: float = 0.1,
    ):
        """
        Initialize ecDNA-Former.

        Args:
            sequence_model: Type of sequence encoder
            sequence_dim: Sequence embedding dimension
            max_sequence_length: Maximum sequence length
            freeze_sequence_encoder: Whether to freeze pre-trained weights
            topology_input_dim: Input dimension for topology encoder
            topology_hidden_dim: Hidden dimension for topology encoder
            topology_output_dim: Output dimension for topology encoder
            num_topology_levels: Number of hierarchical levels
            num_fragile_sites: Maximum fragile sites to consider
            fragile_hidden_dim: Hidden dimension for fragile site encoder
            fragile_output_dim: Output dimension for fragile site encoder
            fusion_type: Type of cross-modal fusion
            fusion_dim: Dimension of fused representation
            num_bottleneck_tokens: Number of bottleneck tokens for fusion
            num_oncogenes: Number of oncogenes to predict
            use_uncertainty: Whether to use uncertainty estimation
            dropout: Dropout rate
        """
        super().__init__()

        self.use_uncertainty = use_uncertainty

        # === Module 1: Sequence Encoder ===
        self.sequence_encoder = SequenceEncoder(
            model_name=sequence_model,
            pretrained=(sequence_model != "cnn"),
            hidden_dim=sequence_dim,
            output_dim=sequence_dim,
            max_length=max_sequence_length,
            freeze_encoder=freeze_sequence_encoder,
        )

        # === Module 2: Topology Encoder ===
        self.topology_encoder = TopologyEncoder(
            input_dim=topology_input_dim,
            hidden_dim=topology_hidden_dim,
            output_dim=topology_output_dim,
            num_levels=num_topology_levels,
            dropout=dropout,
        )

        # === Module 3: Fragile Site Encoder ===
        self.fragile_encoder = FragileSiteEncoder(
            num_fragile_sites=num_fragile_sites,
            hidden_dim=fragile_hidden_dim,
            output_dim=fragile_output_dim,
        )

        # === Module 4: Copy Number Encoder ===
        self.cn_encoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

        # === Cross-Modal Fusion ===
        modality_dims = {
            "sequence": sequence_dim,
            "topology": topology_output_dim,
            "fragile": fragile_output_dim,
            "copy_number": 64,
        }

        if fusion_type == "bottleneck":
            self.fusion = CrossModalFusion(
                modality_dims=modality_dims,
                bottleneck_dim=fusion_dim,
                output_dim=fusion_dim,
                num_bottleneck_tokens=num_bottleneck_tokens,
                dropout=dropout,
            )
        elif fusion_type == "hierarchical":
            self.fusion = HierarchicalFusion(
                sequence_dim=sequence_dim,
                topology_dim=topology_output_dim,
                fragile_dim=fragile_output_dim + 64,  # Include CN
                hidden_dim=fusion_dim,
                output_dim=fusion_dim,
                dropout=dropout,
            )
            self.fusion_type = "hierarchical"
        else:
            from .fusion import GatedFusion
            self.fusion = GatedFusion(
                modality_dims=modality_dims,
                hidden_dim=fusion_dim,
                output_dim=fusion_dim,
                dropout=dropout,
            )

        self.fusion_type = fusion_type

        # === Prediction Heads ===
        self.formation_head = FormationHead(
            input_dim=fusion_dim,
            hidden_dim=fusion_dim,
            dropout=dropout,
        )

        self.oncogene_head = OncogeneHead(
            input_dim=fusion_dim,
            num_oncogenes=num_oncogenes,
            hidden_dim=fusion_dim,
            dropout=dropout,
        )

        if use_uncertainty:
            self.uncertainty_head = UncertaintyHead(
                input_dim=fusion_dim,
                hidden_dim=fusion_dim,
                dropout=dropout,
            )

    def forward(
        self,
        # Sequence inputs
        sequences: Optional[torch.Tensor] = None,
        sequence_mask: Optional[torch.Tensor] = None,
        sequence_features: Optional[torch.Tensor] = None,
        # Topology inputs
        node_features: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        topology_features: Optional[torch.Tensor] = None,
        # Fragile site inputs
        fragile_site_features: Optional[torch.Tensor] = None,
        query_positions: Optional[torch.Tensor] = None,
        fragile_positions: Optional[torch.Tensor] = None,
        fragile_types: Optional[torch.Tensor] = None,
        fragile_chromosomes: Optional[torch.Tensor] = None,
        query_chromosomes: Optional[torch.Tensor] = None,
        # Copy number inputs
        copy_number_features: Optional[torch.Tensor] = None,
        # Control flags
        is_circular: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ecDNA-Former.

        Args:
            sequences: DNA sequences [batch, seq_len]
            sequence_mask: Sequence attention mask
            sequence_features: Pre-computed sequence features
            node_features: Hi-C graph node features
            edge_index: Hi-C graph edge indices
            edge_attr: Hi-C graph edge attributes
            batch: Graph batch assignment
            topology_features: Pre-computed topology features
            fragile_site_features: Pre-computed fragile site features
            query_positions: Query region positions [batch, 2]
            fragile_positions: Fragile site positions
            fragile_types: Fragile site types
            fragile_chromosomes: Fragile site chromosomes
            query_chromosomes: Query chromosomes
            copy_number_features: Copy number features
            is_circular: Whether regions are circular
            return_embeddings: Whether to return intermediate embeddings

        Returns:
            Dictionary with predictions and optionally embeddings
        """
        # === Encode Sequences ===
        if sequence_features is not None:
            seq_emb = sequence_features
        elif sequences is not None:
            _, seq_emb = self.sequence_encoder(
                sequences, sequence_mask, is_circular
            )
        else:
            # Use placeholder
            batch_size = self._infer_batch_size(
                node_features, topology_features, copy_number_features
            )
            device = self._infer_device(
                node_features, topology_features, copy_number_features
            )
            seq_emb = torch.zeros(batch_size, 256, device=device)

        # === Encode Topology ===
        if topology_features is not None:
            topo_emb = topology_features
        elif node_features is not None and edge_index is not None:
            _, topo_emb = self.topology_encoder(
                node_features, edge_index, edge_attr, batch
            )
        else:
            batch_size = seq_emb.shape[0]
            topo_emb = torch.zeros(batch_size, 256, device=seq_emb.device)

        # === Encode Fragile Sites ===
        if fragile_site_features is not None:
            frag_emb = fragile_site_features
        elif query_positions is not None and fragile_positions is not None:
            frag_emb = self.fragile_encoder(
                query_positions=query_positions,
                fragile_site_positions=fragile_positions,
                fragile_site_types=fragile_types,
                fragile_site_chromosomes=fragile_chromosomes,
                query_chromosomes=query_chromosomes,
            )
        else:
            batch_size = seq_emb.shape[0]
            frag_emb = torch.zeros(batch_size, 64, device=seq_emb.device)

        # === Encode Copy Number ===
        if copy_number_features is not None:
            cn_emb = self.cn_encoder(copy_number_features)
        else:
            batch_size = seq_emb.shape[0]
            cn_emb = torch.zeros(batch_size, 64, device=seq_emb.device)

        # === Fuse Modalities ===
        if self.fusion_type == "hierarchical":
            fused = self.fusion(
                seq_emb, topo_emb,
                torch.cat([frag_emb, cn_emb], dim=-1)
            )
        else:
            fused = self.fusion({
                "sequence": seq_emb,
                "topology": topo_emb,
                "fragile": frag_emb,
                "copy_number": cn_emb,
            })

        # === Predictions ===
        formation_prob = self.formation_head(fused)
        oncogene_probs, cooccurrence = self.oncogene_head(fused)

        results = {
            "formation_probability": formation_prob,
            "oncogene_probabilities": oncogene_probs,
        }

        if cooccurrence is not None:
            results["oncogene_cooccurrence"] = cooccurrence

        if self.use_uncertainty:
            mean, std = self.uncertainty_head(fused)
            results["formation_mean"] = mean
            results["formation_std"] = std

        if return_embeddings:
            results["sequence_embedding"] = seq_emb
            results["topology_embedding"] = topo_emb
            results["fragile_embedding"] = frag_emb
            results["fused_embedding"] = fused

        return results

    def _infer_batch_size(self, *tensors) -> int:
        """Infer batch size from available tensors."""
        for t in tensors:
            if t is not None:
                return t.shape[0]
        return 1

    def _infer_device(self, *tensors) -> torch.device:
        """Infer device from available tensors."""
        for t in tensors:
            if t is not None:
                return t.device
        return torch.device("cpu")

    def get_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        formation_labels: torch.Tensor,
        oncogene_labels: Optional[torch.Tensor] = None,
        formation_weight: float = 1.0,
        oncogene_weight: float = 0.5,
        focal_gamma: float = 2.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            outputs: Model outputs
            formation_labels: Binary ecDNA formation labels
            oncogene_labels: Multi-label oncogene labels
            formation_weight: Weight for formation loss
            oncogene_weight: Weight for oncogene loss
            focal_gamma: Gamma for focal loss

        Returns:
            Dictionary with loss components
        """
        losses = {}

        # Formation loss (focal loss for class imbalance)
        formation_logits = self.formation_head(
            outputs.get("fused_embedding", torch.zeros(1)),
            return_logits=True
        )
        formation_loss = self._focal_loss(
            formation_logits, formation_labels, gamma=focal_gamma
        )
        losses["formation_loss"] = formation_weight * formation_loss

        # Oncogene loss (only for ecDNA-positive samples)
        if oncogene_labels is not None:
            oncogene_loss = self.oncogene_head.get_loss(
                outputs["oncogene_probabilities"],
                oncogene_labels,
                formation_mask=formation_labels,
            )
            losses["oncogene_loss"] = oncogene_weight * oncogene_loss

        # Uncertainty loss if applicable
        if self.use_uncertainty:
            uncertainty_loss = self.uncertainty_head.get_loss(
                outputs.get("fused_embedding", torch.zeros(1)),
                formation_labels,
            )
            losses["uncertainty_loss"] = 0.1 * uncertainty_loss

        # Total loss
        losses["total_loss"] = sum(losses.values())

        return losses

    def _focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        gamma: float = 2.0,
        alpha: float = 0.25,
    ) -> torch.Tensor:
        """
        Focal loss for handling class imbalance.

        ecDNA-positive samples are ~30%, so we use focal loss
        to focus on hard examples.
        """
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction='none'
        )

        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)

        # Focal weight
        focal_weight = (1 - p_t) ** gamma

        # Alpha balance
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

        loss = alpha_t * focal_weight * bce

        return loss.mean()

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs) -> "ECDNAFormer":
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract config
        config = checkpoint.get("config", {})
        config.update(kwargs)

        # Create model
        model = cls(**config)

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])

        return model

    def save_pretrained(self, path: str, config: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": config or {},
        }
        torch.save(checkpoint, path)
