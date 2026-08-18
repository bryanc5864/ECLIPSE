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
    """ecDNA-Former: Topological Deep Learning for ecDNA Formation Prediction."""

    def __init__(
        self,
        # sequence encoder config
        sequence_model: str = "cnn",  # "nucleotide_transformer", "dnabert2", or "cnn"
        sequence_dim: int = 256,
        max_sequence_length: int = 6000,
        freeze_sequence_encoder: bool = True,
        # topology encoder config
        topology_input_dim: int = 16,
        topology_hidden_dim: int = 256,
        topology_output_dim: int = 256,
        num_topology_levels: int = 4,
        # fragile site encoder config
        num_fragile_sites: int = 100,
        fragile_hidden_dim: int = 128,
        fragile_output_dim: int = 64,
        # fusion config
        fusion_type: str = "bottleneck",  # "bottleneck", "hierarchical", or "gated"
        fusion_dim: int = 256,
        num_bottleneck_tokens: int = 16,
        # prediction heads config
        num_oncogenes: int = 20,
        use_uncertainty: bool = False,
        # general config
        dropout: float = 0.1,
    ):
        super().__init__()

        self.use_uncertainty = use_uncertainty

        self.sequence_encoder = SequenceEncoder(
            model_name=sequence_model,
            pretrained=(sequence_model != "cnn"),
            hidden_dim=sequence_dim,
            output_dim=sequence_dim,
            max_length=max_sequence_length,
            freeze_encoder=freeze_sequence_encoder,
        )

        self.topology_encoder = TopologyEncoder(
            input_dim=topology_input_dim,
            hidden_dim=topology_hidden_dim,
            output_dim=topology_output_dim,
            num_levels=num_topology_levels,
            dropout=dropout,
        )

        self.fragile_encoder = FragileSiteEncoder(
            num_fragile_sites=num_fragile_sites,
            hidden_dim=fragile_hidden_dim,
            output_dim=fragile_output_dim,
        )

        self.cn_encoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

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
                fragile_dim=fragile_output_dim + 64,  # include CN
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
        # sequence inputs
        sequences: Optional[torch.Tensor] = None,
        sequence_mask: Optional[torch.Tensor] = None,
        sequence_features: Optional[torch.Tensor] = None,
        # topology inputs
        node_features: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        topology_features: Optional[torch.Tensor] = None,
        # fragile site inputs
        fragile_site_features: Optional[torch.Tensor] = None,
        query_positions: Optional[torch.Tensor] = None,
        fragile_positions: Optional[torch.Tensor] = None,
        fragile_types: Optional[torch.Tensor] = None,
        fragile_chromosomes: Optional[torch.Tensor] = None,
        query_chromosomes: Optional[torch.Tensor] = None,
        # copy number inputs
        copy_number_features: Optional[torch.Tensor] = None,
        # control flags
        is_circular: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        forward pass through ecDNA-Former.

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
        """
        if sequence_features is not None:
            seq_emb = sequence_features
        elif sequences is not None:
            _, seq_emb = self.sequence_encoder(
                sequences, sequence_mask, is_circular
            )
        else:
            # use placeholder
            batch_size = self._infer_batch_size(
                node_features, topology_features, copy_number_features
            )
            device = self._infer_device(
                node_features, topology_features, copy_number_features
            )
            seq_emb = torch.zeros(batch_size, 256, device=device)

        if topology_features is not None:
            topo_emb = topology_features
        elif node_features is not None and edge_index is not None:
            _, topo_emb = self.topology_encoder(
                node_features, edge_index, edge_attr, batch
            )
        else:
            batch_size = seq_emb.shape[0]
            topo_emb = torch.zeros(batch_size, 256, device=seq_emb.device)

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

        if copy_number_features is not None:
            cn_emb = self.cn_encoder(copy_number_features)
        else:
            batch_size = seq_emb.shape[0]
            cn_emb = torch.zeros(batch_size, 64, device=seq_emb.device)

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
        """compute training loss."""
        losses = {}

        # formation loss (focal loss for class imbalance)
        formation_logits = self.formation_head(
            outputs.get("fused_embedding", torch.zeros(1)),
            return_logits=True
        )
        formation_loss = self._focal_loss(
            formation_logits, formation_labels, gamma=focal_gamma
        )
        losses["formation_loss"] = formation_weight * formation_loss

        # oncogene loss (only for ecDNA-positive samples)
        if oncogene_labels is not None:
            oncogene_loss = self.oncogene_head.get_loss(
                outputs["oncogene_probabilities"],
                oncogene_labels,
                formation_mask=formation_labels,
            )
            losses["oncogene_loss"] = oncogene_weight * oncogene_loss

        # uncertainty loss if applicable
        if self.use_uncertainty:
            uncertainty_loss = self.uncertainty_head.get_loss(
                outputs.get("fused_embedding", torch.zeros(1)),
                formation_labels,
            )
            losses["uncertainty_loss"] = 0.1 * uncertainty_loss

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

        focal_weight = (1 - p_t) ** gamma

        # alpha balance
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

        loss = alpha_t * focal_weight * bce

        return loss.mean()

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs) -> "ECDNAFormer":
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # extract config
        config = checkpoint.get("config", {})
        config.update(kwargs)

        model = cls(**config)

        # load weights
        model.load_state_dict(checkpoint["model_state_dict"])

        return model

    def save_pretrained(self, path: str, config: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": config or {},
        }
        torch.save(checkpoint, path)
