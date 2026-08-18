"""
Cross-Modal Fusion for ecDNA-Former.

Fuses information from:
- Sequence encoder (DNA language model embeddings)
- Topology encoder (Hi-C graph embeddings)
- Fragile site encoder (chromosomal fragile site context)
Uses bottleneck cross-attention for efficient fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List


class CrossModalFusion(nn.Module):
    """
    Cross-modal fusion using bottleneck cross-attention.

    Efficiently fuses multiple modalities (sequence, topology, fragile sites)
    through a learned bottleneck representation.
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        bottleneck_dim: int = 128,
        output_dim: int = 256,
        num_heads: int = 8,
        num_bottleneck_tokens: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.modality_names = list(modality_dims.keys())
        self.bottleneck_dim = bottleneck_dim
        self.num_bottleneck_tokens = num_bottleneck_tokens

        # project each modality to common dimension
        self.modality_projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, bottleneck_dim),
                nn.LayerNorm(bottleneck_dim),
                nn.GELU(),
            )
            for name, dim in modality_dims.items()
        })

        # learnable bottleneck tokens
        self.bottleneck_tokens = nn.Parameter(
            torch.randn(1, num_bottleneck_tokens, bottleneck_dim)
        )

        # cross-attention: bottleneck attends to each modality
        self.cross_attention_layers = nn.ModuleDict({
            name: nn.MultiheadAttention(
                embed_dim=bottleneck_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for name in modality_dims.keys()
        })

        # layer norms for cross-attention
        self.cross_attn_norms = nn.ModuleDict({
            name: nn.LayerNorm(bottleneck_dim)
            for name in modality_dims.keys()
        })

        # Self-attention for bottleneck
        self.self_attention = nn.MultiheadAttention(
            embed_dim=bottleneck_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn_norm = nn.LayerNorm(bottleneck_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim * 4, bottleneck_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(bottleneck_dim)

        self.output_projection = nn.Sequential(
            nn.Linear(bottleneck_dim * num_bottleneck_tokens, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
        )

        self.modality_presence = nn.Embedding(len(modality_dims), bottleneck_dim)

    def forward(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        fuse multiple modalities.

        Args:
            modality_inputs: Dictionary of modality tensors
            - For sequence: [batch, seq_len, dim] or [batch, dim]
            - For others: [batch, dim]
            modality_masks: Optional masks for sequence modalities

        Returns:
            Fused representation [batch, output_dim]
        """
        batch_size = next(iter(modality_inputs.values())).shape[0]
        device = next(iter(modality_inputs.values())).device

        # initialize bottleneck
        bottleneck = self.bottleneck_tokens.expand(batch_size, -1, -1)

        # project and cross-attend to each modality
        for i, name in enumerate(self.modality_names):
            if name not in modality_inputs:
                continue

            x = modality_inputs[name]

            # project to common dimension
            x = self.modality_projections[name](x)

            # ensure 3D for attention
            if x.dim() == 2:
                x = x.unsqueeze(1)  # [B, 1, D]

            mask = None
            if modality_masks and name in modality_masks:
                mask = ~modality_masks[name]  # invert for attention

            # cross-attention: bottleneck queries modality
            attended, _ = self.cross_attention_layers[name](
                query=bottleneck,
                key=x,
                value=x,
                key_padding_mask=mask,
            )

            # add modality presence embedding
            presence_emb = self.modality_presence(
                torch.tensor([i], device=device)
            ).unsqueeze(0)
            attended = attended + presence_emb

            # residual connection with norm
            bottleneck = self.cross_attn_norms[name](bottleneck + attended)

        # Self-attention within bottleneck
        self_attended, _ = self.self_attention(
            query=bottleneck,
            key=bottleneck,
            value=bottleneck,
        )
        bottleneck = self.self_attn_norm(bottleneck + self_attended)

        ffn_out = self.ffn(bottleneck)
        bottleneck = self.ffn_norm(bottleneck + ffn_out)

        # flatten and project
        bottleneck_flat = bottleneck.view(batch_size, -1)
        output = self.output_projection(bottleneck_flat)

        return output


class GatedFusion(nn.Module):
    """
    alternative fusion using gating mechanism.

    Each modality contributes via a learned gate, allowing
    the model to weight modalities dynamically.
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.modality_names = list(modality_dims.keys())
        n_modalities = len(modality_dims)

        # project each modality
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            for name, dim in modality_dims.items()
        })

        # gating network
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * n_modalities, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_modalities),
            nn.Softmax(dim=-1),
        )

        # output projection
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        modality_inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        fuse modalities with gating.

        Args:
            modality_inputs: Dictionary of modality tensors [batch, dim]

        Returns:
            Fused representation [batch, output_dim]
        """
        # project all modalities
        projected = []
        for name in self.modality_names:
            if name in modality_inputs:
                x = modality_inputs[name]
                if x.dim() > 2:
                    x = x.mean(dim=1)  # pool if sequence
                projected.append(self.projections[name](x))
            else:
                # use zeros for missing modalities
                batch_size = next(iter(modality_inputs.values())).shape[0]
                device = next(iter(modality_inputs.values())).device
                projected.append(torch.zeros(batch_size, self.projections[name][0].out_features, device=device))

        # compute gates
        concat = torch.cat(projected, dim=-1)
        gates = self.gate_network(concat)  # [B, n_modalities]

        # weighted sum
        stacked = torch.stack(projected, dim=-1)  # [B, H, n_modalities]
        gated = (stacked * gates.unsqueeze(1)).sum(dim=-1)  # [B, H]

        return self.output(gated)


class HierarchicalFusion(nn.Module):
    """
    hierarchical fusion with progressive combination.

    Fuses modalities in a hierarchical manner:
    1. Sequence + Topology -> Genomic context
    2. Genomic context + Fragile sites -> Formation context
    """

    def __init__(
        self,
        sequence_dim: int,
        topology_dim: int,
        fragile_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # level 1: sequence + Topology
        self.genomic_fusion = nn.Sequential(
            nn.Linear(sequence_dim + topology_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # level 2: Genomic + Fragile sites
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dim + fragile_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # skip connections
        self.sequence_skip = nn.Linear(sequence_dim, output_dim)
        self.topology_skip = nn.Linear(topology_dim, output_dim)

        # final combination
        self.final = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(
        self,
        sequence_emb: torch.Tensor,
        topology_emb: torch.Tensor,
        fragile_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        hierarchical fusion of modalities.

        Args:
            sequence_emb: Sequence embedding [batch, seq_dim]
            topology_emb: Topology embedding [batch, topo_dim]
            fragile_emb: Fragile site embedding [batch, frag_dim]

        Returns:
            Fused representation [batch, output_dim]
        """
        # level 1
        genomic = self.genomic_fusion(
            torch.cat([sequence_emb, topology_emb], dim=-1)
        )

        # level 2
        context = self.context_fusion(
            torch.cat([genomic, fragile_emb], dim=-1)
        )

        # skip connections
        seq_skip = self.sequence_skip(sequence_emb)
        topo_skip = self.topology_skip(topology_emb)

        combined = torch.cat([context, seq_skip, topo_skip], dim=-1)
        return self.final(combined)
