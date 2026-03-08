"""
Topology Encoder for ecDNA-Former.

Encodes chromatin topology from Hi-C contact maps using
hierarchical graph neural networks.

Key innovations:
- Multi-resolution encoding (compartment, TAD, loop, enhancer-promoter)
- Hierarchical graph attention
- Circular topology awareness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv, GCNConv, GraphNorm, global_mean_pool, global_add_pool
)
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple, List, Dict
import math


class TopologyEncoder(nn.Module):
    """
    Multi-resolution topology encoder for Hi-C data.

    Encodes chromatin contact maps at multiple scales:
    - Compartment level (A/B compartments, ~1Mb)
    - TAD level (topologically associating domains, ~500kb)
    - Loop level (chromatin loops, ~50kb)
    - Enhancer-promoter level (regulatory contacts, ~10kb)
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_levels: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize topology encoder.

        Args:
            input_dim: Node feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output embedding dimension
            num_levels: Number of hierarchical levels
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.num_levels = num_levels
        self.hidden_dim = hidden_dim

        # Level-specific encoders
        self.level_encoders = nn.ModuleList([
            HierarchicalGraphTransformer(
                input_dim=input_dim if i == 0 else hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=2,
                dropout=dropout,
            )
            for i in range(num_levels)
        ])

        # Inter-level connections (learned)
        self.level_connections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_levels - 1)
        ])

        # Level aggregation
        self.level_attention = nn.Sequential(
            nn.Linear(hidden_dim * num_levels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_levels),
            nn.Softmax(dim=-1),
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Resolution embeddings (learnable)
        self.resolution_embeddings = nn.Embedding(num_levels, hidden_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        level_assignments: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode topology at multiple resolutions.

        Args:
            node_features: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim]
            batch: Batch assignment [num_nodes]
            level_assignments: Node-to-level assignments per level

        Returns:
            Tuple of:
                - Node embeddings [num_nodes, output_dim]
                - Graph embedding [batch_size, output_dim]
        """
        # Encode at each level
        level_outputs = []
        current_features = node_features

        for i, encoder in enumerate(self.level_encoders):
            # Add resolution embedding
            res_emb = self.resolution_embeddings(
                torch.full((current_features.shape[0],), i, device=current_features.device)
            )
            if current_features.shape[-1] == res_emb.shape[-1]:
                level_input = current_features + res_emb
            else:
                level_input = current_features

            # Encode at this level
            level_out, _ = encoder(level_input, edge_index, edge_attr, batch)
            level_outputs.append(level_out)

            # Pass to next level with connection
            if i < len(self.level_connections):
                current_features = self.level_connections[i](level_out)

        # Stack level outputs
        stacked = torch.stack(level_outputs, dim=-1)  # [N, H, L]

        # Attention over levels
        concat_levels = stacked.view(stacked.shape[0], -1)  # [N, H*L]
        level_weights = self.level_attention(concat_levels)  # [N, L]
        level_weights = level_weights.unsqueeze(1)  # [N, 1, L]

        # Weighted combination
        combined = (stacked * level_weights).sum(dim=-1)  # [N, H]

        # Project to output
        node_embeddings = self.output_proj(combined)

        # Graph-level pooling
        if batch is not None:
            graph_embedding = global_mean_pool(node_embeddings, batch)
        else:
            graph_embedding = node_embeddings.mean(dim=0, keepdim=True)

        return node_embeddings, graph_embedding


class HierarchicalGraphTransformer(nn.Module):
    """
    Graph Transformer for a single resolution level.

    Uses Graph Attention Network v2 with multi-head attention
    and skip connections.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
    ):
        """
        Initialize graph transformer.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            edge_dim: Edge feature dimension (optional)
        """
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            # GATv2 layer
            self.layers.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=True,
                    concat=True,
                )
            )
            self.norms.append(GraphNorm(hidden_dim))

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = GraphNorm(hidden_dim)

        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through graph transformer.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim]
            batch: Batch assignment [num_nodes]

        Returns:
            Tuple of:
                - Node embeddings [num_nodes, output_dim]
                - Graph embedding [batch_size, output_dim]
        """
        # Input projection
        h = self.input_proj(x)

        # Graph attention layers
        for layer, norm in zip(self.layers, self.norms):
            # Attention with residual
            h_attn = layer(h, edge_index, edge_attr=edge_attr)
            h = norm(h + self.dropout(h_attn), batch)

        # FFN with residual
        h_ffn = self.ffn(h)
        h = self.ffn_norm(h + h_ffn, batch)

        # Output projection
        node_out = self.output_proj(h)

        # Graph pooling
        if batch is not None:
            graph_out = global_mean_pool(node_out, batch)
        else:
            graph_out = node_out.mean(dim=0, keepdim=True)

        return node_out, graph_out


class HiCGraphBuilder:
    """
    Build graph representation from Hi-C contact matrices.

    Converts Hi-C matrices to PyTorch Geometric graph format
    with multi-resolution structure.
    """

    def __init__(
        self,
        resolutions: List[int] = [1000000, 500000, 100000, 50000],
        contact_threshold: float = 0.01,
        max_distance: int = 10000000,
    ):
        """
        Initialize graph builder.

        Args:
            resolutions: Resolution levels in bp
            contact_threshold: Minimum contact frequency for edge
            max_distance: Maximum genomic distance for edges
        """
        self.resolutions = resolutions
        self.contact_threshold = contact_threshold
        self.max_distance = max_distance

    def build_graph(
        self,
        contact_matrix: torch.Tensor,
        bin_positions: Optional[torch.Tensor] = None,
        resolution: int = 50000,
    ) -> Data:
        """
        Build graph from contact matrix.

        Args:
            contact_matrix: Hi-C contact matrix [N, N]
            bin_positions: Genomic positions of bins [N, 2] (start, end)
            resolution: Resolution of the matrix

        Returns:
            PyTorch Geometric Data object
        """
        n_bins = contact_matrix.shape[0]

        # Threshold and extract edges
        edges_i, edges_j = torch.where(contact_matrix > self.contact_threshold)

        # Filter by distance if positions available
        if bin_positions is not None:
            distances = torch.abs(
                bin_positions[edges_i, 0] - bin_positions[edges_j, 0]
            )
            distance_mask = distances < self.max_distance
            edges_i = edges_i[distance_mask]
            edges_j = edges_j[distance_mask]

        edge_index = torch.stack([edges_i, edges_j], dim=0)

        # Edge weights (contact frequency)
        edge_attr = contact_matrix[edges_i, edges_j].unsqueeze(-1)

        # Node features (could be augmented with genomic features)
        node_features = self._compute_node_features(
            contact_matrix, bin_positions, resolution
        )

        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=n_bins,
        )

    def _compute_node_features(
        self,
        contact_matrix: torch.Tensor,
        bin_positions: Optional[torch.Tensor],
        resolution: int,
    ) -> torch.Tensor:
        """Compute node features from contact patterns."""
        n_bins = contact_matrix.shape[0]
        features = []

        # Local contact density
        row_sums = contact_matrix.sum(dim=1)
        features.append(row_sums.unsqueeze(-1))

        # Local vs long-range contact ratio
        local_mask = torch.eye(n_bins, device=contact_matrix.device).bool()
        for i in range(1, 6):
            local_mask |= torch.diag(torch.ones(n_bins - i), i).bool().to(contact_matrix.device)
            local_mask |= torch.diag(torch.ones(n_bins - i), -i).bool().to(contact_matrix.device)

        local_contacts = (contact_matrix * local_mask.float()).sum(dim=1)
        long_range = row_sums - local_contacts
        ratio = local_contacts / (long_range + 1e-8)
        features.append(ratio.unsqueeze(-1))

        # Insulation score (boundary detection)
        insulation = self._compute_insulation_score(contact_matrix)
        features.append(insulation.unsqueeze(-1))

        # A/B compartment score (simplified)
        compartment = self._compute_compartment_score(contact_matrix)
        features.append(compartment.unsqueeze(-1))

        # Position embedding
        pos_emb = self._positional_embedding(n_bins, 12)
        features.append(pos_emb)

        return torch.cat(features, dim=-1)

    def _compute_insulation_score(
        self,
        contact_matrix: torch.Tensor,
        window: int = 5
    ) -> torch.Tensor:
        """Compute insulation score for TAD boundary detection."""
        n = contact_matrix.shape[0]
        insulation = torch.zeros(n, device=contact_matrix.device)

        for i in range(window, n - window):
            # Contacts across the bin
            upstream = contact_matrix[i-window:i, i-window:i].mean()
            downstream = contact_matrix[i:i+window, i:i+window].mean()
            cross = contact_matrix[i-window:i, i:i+window].mean()

            insulation[i] = cross / ((upstream + downstream) / 2 + 1e-8)

        return insulation

    def _compute_compartment_score(
        self,
        contact_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Compute simplified A/B compartment score."""
        # Use correlation with distance-normalized matrix
        n = contact_matrix.shape[0]

        # Distance normalize
        expected = torch.zeros_like(contact_matrix)
        for d in range(n):
            diag = torch.diagonal(contact_matrix, d)
            mean_val = diag.mean()
            expected.diagonal(d).fill_(mean_val)
            if d > 0:
                expected.diagonal(-d).fill_(mean_val)

        oe = contact_matrix / (expected + 1e-8)

        # Correlation matrix
        oe_centered = oe - oe.mean(dim=1, keepdim=True)
        corr = torch.mm(oe_centered, oe_centered.t())
        norms = torch.sqrt((oe_centered ** 2).sum(dim=1, keepdim=True))
        corr = corr / (norms @ norms.t() + 1e-8)

        # First eigenvector gives compartment
        try:
            _, eigvecs = torch.linalg.eigh(corr)
            compartment = eigvecs[:, -1]
        except Exception:
            compartment = corr.mean(dim=1)

        return compartment

    def _positional_embedding(
        self,
        n_bins: int,
        dim: int
    ) -> torch.Tensor:
        """Sinusoidal positional embedding."""
        positions = torch.arange(n_bins).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )

        pe = torch.zeros(n_bins, dim)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)

        return pe
