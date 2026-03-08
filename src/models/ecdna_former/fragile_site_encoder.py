"""
Fragile Site Encoder for ecDNA-Former.

Encodes proximity and features of chromosomal fragile sites,
which are known to be associated with ecDNA formation events.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class FragileSiteEncoder(nn.Module):
    """
    Encode chromosomal fragile site context for ecDNA prediction.

    Fragile sites are regions prone to breakage under replication stress
    and are associated with ecDNA formation. This module encodes:
    - Distance to nearest fragile sites
    - Fragile site type (common vs rare)
    - Associated gene information
    - Replication timing context
    """

    def __init__(
        self,
        num_fragile_sites: int = 100,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_heads: int = 4,
        max_distance: float = 10e6,  # 10 Mb
    ):
        """
        Initialize fragile site encoder.

        Args:
            num_fragile_sites: Maximum number of fragile sites to consider
            hidden_dim: Hidden dimension
            output_dim: Output embedding dimension
            num_heads: Number of attention heads
            max_distance: Maximum distance to consider (bp)
        """
        super().__init__()

        self.num_fragile_sites = num_fragile_sites
        self.hidden_dim = hidden_dim
        self.max_distance = max_distance

        # Fragile site embeddings
        self.site_type_embedding = nn.Embedding(4, hidden_dim // 4)  # CFS, RFS, etc.
        self.chromosome_embedding = nn.Embedding(25, hidden_dim // 4)  # 1-22, X, Y, MT

        # Distance encoding
        self.distance_encoder = DistanceEncoder(hidden_dim // 2)

        # Site feature projection
        self.site_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Attention over fragile sites
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )

        # Query for attention (learnable)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Output projection
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        query_positions: torch.Tensor,
        fragile_site_positions: torch.Tensor,
        fragile_site_types: torch.Tensor,
        fragile_site_chromosomes: torch.Tensor,
        query_chromosomes: torch.Tensor,
        fragile_site_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode fragile site context.

        Args:
            query_positions: Query genomic positions [batch, 2] (start, end)
            fragile_site_positions: Fragile site positions [batch, num_sites, 2]
            fragile_site_types: Fragile site types [batch, num_sites]
            fragile_site_chromosomes: Fragile site chromosomes [batch, num_sites]
            query_chromosomes: Query chromosomes [batch]
            fragile_site_mask: Mask for valid sites [batch, num_sites]

        Returns:
            Fragile site context embedding [batch, output_dim]
        """
        batch_size = query_positions.shape[0]

        # Get query midpoint
        query_midpoint = (query_positions[:, 0] + query_positions[:, 1]) / 2  # [B]
        query_midpoint = query_midpoint.unsqueeze(1)  # [B, 1]

        # Get fragile site midpoints
        site_midpoints = (
            fragile_site_positions[:, :, 0] + fragile_site_positions[:, :, 1]
        ) / 2  # [B, S]

        # Compute distances
        distances = torch.abs(site_midpoints - query_midpoint)  # [B, S]

        # Same chromosome mask
        same_chrom = (
            fragile_site_chromosomes == query_chromosomes.unsqueeze(1)
        ).float()

        # Distance encoding
        dist_features = self.distance_encoder(distances)  # [B, S, H/2]

        # Site type embedding
        type_emb = self.site_type_embedding(fragile_site_types)  # [B, S, H/4]

        # Chromosome embedding
        chrom_emb = self.chromosome_embedding(fragile_site_chromosomes)  # [B, S, H/4]

        # Combine features
        site_features = torch.cat([
            dist_features,
            type_emb,
            chrom_emb,
        ], dim=-1)  # [B, S, H]

        # Project
        site_features = self.site_projection(site_features)

        # Mask distant sites and different chromosomes
        distance_mask = (distances < self.max_distance) & (same_chrom > 0.5)
        if fragile_site_mask is not None:
            distance_mask = distance_mask & fragile_site_mask

        # Create attention mask (True = mask out)
        attn_mask = ~distance_mask

        # Expand query
        query = self.query.expand(batch_size, -1, -1)  # [B, 1, H]

        # Attention over fragile sites
        attended, _ = self.attention(
            query=query,
            key=site_features,
            value=site_features,
            key_padding_mask=attn_mask,
        )  # [B, 1, H]

        # Project to output
        output = self.output(attended.squeeze(1))  # [B, output_dim]

        return output


class DistanceEncoder(nn.Module):
    """
    Encode genomic distances with learned or fixed representations.

    Uses log-scaled distance bins with learned embeddings.
    """

    def __init__(
        self,
        output_dim: int,
        max_distance: float = 10e6,
        num_bins: int = 32,
    ):
        super().__init__()

        self.max_distance = max_distance
        self.num_bins = num_bins

        # Create log-spaced bins
        self.register_buffer(
            'bin_edges',
            torch.logspace(3, math.log10(max_distance), num_bins)
        )

        # Bin embeddings
        self.bin_embedding = nn.Embedding(num_bins + 1, output_dim)

        # Continuous distance encoding
        self.continuous_encoder = nn.Sequential(
            nn.Linear(1, output_dim // 2),
            nn.GELU(),
            nn.Linear(output_dim // 2, output_dim // 2),
        )

        # Combination
        self.combine = nn.Linear(output_dim + output_dim // 2, output_dim)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Encode distances.

        Args:
            distances: Genomic distances [batch, seq_len]

        Returns:
            Distance embeddings [batch, seq_len, output_dim]
        """
        # Discretize to bins
        bin_indices = torch.bucketize(distances, self.bin_edges)
        bin_emb = self.bin_embedding(bin_indices)

        # Continuous encoding (log-scaled)
        log_dist = torch.log1p(distances / 1000).unsqueeze(-1)  # Scale to kb
        cont_emb = self.continuous_encoder(log_dist)

        # Combine
        combined = torch.cat([bin_emb, cont_emb], dim=-1)
        return self.combine(combined)


class FragileSiteDatabase:
    """
    Database of chromosomal fragile sites for feature lookup.

    Provides interface to query fragile site information
    for any genomic region.
    """

    # Known common fragile sites (CFS) with associated genes
    COMMON_FRAGILE_SITES = {
        "FRA3B": {"chrom": "chr3", "start": 60400000, "end": 63000000, "gene": "FHIT"},
        "FRA16D": {"chrom": "chr16", "start": 78400000, "end": 79000000, "gene": "WWOX"},
        "FRA7G": {"chrom": "chr7", "start": 116000000, "end": 117000000, "gene": "CAV1"},
        "FRA6E": {"chrom": "chr6", "start": 162000000, "end": 163500000, "gene": "PARK2"},
        "FRA4F": {"chrom": "chr4", "start": 87000000, "end": 88500000, "gene": "GRID2"},
        "FRAXB": {"chrom": "chrX", "start": 6500000, "end": 7500000, "gene": "DMD"},
        "FRA2G": {"chrom": "chr2", "start": 168000000, "end": 169000000, "gene": ""},
        "FRA7H": {"chrom": "chr7", "start": 130000000, "end": 131000000, "gene": ""},
        "FRA1H": {"chrom": "chr1", "start": 61000000, "end": 62000000, "gene": ""},
        "FRA9E": {"chrom": "chr9", "start": 117000000, "end": 118000000, "gene": ""},
    }

    # Site type mapping
    SITE_TYPES = {
        "CFS": 0,  # Common fragile site
        "RFS": 1,  # Rare fragile site
        "ERFS": 2,  # Early replicating fragile site
        "unknown": 3,
    }

    # Chromosome mapping
    CHROM_TO_IDX = {f"chr{i}": i for i in range(1, 23)}
    CHROM_TO_IDX.update({"chrX": 23, "chrY": 24, "chrM": 0})

    def __init__(self, additional_sites: Optional[dict] = None):
        """Initialize with default and optional additional sites."""
        self.sites = dict(self.COMMON_FRAGILE_SITES)
        if additional_sites:
            self.sites.update(additional_sites)

    def get_nearby_sites(
        self,
        chrom: str,
        position: int,
        max_distance: int = 10000000
    ) -> List[dict]:
        """Get fragile sites near a genomic position."""
        nearby = []
        for site_id, info in self.sites.items():
            if info["chrom"] == chrom:
                midpoint = (info["start"] + info["end"]) / 2
                distance = abs(midpoint - position)
                if distance < max_distance:
                    nearby.append({
                        "site_id": site_id,
                        "distance": distance,
                        **info
                    })

        return sorted(nearby, key=lambda x: x["distance"])

    def get_tensors_for_batch(
        self,
        query_chroms: List[str],
        query_positions: List[Tuple[int, int]],
        max_sites: int = 20,
        device: str = "cpu"
    ) -> Tuple[torch.Tensor, ...]:
        """
        Get tensor representations for a batch of queries.

        Returns tensors suitable for FragileSiteEncoder forward pass.
        """
        batch_size = len(query_chroms)

        positions = torch.zeros(batch_size, max_sites, 2, device=device)
        types = torch.zeros(batch_size, max_sites, dtype=torch.long, device=device)
        chromosomes = torch.zeros(batch_size, max_sites, dtype=torch.long, device=device)
        masks = torch.zeros(batch_size, max_sites, dtype=torch.bool, device=device)

        for i, (chrom, (start, end)) in enumerate(zip(query_chroms, query_positions)):
            midpoint = (start + end) // 2
            nearby = self.get_nearby_sites(chrom, midpoint)

            for j, site in enumerate(nearby[:max_sites]):
                positions[i, j, 0] = site["start"]
                positions[i, j, 1] = site["end"]
                types[i, j] = self.SITE_TYPES.get("CFS", 3)
                chromosomes[i, j] = self.CHROM_TO_IDX.get(site["chrom"], 0)
                masks[i, j] = True

        query_chrom_idx = torch.tensor(
            [self.CHROM_TO_IDX.get(c, 0) for c in query_chroms],
            device=device
        )

        return positions, types, chromosomes, query_chrom_idx, masks
