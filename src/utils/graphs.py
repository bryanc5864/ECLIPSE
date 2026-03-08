"""
Graph Utilities for ECLIPSE.

Handles:
- Hi-C contact graph construction
- Graph feature computation
- Adjacency matrix normalization
"""

import numpy as np
import torch
from typing import Optional, Tuple, Dict, List
import math


def build_hic_graph(
    contact_matrix: np.ndarray,
    threshold: float = 0.01,
    max_distance: Optional[int] = None,
    bin_size: int = 50000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build graph from Hi-C contact matrix.

    Args:
        contact_matrix: Hi-C contact matrix [N, N]
        threshold: Minimum contact frequency for edge
        max_distance: Maximum genomic distance for edges (bins)
        bin_size: Size of each bin in bp

    Returns:
        Tuple of (edge_index, edge_weights, node_features)
    """
    n_bins = contact_matrix.shape[0]

    # Apply threshold
    edges_i, edges_j = np.where(contact_matrix > threshold)

    # Filter by distance if specified
    if max_distance is not None:
        max_bins = max_distance // bin_size
        distance_mask = np.abs(edges_i - edges_j) <= max_bins
        edges_i = edges_i[distance_mask]
        edges_j = edges_j[distance_mask]

    # Edge index
    edge_index = np.stack([edges_i, edges_j], axis=0)

    # Edge weights
    edge_weights = contact_matrix[edges_i, edges_j]

    # Node features
    node_features = compute_graph_features(contact_matrix)

    return edge_index, edge_weights, node_features


def compute_graph_features(
    contact_matrix: np.ndarray,
    feature_types: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Compute node features from contact matrix.

    Args:
        contact_matrix: Hi-C contact matrix
        feature_types: Types of features to compute

    Returns:
        Node features [N, F]
    """
    if feature_types is None:
        feature_types = [
            "degree", "local_density", "insulation",
            "compartment", "position"
        ]

    n_bins = contact_matrix.shape[0]
    features = []

    if "degree" in feature_types:
        # Weighted degree
        degree = contact_matrix.sum(axis=1)
        features.append(degree.reshape(-1, 1))

    if "local_density" in feature_types:
        # Local contact density
        window = 5
        local_density = np.zeros(n_bins)
        for i in range(n_bins):
            start = max(0, i - window)
            end = min(n_bins, i + window + 1)
            local_density[i] = contact_matrix[start:end, start:end].mean()
        features.append(local_density.reshape(-1, 1))

    if "insulation" in feature_types:
        # Insulation score (TAD boundary detection)
        insulation = compute_insulation_score(contact_matrix)
        features.append(insulation.reshape(-1, 1))

    if "compartment" in feature_types:
        # A/B compartment score
        compartment = compute_compartment_score(contact_matrix)
        features.append(compartment.reshape(-1, 1))

    if "position" in feature_types:
        # Positional encoding
        position = positional_encoding(n_bins, 8)
        features.append(position)

    return np.concatenate(features, axis=1)


def compute_insulation_score(
    contact_matrix: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    """
    Compute insulation score for TAD boundary detection.

    Args:
        contact_matrix: Hi-C contact matrix
        window: Window size for insulation calculation

    Returns:
        Insulation scores [N]
    """
    n = contact_matrix.shape[0]
    insulation = np.zeros(n)

    for i in range(window, n - window):
        # Contacts within upstream window
        upstream = contact_matrix[i-window:i, i-window:i].mean()
        # Contacts within downstream window
        downstream = contact_matrix[i:i+window, i:i+window].mean()
        # Contacts across the boundary
        cross = contact_matrix[i-window:i, i:i+window].mean()

        # Insulation score
        insulation[i] = cross / (0.5 * (upstream + downstream) + 1e-10)

    return insulation


def compute_compartment_score(
    contact_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute A/B compartment score using correlation analysis.

    Args:
        contact_matrix: Hi-C contact matrix

    Returns:
        Compartment scores [N] (positive = A, negative = B)
    """
    n = contact_matrix.shape[0]

    # Compute observed/expected matrix
    expected = np.zeros_like(contact_matrix)
    for d in range(n):
        diag_vals = np.diagonal(contact_matrix, d)
        mean_val = diag_vals.mean() if len(diag_vals) > 0 else 0
        np.fill_diagonal(expected[d:], mean_val)
        if d > 0:
            np.fill_diagonal(expected[:, d:], mean_val)

    oe = contact_matrix / (expected + 1e-10)

    # Compute correlation matrix
    oe_centered = oe - oe.mean(axis=1, keepdims=True)
    norms = np.sqrt((oe_centered ** 2).sum(axis=1, keepdims=True))
    oe_normalized = oe_centered / (norms + 1e-10)
    correlation = oe_normalized @ oe_normalized.T

    # First eigenvector gives compartment
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        compartment = eigenvectors[:, -1]
    except np.linalg.LinAlgError:
        compartment = correlation.mean(axis=1)

    return compartment


def positional_encoding(
    n_positions: int,
    d_model: int,
) -> np.ndarray:
    """
    Sinusoidal positional encoding.

    Args:
        n_positions: Number of positions
        d_model: Encoding dimension

    Returns:
        Positional encoding [n_positions, d_model]
    """
    position = np.arange(n_positions).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

    pe = np.zeros((n_positions, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe


def normalize_adjacency(
    adj: np.ndarray,
    method: str = "symmetric",
) -> np.ndarray:
    """
    Normalize adjacency matrix.

    Args:
        adj: Adjacency matrix
        method: Normalization method ("symmetric", "random_walk", "none")

    Returns:
        Normalized adjacency matrix
    """
    if method == "none":
        return adj

    # Add self-loops
    adj = adj + np.eye(adj.shape[0])

    # Degree matrix
    degree = adj.sum(axis=1)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0

    if method == "symmetric":
        # D^(-1/2) A D^(-1/2)
        d_mat = np.diag(degree_inv_sqrt)
        return d_mat @ adj @ d_mat

    elif method == "random_walk":
        # D^(-1) A
        degree_inv = np.power(degree, -1)
        degree_inv[np.isinf(degree_inv)] = 0
        d_mat = np.diag(degree_inv)
        return d_mat @ adj

    return adj


def torch_sparse_to_edge_index(
    adj: torch.Tensor,
    threshold: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert dense adjacency to sparse edge index format.

    Args:
        adj: Dense adjacency matrix [N, N]
        threshold: Minimum edge weight

    Returns:
        Tuple of (edge_index [2, E], edge_weight [E])
    """
    mask = adj > threshold
    edge_index = mask.nonzero().t()
    edge_weight = adj[mask]

    return edge_index, edge_weight


def compute_graph_laplacian(
    adj: np.ndarray,
    normalized: bool = True,
) -> np.ndarray:
    """
    Compute graph Laplacian.

    Args:
        adj: Adjacency matrix
        normalized: Whether to compute normalized Laplacian

    Returns:
        Laplacian matrix
    """
    degree = adj.sum(axis=1)

    if normalized:
        # L = I - D^(-1/2) A D^(-1/2)
        d_inv_sqrt = np.power(degree, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
        d_mat = np.diag(d_inv_sqrt)
        norm_adj = d_mat @ adj @ d_mat
        return np.eye(adj.shape[0]) - norm_adj
    else:
        # L = D - A
        return np.diag(degree) - adj
