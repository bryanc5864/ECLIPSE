"""
Neural Causal Discovery for VulnCausal.

Learns causal graph structure from observational data using
continuous optimization (NOTEARS framework).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import math


class NeuralCausalDiscovery(nn.Module):
    """
    Neural network for causal structure learning.

    Uses NOTEARS-style continuous optimization to learn
    a directed acyclic graph (DAG) representing causal relationships.

    Based on: Zheng et al., "DAGs with NO TEARS" (2018)
    """

    def __init__(
        self,
        num_variables: int,
        hidden_dim: int = 64,
        sparsity_penalty: float = 0.1,
        dag_penalty: float = 1.0,
    ):
        """
        Initialize causal discovery module.

        Args:
            num_variables: Number of variables in the graph
            hidden_dim: Hidden dimension for edge MLPs
            sparsity_penalty: L1 penalty for sparse graphs
            dag_penalty: Penalty weight for DAG constraint
        """
        super().__init__()

        self.num_variables = num_variables
        self.sparsity_penalty = sparsity_penalty
        self.dag_penalty = dag_penalty

        # Learnable adjacency matrix (soft, continuous)
        # A[i,j] represents edge from i to j
        self.adj_weights = nn.Parameter(torch.randn(num_variables, num_variables) * 0.01)

        # Edge strength networks (for nonlinear relationships)
        self.edge_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_variables)
        ])

        # Mask for no self-loops
        self.register_buffer(
            'no_self_loop_mask',
            1 - torch.eye(num_variables)
        )

    def get_adjacency_matrix(self) -> torch.Tensor:
        """
        Get current adjacency matrix.

        Returns:
            Adjacency matrix [num_variables, num_variables]
        """
        # Sigmoid to get probabilities
        adj = torch.sigmoid(self.adj_weights)

        # Mask self-loops
        adj = adj * self.no_self_loop_mask

        return adj

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict variable values given parents.

        Args:
            x: Variable values [batch, num_variables]

        Returns:
            Tuple of:
                - Predictions [batch, num_variables]
                - Adjacency matrix [num_variables, num_variables]
        """
        batch_size = x.shape[0]
        adj = self.get_adjacency_matrix()

        # For each variable, predict from parents
        predictions = torch.zeros_like(x)

        for i in range(self.num_variables):
            # Get parent weights
            parent_weights = adj[:, i]  # [num_variables]

            # Weighted sum of parent values
            parent_input = (x * parent_weights.unsqueeze(0)).sum(dim=1, keepdim=True)

            # Apply nonlinear transformation
            predictions[:, i] = self.edge_networks[i](parent_input).squeeze(-1)

        return predictions, adj

    def dag_constraint(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Compute DAG constraint using matrix exponential.

        h(A) = tr(e^A) - d = 0 iff A is a DAG

        Args:
            adj: Adjacency matrix

        Returns:
            DAG constraint value (0 if DAG)
        """
        d = self.num_variables

        # Element-wise square (for gradient flow)
        adj_sq = adj * adj

        # Matrix exponential approximation (faster than torch.matrix_exp)
        # e^A ≈ I + A + A²/2 + A³/6 + ...
        eye = torch.eye(d, device=adj.device)
        M = eye + adj_sq
        for k in range(2, 10):
            M = M + torch.matrix_power(adj_sq, k) / math.factorial(k)

        # Trace
        h = torch.trace(M) - d

        return h

    def get_loss(
        self,
        x: torch.Tensor,
        lambda_dag: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute NOTEARS-style loss.

        Args:
            x: Observed data [batch, num_variables]
            lambda_dag: Optional override for DAG penalty

        Returns:
            Loss dictionary
        """
        if lambda_dag is None:
            lambda_dag = self.dag_penalty

        # Forward pass
        predictions, adj = self.forward(x)

        # Reconstruction loss
        recon_loss = F.mse_loss(predictions, x)

        # Sparsity loss (L1)
        sparsity_loss = self.sparsity_penalty * adj.abs().sum()

        # DAG constraint
        dag_loss = lambda_dag * self.dag_constraint(adj)

        return {
            "reconstruction_loss": recon_loss,
            "sparsity_loss": sparsity_loss,
            "dag_loss": dag_loss,
            "total_loss": recon_loss + sparsity_loss + dag_loss,
        }

    def get_graph(self, threshold: float = 0.3) -> torch.Tensor:
        """
        Get thresholded binary adjacency matrix.

        Args:
            threshold: Edge probability threshold

        Returns:
            Binary adjacency matrix
        """
        adj = self.get_adjacency_matrix()
        return (adj > threshold).float()

    def get_parents(self, variable_idx: int, threshold: float = 0.3) -> List[int]:
        """
        Get parents of a variable.

        Args:
            variable_idx: Index of variable
            threshold: Edge threshold

        Returns:
            List of parent indices
        """
        adj = self.get_adjacency_matrix()
        parent_probs = adj[:, variable_idx]
        return (parent_probs > threshold).nonzero().squeeze(-1).tolist()

    def get_children(self, variable_idx: int, threshold: float = 0.3) -> List[int]:
        """
        Get children of a variable.

        Args:
            variable_idx: Index of variable
            threshold: Edge threshold

        Returns:
            List of children indices
        """
        adj = self.get_adjacency_matrix()
        child_probs = adj[variable_idx, :]
        return (child_probs > threshold).nonzero().squeeze(-1).tolist()

    def get_adjustment_set(
        self,
        treatment_idx: int,
        outcome_idx: int,
        threshold: float = 0.3,
    ) -> List[int]:
        """
        Get adjustment set for causal effect estimation.

        Uses backdoor criterion: adjust for parents of treatment
        that are not descendants of treatment.

        Args:
            treatment_idx: Treatment variable index
            outcome_idx: Outcome variable index
            threshold: Edge threshold

        Returns:
            List of variable indices to adjust for
        """
        adj = self.get_graph(threshold)

        # Get all parents of treatment
        treatment_parents = set(self.get_parents(treatment_idx, threshold))

        # Get descendants of treatment (BFS)
        descendants = set()
        queue = [treatment_idx]
        while queue:
            node = queue.pop(0)
            children = self.get_children(node, threshold)
            for child in children:
                if child not in descendants:
                    descendants.add(child)
                    queue.append(child)

        # Adjustment set: parents that are not descendants
        adjustment = treatment_parents - descendants - {treatment_idx, outcome_idx}

        return list(adjustment)


class CausalGraphPrior(nn.Module):
    """
    Encodes prior knowledge about causal structure.

    For ecDNA vulnerabilities, we have prior knowledge:
    - ecDNA → oncogene expression
    - ecDNA → pathway activity
    - pathway → cell viability
    """

    def __init__(
        self,
        num_variables: int,
        prior_edges: Optional[List[Tuple[int, int]]] = None,
        prior_strength: float = 1.0,
    ):
        """
        Initialize prior.

        Args:
            num_variables: Number of variables
            prior_edges: List of (from, to) edge tuples
            prior_strength: Weight for prior
        """
        super().__init__()

        self.prior_strength = prior_strength

        # Create prior matrix
        prior = torch.zeros(num_variables, num_variables)
        if prior_edges:
            for i, j in prior_edges:
                prior[i, j] = 1.0

        self.register_buffer('prior_matrix', prior)

    def prior_loss(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Compute prior loss.

        Encourages learned adjacency to match prior edges.

        Args:
            adj: Learned adjacency matrix

        Returns:
            Prior loss
        """
        # Binary cross-entropy with prior
        loss = F.binary_cross_entropy(
            adj, self.prior_matrix,
            reduction='none'
        )

        # Weight prior edges more heavily
        weights = 1 + self.prior_matrix * 9  # 10x weight for prior edges
        loss = (loss * weights).mean()

        return self.prior_strength * loss
