"""
Evaluation Metrics for ECLIPSE.

Provides:
- Classification metrics (AUROC, AUPRC, F1)
- Calibration metrics
- Regression metrics for dynamics
- Causal inference metrics
"""

import numpy as np
import torch
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error,
)


def compute_auroc(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Compute Area Under ROC Curve.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities

    Returns:
        AUROC score
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)


def compute_auprc(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Compute Area Under Precision-Recall Curve.

    More informative than AUROC for imbalanced data like ecDNA prediction.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities

    Returns:
        AUPRC score
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    if len(np.unique(y_true)) < 2:
        return y_true.mean()

    return average_precision_score(y_true, y_pred)


def compute_calibration_error(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    n_bins: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error.

    Measures how well predicted probabilities match observed frequencies.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        n_bins: Number of calibration bins

    Returns:
        Tuple of (ECE, bin_accuracies, bin_confidences)
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (y_pred >= bin_boundaries[i]) & (y_pred < bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_acc = y_true[in_bin].mean()
            bin_conf = y_pred[in_bin].mean()
            bin_count = in_bin.sum()
        else:
            bin_acc = 0
            bin_conf = (bin_boundaries[i] + bin_boundaries[i + 1]) / 2
            bin_count = 0

        bin_accuracies.append(bin_acc)
        bin_confidences.append(bin_conf)
        bin_counts.append(bin_count)

    bin_accuracies = np.array(bin_accuracies)
    bin_confidences = np.array(bin_confidences)
    bin_counts = np.array(bin_counts)

    # ECE: weighted average of |accuracy - confidence|
    total = bin_counts.sum()
    if total > 0:
        ece = (bin_counts * np.abs(bin_accuracies - bin_confidences)).sum() / total
    else:
        ece = 0.0

    return ece, bin_accuracies, bin_confidences


def compute_f1_multilabel(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    average: str = "macro",
) -> float:
    """
    Compute F1 score for multi-label classification.

    Used for oncogene prediction in ecDNA-Former.

    Args:
        y_true: True multi-label matrix [N, num_labels]
        y_pred: Predicted probabilities [N, num_labels]
        threshold: Threshold for positive prediction
        average: Averaging method ("macro", "micro", "weighted")

    Returns:
        F1 score
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    y_pred_binary = (y_pred >= threshold).astype(int)

    return f1_score(y_true, y_pred_binary, average=average, zero_division=0)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    auroc: float
    auprc: float
    ece: float
    f1: float
    precision: float
    recall: float
    mse: Optional[float] = None
    mae: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "auroc": self.auroc,
            "auprc": self.auprc,
            "ece": self.ece,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "mse": self.mse,
            "mae": self.mae,
        }


def compute_all_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
) -> EvaluationMetrics:
    """
    Compute all evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        threshold: Classification threshold

    Returns:
        EvaluationMetrics object
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    y_pred_binary = (y_pred >= threshold).astype(int)

    auroc = compute_auroc(y_true, y_pred)
    auprc = compute_auprc(y_true, y_pred)
    ece, _, _ = compute_calibration_error(y_true, y_pred)

    if len(np.unique(y_true)) >= 2:
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
    else:
        f1 = precision = recall = 0.0

    return EvaluationMetrics(
        auroc=auroc,
        auprc=auprc,
        ece=ece,
        f1=f1,
        precision=precision,
        recall=recall,
    )


def compute_dynamics_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Dict[str, float]:
    """
    Compute metrics for dynamics prediction (CircularODE).

    Args:
        y_true: True trajectories [batch, time]
        y_pred: Predicted trajectories [batch, time]
        mask: Mask for valid time points

    Returns:
        Dictionary of metrics
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    if mask is not None:
        mask = _to_numpy(mask).astype(bool)
        y_true_flat = y_true[mask]
        y_pred_flat = y_pred[mask]
    else:
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

    metrics = {
        "mse": mean_squared_error(y_true_flat, y_pred_flat),
        "mae": mean_absolute_error(y_true_flat, y_pred_flat),
        "rmse": np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)),
    }

    # Log-space metrics (better for copy number)
    log_true = np.log1p(y_true_flat)
    log_pred = np.log1p(np.maximum(y_pred_flat, 0))
    metrics["log_mse"] = mean_squared_error(log_true, log_pred)

    # Correlation
    if len(y_true_flat) > 1:
        correlation = np.corrcoef(y_true_flat, y_pred_flat)[0, 1]
        metrics["correlation"] = correlation if not np.isnan(correlation) else 0.0
    else:
        metrics["correlation"] = 0.0

    return metrics


def compute_causal_metrics(
    estimated_effects: Dict[str, float],
    true_effects: Optional[Dict[str, float]] = None,
    top_k: int = 20,
) -> Dict[str, float]:
    """
    Compute metrics for causal effect estimation (VulnCausal).

    Args:
        estimated_effects: Estimated causal effects per gene
        true_effects: True causal effects (if available from validation)
        top_k: Number of top genes to evaluate

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Sort genes by estimated effect
    sorted_genes = sorted(
        estimated_effects.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    # Top-k statistics
    top_genes = [g for g, _ in sorted_genes[:top_k]]
    top_effects = [e for _, e in sorted_genes[:top_k]]

    metrics["top_k_mean_effect"] = np.mean(np.abs(top_effects))
    metrics["top_k_max_effect"] = np.max(np.abs(top_effects))

    # If true effects available, compute overlap
    if true_effects is not None:
        sorted_true = sorted(
            true_effects.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        true_top_genes = set([g for g, _ in sorted_true[:top_k]])
        estimated_top_genes = set(top_genes)

        overlap = len(true_top_genes & estimated_top_genes)
        metrics["precision_at_k"] = overlap / top_k
        metrics["recall_at_k"] = overlap / len(true_top_genes) if true_top_genes else 0

        # Effect correlation for matched genes
        common_genes = set(estimated_effects.keys()) & set(true_effects.keys())
        if common_genes:
            est_vals = [estimated_effects[g] for g in common_genes]
            true_vals = [true_effects[g] for g in common_genes]
            metrics["effect_correlation"] = np.corrcoef(est_vals, true_vals)[0, 1]

    return metrics


def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
