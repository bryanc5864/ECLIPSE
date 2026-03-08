"""
Trainers for ECLIPSE modules.

Provides training loops with:
- Automatic mixed precision
- Gradient accumulation
- Learning rate scheduling
- Checkpointing
- Logging (WandB integration)
- CSV batch logging
- Comprehensive validation metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable, Any, Tuple
from pathlib import Path
import logging
from tqdm import tqdm
import json
import csv
import time
from datetime import datetime
from abc import ABC, abstractmethod
import numpy as np

try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        precision_score, recall_score, accuracy_score,
        balanced_accuracy_score, matthews_corrcoef,
        confusion_matrix, classification_report,
        mean_squared_error, mean_absolute_error, r2_score
    )
    from sklearn.calibration import calibration_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Base trainer class with common functionality."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 100,
        use_wandb: bool = False,
        classification_threshold: float = 0.35,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer (created if None)
            scheduler: Learning rate scheduler
            device: Device to train on
            mixed_precision: Use automatic mixed precision
            gradient_accumulation_steps: Steps for gradient accumulation
            max_grad_norm: Maximum gradient norm for clipping
            checkpoint_dir: Directory for checkpoints
            log_interval: Steps between logging
            use_wandb: Use Weights & Biases logging
            classification_threshold: Threshold for binary classification (lower for imbalanced data)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.use_wandb = use_wandb
        self.classification_threshold = classification_threshold

        # Optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)  # Lower LR for stability
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda') if mixed_precision else None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # WandB
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                logger.warning("wandb not installed, disabling")
                self.use_wandb = False

        # CSV Logging
        self.log_dir = self.checkpoint_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.train_start_time = None

        # Initialize CSV files
        self._init_csv_logging()

        # Store predictions/labels for validation metrics
        self.val_predictions = []
        self.val_labels = []
        self.val_probabilities = []

    def _init_csv_logging(self):
        """Initialize CSV logging files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Batch log file
        self.batch_log_file = self.log_dir / f"batch_log_{timestamp}.csv"
        self.batch_csv_headers = [
            "timestamp", "epoch", "batch", "global_step",
            "total_loss", "learning_rate", "grad_norm",
            "batch_size", "samples_seen", "batches_per_sec",
            "gpu_memory_mb", "loss_component_1", "loss_component_2",
            "loss_component_3", "loss_component_4"
        ]
        with open(self.batch_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.batch_csv_headers)

        # Epoch validation log file
        self.val_log_file = self.log_dir / f"validation_log_{timestamp}.csv"
        self.val_csv_headers = [
            "timestamp", "epoch", "global_step",
            "val_loss", "val_loss_std",
            "auroc", "auprc", "f1_score", "precision", "recall",
            "accuracy", "balanced_accuracy", "mcc",
            "specificity", "npv", "fpr", "fnr",
            "tp", "fp", "tn", "fn",
            "calibration_error", "brier_score",
            "train_loss", "train_val_gap", "epoch_time_sec"
        ]
        with open(self.val_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.val_csv_headers)

        # Final evaluation log file
        self.final_log_file = self.log_dir / f"final_evaluation_{timestamp}.csv"

        logger.info(f"CSV logs initialized at {self.log_dir}")

    def _log_batch_to_csv(
        self,
        epoch: int,
        batch_idx: int,
        losses: Dict[str, torch.Tensor],
        grad_norm: float,
        batch_size: int,
        batch_time: float,
    ):
        """Log batch metrics to CSV."""
        # Get GPU memory if available
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024

        # Get learning rate
        lr = self.optimizer.param_groups[0]['lr']

        # Extract loss components
        loss_components = []
        for k, v in losses.items():
            if k != "total_loss":
                loss_components.append(v.item() if isinstance(v, torch.Tensor) else v)
        while len(loss_components) < 4:
            loss_components.append(0.0)

        samples_seen = (epoch * len(self.train_loader) + batch_idx + 1) * batch_size
        batches_per_sec = 1.0 / batch_time if batch_time > 0 else 0

        row = [
            datetime.now().isoformat(),
            epoch,
            batch_idx,
            self.global_step,
            losses["total_loss"].item() if isinstance(losses["total_loss"], torch.Tensor) else losses["total_loss"],
            lr,
            grad_norm,
            batch_size,
            samples_seen,
            batches_per_sec,
            gpu_memory,
            *loss_components[:4]
        ]

        with open(self.batch_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    @abstractmethod
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute loss for a batch."""
        pass

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        epoch_start_time = time.time()
        batch_start_time = time.time()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_to_device(batch)

            # Forward pass with mixed precision
            if self.mixed_precision:
                with torch.amp.autocast('cuda'):
                    losses = self.compute_loss(batch)
                    loss = losses["total_loss"] / self.gradient_accumulation_steps
            else:
                losses = self.compute_loss(batch)
                loss = losses["total_loss"] / self.gradient_accumulation_steps

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            grad_norm = 0.0
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    # Compute gradient norm AFTER unscaling (before clipping)
                    for p in self.model.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.data.norm(2).item() ** 2
                    grad_norm = grad_norm ** 0.5
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Compute gradient norm before clipping
                    for p in self.model.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.data.norm(2).item() ** 2
                    grad_norm = grad_norm ** 0.5
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += losses["total_loss"].item()
            num_batches += 1

            # Batch timing
            batch_time = time.time() - batch_start_time

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{losses['total_loss'].item():.4f}",
                "grad": f"{grad_norm:.2f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

            # CSV batch logging (every batch)
            batch_size = batch.get("label", batch.get("labels", next(iter(batch.values())))).shape[0] if batch else 1
            self._log_batch_to_csv(
                epoch=self.epoch,
                batch_idx=batch_idx,
                losses=losses,
                grad_norm=grad_norm,
                batch_size=batch_size,
                batch_time=batch_time,
            )

            # WandB/console logging at intervals
            if self.global_step % self.log_interval == 0:
                self._log_metrics(losses, prefix="train")

            batch_start_time = time.time()

        self.epoch_time = time.time() - epoch_start_time
        return {"train_loss": total_loss / num_batches}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation with comprehensive metrics (10+)."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0
        all_losses = {}
        all_loss_values = []
        num_batches = 0

        # Collect predictions and labels
        all_predictions = []
        all_labels = []
        all_probabilities = []

        for batch in tqdm(self.val_loader, desc="Validation"):
            batch = self._move_to_device(batch)
            losses = self.compute_loss(batch)

            # Get predictions via overridable method
            probs, labels = self._get_validation_predictions(batch)
            if probs is not None:
                all_probabilities.extend(probs.flatten())
                all_predictions.extend((probs > self.classification_threshold).astype(int).flatten())
            if labels is not None:
                all_labels.extend(labels.flatten())

            total_loss += losses["total_loss"].item()
            all_loss_values.append(losses["total_loss"].item())
            for k, v in losses.items():
                if k not in all_losses:
                    all_losses[k] = 0
                all_losses[k] += v.item()
            num_batches += 1

        avg_losses = {f"val_{k}": v / num_batches for k, v in all_losses.items()}

        # Compute comprehensive metrics (10+)
        metrics = self._compute_validation_metrics(
            predictions=all_predictions,
            labels=all_labels,
            probabilities=all_probabilities,
            loss_values=all_loss_values,
        )
        metrics.update(avg_losses)

        # Log to CSV
        self._log_validation_to_csv(metrics)

        # Log to console/wandb
        self._log_metrics(metrics)

        return metrics

    def _get_validation_predictions(self, batch: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions and labels for validation metrics.
        Override in subclass for model-specific logic.
        Returns (probabilities, labels) as numpy arrays, or (None, None) if not applicable.
        """
        # Default: no predictions (subclasses should override)
        return None, None

    def _compute_validation_metrics(
        self,
        predictions: List,
        labels: List,
        probabilities: List,
        loss_values: List,
    ) -> Dict[str, float]:
        """Compute comprehensive validation metrics (10+)."""
        metrics = {}

        if not SKLEARN_AVAILABLE or len(labels) == 0:
            return metrics

        predictions = np.array(predictions)
        labels = np.array(labels)
        probabilities = np.array(probabilities)
        loss_values = np.array(loss_values)

        # Loss statistics
        metrics["val_loss_mean"] = float(np.mean(loss_values))
        metrics["val_loss_std"] = float(np.std(loss_values))

        # Only compute classification metrics if we have both classes
        if len(np.unique(labels)) < 2:
            logger.warning("Only one class in validation set, skipping some metrics")
            return metrics

        try:
            # 1. AUROC
            metrics["auroc"] = float(roc_auc_score(labels, probabilities))

            # 2. AUPRC
            metrics["auprc"] = float(average_precision_score(labels, probabilities))

            # 3. F1 Score
            metrics["f1_score"] = float(f1_score(labels, predictions, zero_division=0))

            # 4. Precision
            metrics["precision"] = float(precision_score(labels, predictions, zero_division=0))

            # 5. Recall (Sensitivity/TPR)
            metrics["recall"] = float(recall_score(labels, predictions, zero_division=0))

            # 6. Accuracy
            metrics["accuracy"] = float(accuracy_score(labels, predictions))

            # 7. Balanced Accuracy
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(labels, predictions))

            # 8. Matthews Correlation Coefficient
            metrics["mcc"] = float(matthews_corrcoef(labels, predictions))

            # 9-12. Confusion matrix derived metrics
            tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
            metrics["tp"] = int(tp)
            metrics["fp"] = int(fp)
            metrics["tn"] = int(tn)
            metrics["fn"] = int(fn)

            # 10. Specificity (TNR)
            metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

            # 11. Negative Predictive Value
            metrics["npv"] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

            # 12. False Positive Rate
            metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

            # 13. False Negative Rate
            metrics["fnr"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

            # 14. Brier Score
            metrics["brier_score"] = float(np.mean((probabilities - labels) ** 2))

            # 15. Calibration Error (ECE approximation)
            try:
                prob_true, prob_pred = calibration_curve(labels, probabilities, n_bins=10, strategy='uniform')
                metrics["calibration_error"] = float(np.mean(np.abs(prob_true - prob_pred)))
            except:
                metrics["calibration_error"] = 0.0

        except Exception as e:
            logger.warning(f"Error computing validation metrics: {e}")

        return metrics

    def _log_validation_to_csv(self, metrics: Dict[str, float]):
        """Log validation metrics to CSV."""
        row = [
            datetime.now().isoformat(),
            self.epoch,
            self.global_step,
            metrics.get("val_total_loss", metrics.get("val_loss_mean", 0)),
            metrics.get("val_loss_std", 0),
            metrics.get("auroc", 0),
            metrics.get("auprc", 0),
            metrics.get("f1_score", 0),
            metrics.get("precision", 0),
            metrics.get("recall", 0),
            metrics.get("accuracy", 0),
            metrics.get("balanced_accuracy", 0),
            metrics.get("mcc", 0),
            metrics.get("specificity", 0),
            metrics.get("npv", 0),
            metrics.get("fpr", 0),
            metrics.get("fnr", 0),
            metrics.get("tp", 0),
            metrics.get("fp", 0),
            metrics.get("tn", 0),
            metrics.get("fn", 0),
            metrics.get("calibration_error", 0),
            metrics.get("brier_score", 0),
            getattr(self, 'last_train_loss', 0),
            metrics.get("val_total_loss", 0) - getattr(self, 'last_train_loss', 0),
            getattr(self, 'epoch_time', 0),
        ]

        with open(self.val_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 5,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            num_epochs: Number of epochs
            early_stopping_patience: Patience for early stopping

        Returns:
            Training history
        """
        history = {"train_loss": [], "val_loss": [], "metrics": []}
        patience_counter = 0
        self.train_start_time = time.time()

        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Batch log: {self.batch_log_file}")
        logger.info(f"Validation log: {self.val_log_file}")

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            history["train_loss"].append(train_metrics["train_loss"])
            self.last_train_loss = train_metrics["train_loss"]

            # Validate
            val_metrics = self.validate()
            history["metrics"].append(val_metrics)

            val_loss_key = "val_total_loss" if "val_total_loss" in val_metrics else "val_loss_mean"
            if val_loss_key in val_metrics:
                history["val_loss"].append(val_metrics[val_loss_key])

                # Early stopping
                if val_metrics[val_loss_key] < self.best_val_loss:
                    self.best_val_loss = val_metrics[val_loss_key]
                    self.save_checkpoint("best.pt")
                    patience_counter = 0
                    logger.info(f"New best model saved (val_loss: {self.best_val_loss:.4f})")
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            # Save checkpoint
            self.save_checkpoint(f"epoch_{epoch}.pt")

            # Log epoch summary
            logger.info(
                f"Epoch {epoch}: train_loss={train_metrics['train_loss']:.4f}, "
                f"val_loss={val_metrics.get(val_loss_key, 0):.4f}, "
                f"auroc={val_metrics.get('auroc', 0):.4f}"
            )

        # Final evaluation with 20+ metrics
        final_metrics = self.final_evaluation()
        history["final_metrics"] = final_metrics

        total_time = time.time() - self.train_start_time
        logger.info(f"Training completed in {total_time/60:.1f} minutes")

        return history

    def final_evaluation(self) -> Dict[str, float]:
        """
        Comprehensive final evaluation with 20+ metrics.
        """
        if self.val_loader is None:
            return {}

        logger.info("Running final evaluation...")

        # Load best model
        try:
            self.load_checkpoint("best.pt")
            logger.info("Loaded best checkpoint for final evaluation")
        except:
            logger.warning("Could not load best checkpoint, using current model")

        self.model.eval()

        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_losses = []
        all_embeddings = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Final Evaluation"):
                batch = self._move_to_device(batch)

                losses = self.compute_loss(batch)
                all_losses.append(losses["total_loss"].item())

                # Get predictions via overridable method
                probs, labels = self._get_validation_predictions(batch)
                if probs is not None:
                    all_probabilities.extend(probs.flatten())
                    all_predictions.extend((probs > self.classification_threshold).astype(int).flatten())
                if labels is not None:
                    all_labels.extend(labels.flatten())

        # Compute 20+ final metrics
        metrics = self._compute_final_metrics(
            predictions=all_predictions,
            labels=all_labels,
            probabilities=all_probabilities,
            losses=all_losses,
            embeddings=all_embeddings,
        )

        # Save to CSV
        self._save_final_evaluation(metrics)

        return metrics

    def _compute_final_metrics(
        self,
        predictions: List,
        labels: List,
        probabilities: List,
        losses: List,
        embeddings: List,
    ) -> Dict[str, float]:
        """Compute 20+ final evaluation metrics."""
        metrics = {}

        predictions = np.array(predictions)
        labels = np.array(labels)
        probabilities = np.array(probabilities)
        losses = np.array(losses)

        # === Loss Metrics (4) ===
        metrics["final_loss_mean"] = float(np.mean(losses))
        metrics["final_loss_std"] = float(np.std(losses))
        metrics["final_loss_min"] = float(np.min(losses))
        metrics["final_loss_max"] = float(np.max(losses))

        if not SKLEARN_AVAILABLE or len(labels) == 0 or len(np.unique(labels)) < 2:
            return metrics

        try:
            # === Classification Metrics (8) ===
            metrics["final_auroc"] = float(roc_auc_score(labels, probabilities))
            metrics["final_auprc"] = float(average_precision_score(labels, probabilities))
            metrics["final_f1"] = float(f1_score(labels, predictions, zero_division=0))
            metrics["final_precision"] = float(precision_score(labels, predictions, zero_division=0))
            metrics["final_recall"] = float(recall_score(labels, predictions, zero_division=0))
            metrics["final_accuracy"] = float(accuracy_score(labels, predictions))
            metrics["final_balanced_accuracy"] = float(balanced_accuracy_score(labels, predictions))
            metrics["final_mcc"] = float(matthews_corrcoef(labels, predictions))

            # === Confusion Matrix Metrics (8) ===
            tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
            metrics["final_tp"] = int(tp)
            metrics["final_fp"] = int(fp)
            metrics["final_tn"] = int(tn)
            metrics["final_fn"] = int(fn)
            metrics["final_specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            metrics["final_npv"] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
            metrics["final_fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            metrics["final_fnr"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

            # === Calibration Metrics (3) ===
            metrics["final_brier_score"] = float(np.mean((probabilities - labels) ** 2))
            try:
                prob_true, prob_pred = calibration_curve(labels, probabilities, n_bins=10, strategy='uniform')
                metrics["final_calibration_error"] = float(np.mean(np.abs(prob_true - prob_pred)))
                metrics["final_calibration_max_error"] = float(np.max(np.abs(prob_true - prob_pred)))
            except:
                metrics["final_calibration_error"] = 0.0
                metrics["final_calibration_max_error"] = 0.0

            # === Threshold Analysis (4) ===
            for thresh in [0.3, 0.5, 0.7]:
                preds_at_thresh = (probabilities > thresh).astype(int)
                metrics[f"final_f1_at_{thresh}"] = float(f1_score(labels, preds_at_thresh, zero_division=0))

            # Optimal threshold (Youden's J)
            from sklearn.metrics import roc_curve
            fpr_curve, tpr_curve, thresholds = roc_curve(labels, probabilities)
            j_scores = tpr_curve - fpr_curve
            optimal_idx = np.argmax(j_scores)
            metrics["final_optimal_threshold"] = float(thresholds[optimal_idx])

            # === Distribution Metrics (3) ===
            pos_probs = probabilities[labels == 1]
            neg_probs = probabilities[labels == 0]
            if len(pos_probs) > 0 and len(neg_probs) > 0:
                metrics["final_pos_prob_mean"] = float(np.mean(pos_probs))
                metrics["final_neg_prob_mean"] = float(np.mean(neg_probs))
                metrics["final_prob_separation"] = float(np.mean(pos_probs) - np.mean(neg_probs))

            # === Sample Counts ===
            metrics["final_n_samples"] = int(len(labels))
            metrics["final_n_positive"] = int(np.sum(labels))
            metrics["final_n_negative"] = int(len(labels) - np.sum(labels))
            metrics["final_class_balance"] = float(np.mean(labels))

        except Exception as e:
            logger.warning(f"Error computing final metrics: {e}")

        return metrics

    def _save_final_evaluation(self, metrics: Dict[str, float]):
        """Save final evaluation metrics to CSV."""
        # Write as key-value pairs for readability
        with open(self.final_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["timestamp", datetime.now().isoformat()])
            writer.writerow(["total_epochs", self.epoch + 1])
            writer.writerow(["total_steps", self.global_step])
            writer.writerow(["best_val_loss", self.best_val_loss])
            writer.writerow(["training_time_sec", time.time() - self.train_start_time])
            for k, v in sorted(metrics.items()):
                writer.writerow([k, v])

        logger.info(f"Final evaluation saved to {self.final_log_file}")
        logger.info(f"Final AUROC: {metrics.get('final_auroc', 'N/A')}")
        logger.info(f"Final F1: {metrics.get('final_f1', 'N/A')}")

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, filename: str):
        """Load training checkpoint."""
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _log_metrics(self, metrics: Dict[str, Any], prefix: str = ""):
        """Log metrics."""
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        if self.use_wandb:
            self.wandb.log(metrics, step=self.global_step)

        logger.info(f"Step {self.global_step}: {metrics}")


class ECDNAFormerTrainer(BaseTrainer):
    """Trainer for ecDNA-Former (Module 1)."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,  # Weight for positive class (higher = more focus on minority)
        oncogene_weight: float = 0.5,
        **kwargs
    ):
        super().__init__(model, train_loader, val_loader, **kwargs)
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.oncogene_weight = oncogene_weight

        from .losses import FocalLoss
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def _get_validation_predictions(self, batch: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions for ECDNAFormer validation."""
        with torch.no_grad():
            outputs = self.model(
                sequence_features=batch.get("sequence_features"),
                topology_features=batch.get("topology_features"),
                fragile_site_features=batch.get("fragile_site_features"),
                copy_number_features=batch.get("copy_number_features"),
            )

        probs = None
        labels = None

        if "formation_probability" in outputs:
            probs = outputs["formation_probability"].detach().cpu().numpy()

        if "label" in batch:
            labels = batch["label"].detach().cpu().numpy()

        return probs, labels

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute ecDNA-Former loss."""
        # Forward pass
        outputs = self.model(
            sequence_features=batch.get("sequence_features"),
            topology_features=batch.get("topology_features"),
            fragile_site_features=batch.get("fragile_site_features"),
            copy_number_features=batch.get("copy_number_features"),
            return_embeddings=True,
        )

        losses = {}

        # Get logits for focal loss (FocalLoss expects logits, not probabilities)
        fused_emb = outputs.get("fused_embedding")
        if fused_emb is not None:
            formation_logits = self.model.formation_head(fused_emb, return_logits=True)
        else:
            # Fallback: use probability and convert (less accurate)
            formation_logits = torch.log(
                outputs["formation_probability"] / (1 - outputs["formation_probability"] + 1e-8) + 1e-8
            )

        # Formation prediction loss (focal)
        formation_loss = self.focal_loss(
            formation_logits,
            batch["label"].unsqueeze(-1),
        )
        losses["formation_loss"] = formation_loss

        # Oncogene prediction loss
        if "oncogene_labels" in batch:
            oncogene_loss = nn.functional.binary_cross_entropy(
                outputs["oncogene_probabilities"],
                batch["oncogene_labels"],
            )
            losses["oncogene_loss"] = self.oncogene_weight * oncogene_loss

        losses["total_loss"] = sum(losses.values())
        return losses


class CircularODETrainer(BaseTrainer):
    """Trainer for CircularODE (Module 2)."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        physics_weight: float = 0.1,
        **kwargs
    ):
        super().__init__(model, train_loader, val_loader, **kwargs)
        self.physics_weight = physics_weight

        from .losses import PhysicsInformedLoss
        self.physics_loss = PhysicsInformedLoss()

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute CircularODE loss."""
        # Forward pass
        outputs = self.model(
            initial_state=batch["initial_state"],
            time_points=batch["time_points"][0],  # Shared time points
            treatment_info=batch.get("treatment"),
        )

        # Physics-informed loss
        losses = self.physics_loss(
            predicted_trajectory=outputs["copy_number_trajectory"],
            observed_trajectory=batch["copy_numbers"],
            mask=batch.get("mask"),
        )

        return losses

    def _get_validation_predictions(self, batch: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """
        CircularODE is a regression task, not classification.
        Return None to skip classification metrics.
        Regression metrics are computed in compute_loss via PhysicsInformedLoss.
        """
        return None, None


class VulnCausalTrainer(BaseTrainer):
    """Trainer for VulnCausal (Module 3)."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        irm_weight: float = 1.0,
        dag_weight: float = 1.0,
        **kwargs
    ):
        super().__init__(model, train_loader, val_loader, **kwargs)
        self.irm_weight = irm_weight
        self.dag_weight = dag_weight

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute VulnCausal loss."""
        losses = self.model.get_loss(
            expression=batch["expression"],
            crispr_scores=batch["crispr_scores"],
            ecdna_labels=batch["ecdna_label"],
            environments=batch.get("covariates", torch.zeros(batch["expression"].shape[0], device=batch["expression"].device)),
        )

        return losses

    def _get_validation_predictions(self, batch: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """
        VulnCausal performs causal inference, not direct classification.
        The ecDNA labels are inputs, not targets for prediction.
        Return None to skip standard classification metrics.
        """
        return None, None


class ECLIPSETrainer(BaseTrainer):
    """Trainer for full ECLIPSE framework."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        module_weights: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        super().__init__(model, train_loader, val_loader, **kwargs)

        if module_weights is None:
            module_weights = {
                "former": 1.0,
                "dynamics": 1.0,
                "vuln": 1.0,
                "integration": 0.5,
            }
        self.module_weights = module_weights

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute ECLIPSE loss."""
        # Forward pass
        outputs = self.model(
            sequence_features=batch.get("sequence_features"),
            topology_features=batch.get("topology_features"),
            fragile_site_features=batch.get("fragile_site_features"),
            copy_number_features=batch.get("copy_number_features"),
            initial_state=batch.get("initial_state"),
            time_points=batch.get("time_points"),
            expression=batch.get("expression"),
            crispr_scores=batch.get("crispr_scores"),
            ecdna_labels=batch.get("ecdna_label"),
            run_all_modules=True,
        )

        losses = {}

        # Formation loss
        if "formation_probability" in outputs and "label" in batch:
            formation_loss = nn.functional.binary_cross_entropy(
                outputs["formation_probability"].squeeze(),
                batch["label"].float(),
            )
            losses["formation_loss"] = self.module_weights["former"] * formation_loss

        # Risk classification loss
        if "risk_logits" in outputs and "risk_level" in batch:
            risk_loss = nn.functional.cross_entropy(
                outputs["risk_logits"],
                batch["risk_level"],
            )
            losses["risk_loss"] = self.module_weights["integration"] * risk_loss

        losses["total_loss"] = sum(losses.values()) if losses else torch.tensor(0.0)
        return losses
