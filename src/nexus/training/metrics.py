"""
Training Metrics
================

Metrics tracking and visualization for model training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class EpochMetrics:
    """Metrics for a single training epoch."""
    
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float | None = None
    val_accuracy: float | None = None
    learning_rate: float | None = None
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TrainingMetrics:
    """
    Complete training metrics tracking.
    
    Tracks loss, accuracy, and other metrics across training epochs
    with support for visualization and export.
    """
    
    model_name: str = "nexus-intent-classifier"
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime | None = None
    epochs: list[EpochMetrics] = field(default_factory=list)
    best_epoch: int = 0
    best_val_accuracy: float = 0.0
    
    # Hyperparameters
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    
    def add_epoch(self, metrics: EpochMetrics) -> None:
        """Add metrics for an epoch."""
        self.epochs.append(metrics)
        
        # Track best epoch
        if metrics.val_accuracy and metrics.val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = metrics.val_accuracy
            self.best_epoch = metrics.epoch
    
    def finish(self) -> None:
        """Mark training as complete."""
        self.end_time = datetime.utcnow()
    
    @property
    def total_duration_seconds(self) -> float:
        """Get total training duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return sum(e.duration_seconds for e in self.epochs)
    
    @property
    def train_losses(self) -> list[float]:
        """Get all training losses."""
        return [e.train_loss for e in self.epochs]
    
    @property
    def val_losses(self) -> list[float]:
        """Get all validation losses."""
        return [e.val_loss for e in self.epochs if e.val_loss is not None]
    
    @property
    def train_accuracies(self) -> list[float]:
        """Get all training accuracies."""
        return [e.train_accuracy for e in self.epochs]
    
    @property
    def val_accuracies(self) -> list[float]:
        """Get all validation accuracies."""
        return [e.val_accuracy for e in self.epochs if e.val_accuracy is not None]
    
    @property
    def final_metrics(self) -> dict[str, Any]:
        """Get final training metrics."""
        if not self.epochs:
            return {}
        
        last = self.epochs[-1]
        return {
            "final_train_loss": last.train_loss,
            "final_train_accuracy": last.train_accuracy,
            "final_val_loss": last.val_loss,
            "final_val_accuracy": last.val_accuracy,
            "best_val_accuracy": self.best_val_accuracy,
            "best_epoch": self.best_epoch,
            "total_epochs": len(self.epochs),
            "total_duration_seconds": self.total_duration_seconds,
        }
    
    def to_dict(self) -> dict[str, Any]:
        """Export metrics to dictionary."""
        return {
            "model_name": self.model_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "hyperparameters": self.hyperparameters,
            "epochs": [
                {
                    "epoch": e.epoch,
                    "train_loss": e.train_loss,
                    "train_accuracy": e.train_accuracy,
                    "val_loss": e.val_loss,
                    "val_accuracy": e.val_accuracy,
                    "duration_seconds": e.duration_seconds,
                }
                for e in self.epochs
            ],
            **self.final_metrics,
        }
    
    def save_json(self, path: str) -> None:
        """Save metrics to JSON file."""
        import json
        from pathlib import Path as P
        
        P(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def plot(self, save_path: str | None = None) -> None:
        """
        Plot training curves.
        
        Args:
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Cannot plot metrics.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = list(range(1, len(self.epochs) + 1))
        
        # Loss plot
        axes[0].plot(epochs, self.train_losses, label="Train Loss", marker='o')
        if self.val_losses:
            axes[0].plot(epochs, self.val_losses, label="Val Loss", marker='o')
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(epochs, self.train_accuracies, label="Train Accuracy", marker='o')
        if self.val_accuracies:
            axes[1].plot(epochs, self.val_accuracies, label="Val Accuracy", marker='o')
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


class MetricsLogger:
    """
    Real-time metrics logging during training.
    
    Provides formatted output for training progress.
    """
    
    def __init__(self, total_epochs: int) -> None:
        """
        Initialize the logger.
        
        Args:
            total_epochs: Total number of epochs
        """
        self.total_epochs = total_epochs
        self._width = 80
    
    def log_epoch_start(self, epoch: int) -> None:
        """Log epoch start."""
        print(f"\n{'=' * self._width}")
        print(f"Epoch {epoch}/{self.total_epochs}")
        print('=' * self._width)
    
    def log_epoch_end(self, metrics: EpochMetrics) -> None:
        """Log epoch end with metrics."""
        parts = [
            f"Loss: {metrics.train_loss:.4f}",
            f"Acc: {metrics.train_accuracy:.2%}",
        ]
        
        if metrics.val_loss is not None:
            parts.append(f"Val Loss: {metrics.val_loss:.4f}")
        
        if metrics.val_accuracy is not None:
            parts.append(f"Val Acc: {metrics.val_accuracy:.2%}")
        
        parts.append(f"Time: {metrics.duration_seconds:.1f}s")
        
        print(" | ".join(parts))
    
    def log_training_complete(self, metrics: TrainingMetrics) -> None:
        """Log training completion summary."""
        print(f"\n{'=' * self._width}")
        print("Training Complete!")
        print('=' * self._width)
        
        final = metrics.final_metrics
        print(f"Best Validation Accuracy: {final.get('best_val_accuracy', 0):.2%} (Epoch {final.get('best_epoch', 0)})")
        print(f"Total Duration: {final.get('total_duration_seconds', 0):.1f}s")
        print('=' * self._width)
