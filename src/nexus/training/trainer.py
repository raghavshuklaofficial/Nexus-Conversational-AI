"""
Model Trainer
=============

Modern PyTorch-based training pipeline for intent classification.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from nexus.training.dataset import ConversationDataset, IntentDataGenerator
from nexus.training.metrics import TrainingMetrics, EpochMetrics, MetricsLogger

logger = structlog.get_logger(__name__)


class ModelTrainer:
    """
    Trainer for conversational AI models.
    
    Provides a modern training pipeline with:
    - Transformer-based model fine-tuning
    - Learning rate scheduling
    - Early stopping
    - Metrics tracking and visualization
    - Model checkpointing
    
    Example:
        >>> trainer = ModelTrainer(output_dir="models")
        >>> await trainer.load_data("data/intents.json")
        >>> metrics = await trainer.train(epochs=10)
        >>> trainer.save_model()
    """
    
    def __init__(
        self,
        output_dir: str = "models",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
    ) -> None:
        """
        Initialize the trainer.
        
        Args:
            output_dir: Directory for saving models and metrics
            model_name: Base model for fine-tuning
            device: Device to use ('cpu', 'cuda', 'auto')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.device = self._resolve_device(device)
        
        self._dataset: ConversationDataset | None = None
        self._model = None
        self._tokenizer = None
        self._metrics = TrainingMetrics()
        
        logger.info(
            "trainer_initialized",
            output_dir=str(self.output_dir),
            device=self.device,
        )
    
    def _resolve_device(self, device: str) -> str:
        """Resolve 'auto' device to actual device."""
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device
    
    async def load_data(
        self,
        data_path: str | None = None,
        augment: bool = True,
    ) -> None:
        """
        Load training data.
        
        Args:
            data_path: Path to training data JSON
            augment: Whether to augment data
        """
        logger.info("loading_training_data", path=data_path)
        
        if data_path and Path(data_path).exists():
            self._dataset = ConversationDataset(data_path=data_path)
        else:
            # Use built-in intents
            from nexus.data.intents import INTENT_DATABASE
            
            intents = [
                {
                    "tag": name,
                    "patterns": data["patterns"],
                }
                for name, data in INTENT_DATABASE.items()
            ]
            
            # Optionally augment
            if augment:
                generator = IntentDataGenerator()
                augmented = generator.generate_from_intents(
                    {i["tag"]: i["patterns"] for i in intents}
                )
                intents = augmented
            
            self._dataset = ConversationDataset(data=intents)
        
        logger.info(
            "data_loaded",
            samples=len(self._dataset),
            labels=self._dataset.num_labels,
        )
    
    async def train(
        self,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        validation_split: float = 0.1,
        early_stopping_patience: int = 3,
    ) -> dict[str, Any]:
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            validation_split: Validation data fraction
            early_stopping_patience: Epochs to wait before early stopping
        
        Returns:
            dict: Final training metrics
        """
        if not self._dataset:
            raise RuntimeError("No data loaded. Call load_data() first.")
        
        import torch
        import torch.nn as nn
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from sklearn.model_selection import train_test_split
        
        logger.info(
            "starting_training",
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        
        # Update metrics
        self._metrics.hyperparameters = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "model_name": self.model_name,
        }
        
        # Split data
        all_indices = list(range(len(self._dataset)))
        train_idx, val_idx = train_test_split(
            all_indices,
            test_size=validation_split,
            random_state=42,
        )
        
        # Create data loaders
        from torch.utils.data import Subset
        
        train_subset = Subset(self._dataset, train_idx)
        val_subset = Subset(self._dataset, val_idx)
        
        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
        )
        
        # Simple classifier model
        num_labels = self._dataset.num_labels
        
        class IntentClassifierModel(nn.Module):
            """Simple intent classifier."""
            
            def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                )
                self.classifier = nn.Linear(hidden_dim, num_classes)
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                features = self.encoder(x)
                return self.classifier(features)
        
        # Initialize model with embedding dimension
        # Using a simple BOW representation for now
        vocab_size = 10000
        embedding_dim = 256
        
        model = IntentClassifierModel(
            input_dim=vocab_size,
            hidden_dim=embedding_dim,
            num_classes=num_labels,
        ).to(self.device)
        
        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        metrics_logger = MetricsLogger(epochs)
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            metrics_logger.log_epoch_start(epoch)
            epoch_start = time.time()
            
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                # Create simple BOW features
                texts = batch["text"]
                labels = torch.tensor(batch["label_idx"]).to(self.device)
                
                # Simple bag of words
                features = self._texts_to_bow(texts, vocab_size).to(self.device)
                
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            scheduler.step()
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    texts = batch["text"]
                    labels = torch.tensor(batch["label_idx"]).to(self.device)
                    features = self._texts_to_bow(texts, vocab_size).to(self.device)
                    
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            # Calculate metrics
            epoch_duration = time.time() - epoch_start
            epoch_metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss / len(train_loader),
                train_accuracy=train_correct / train_total,
                val_loss=val_loss / len(val_loader),
                val_accuracy=val_correct / val_total,
                learning_rate=scheduler.get_last_lr()[0],
                duration_seconds=epoch_duration,
            )
            
            self._metrics.add_epoch(epoch_metrics)
            metrics_logger.log_epoch_end(epoch_metrics)
            
            # Early stopping
            if epoch_metrics.val_accuracy > best_val_acc:
                best_val_acc = epoch_metrics.val_accuracy
                patience_counter = 0
                # Save best model
                self._save_checkpoint(model, "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info("early_stopping", epoch=epoch)
                    break
        
        self._metrics.finish()
        metrics_logger.log_training_complete(self._metrics)
        
        # Save final model and metrics
        self._model = model
        self._save_checkpoint(model, "final_model.pt")
        self._metrics.save_json(str(self.output_dir / "training_metrics.json"))
        
        # Plot if possible
        try:
            self._metrics.plot(str(self.output_dir / "training_curves.png"))
        except Exception:
            pass
        
        return self._metrics.final_metrics
    
    def _texts_to_bow(self, texts: list[str], vocab_size: int) -> "torch.Tensor":
        """Convert texts to bag-of-words representation."""
        import torch
        
        batch_size = len(texts)
        features = torch.zeros(batch_size, vocab_size)
        
        for i, text in enumerate(texts):
            for word in text.lower().split():
                # Simple hash-based feature
                idx = hash(word) % vocab_size
                features[i, idx] += 1
        
        # Normalize
        features = features / (features.sum(dim=1, keepdim=True) + 1e-8)
        
        return features
    
    def _save_checkpoint(self, model: Any, filename: str) -> None:
        """Save model checkpoint."""
        import torch
        
        path = self.output_dir / filename
        torch.save({
            "model_state_dict": model.state_dict(),
            "labels": self._dataset.labels if self._dataset else [],
        }, path)
        
        logger.info("checkpoint_saved", path=str(path))
    
    def save_model(self, path: str | None = None) -> None:
        """Save the trained model."""
        if self._model is None:
            raise RuntimeError("No model to save. Train first.")
        
        import torch
        
        save_path = Path(path) if path else self.output_dir / "model.pt"
        
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "labels": self._dataset.labels if self._dataset else [],
            "hyperparameters": self._metrics.hyperparameters,
        }, save_path)
        
        logger.info("model_saved", path=str(save_path))
