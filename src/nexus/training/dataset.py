"""
Dataset loading and preprocessing for training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from torch.utils.data import Dataset, DataLoader


class ConversationDataset(Dataset):
    """PyTorch Dataset wrapping intent classification samples."""
    
    def __init__(
        self,
        data_path: str | Path | None = None,
        data: list[dict[str, Any]] | None = None,
        tokenizer: Any = None,
        max_length: int = 128,
    ) -> None:
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.samples: list[dict[str, Any]] = []
        self.labels: list[str] = []
        self.label_to_idx: dict[str, int] = {}
        
        if data_path:
            self._load_from_file(Path(data_path))
        elif data:
            self._load_from_data(data)
    
    def _load_from_file(self, path: Path) -> None:
        """Load data from JSON file."""
        with open(path) as f:
            data = json.load(f)
        self._load_from_data(data.get("intents", data))
    
    def _load_from_data(self, data: list[dict[str, Any]]) -> None:
        """Load data from list of intents."""
        # Collect all labels
        all_labels = sorted(set(intent["tag"] for intent in data))
        self.label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
        self.labels = all_labels
        
        # Create samples
        for intent in data:
            tag = intent["tag"]
            label_idx = self.label_to_idx[tag]
            
            for pattern in intent.get("patterns", []):
                self.samples.append({
                    "text": pattern,
                    "label": tag,
                    "label_idx": label_idx,
                })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        
        result = {
            "text": sample["text"],
            "label": sample["label"],
            "label_idx": sample["label_idx"],
        }
        
        # Tokenize if tokenizer available
        if self.tokenizer:
            encoding = self.tokenizer(
                sample["text"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            result["input_ids"] = encoding["input_ids"].squeeze()
            result["attention_mask"] = encoding["attention_mask"].squeeze()
        
        return result
    
    @property
    def num_labels(self) -> int:
        """Get the number of unique labels."""
        return len(self.labels)
    
    def get_label_name(self, idx: int) -> str:
        """Get label name from index."""
        return self.labels[idx]
    
    def create_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        """Create a DataLoader for this dataset."""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )


class IntentDataGenerator:
    """Simple data augmentation — lowercase, capitalize, strip punctuation, etc."""
    
    def __init__(self) -> None:
        self._variations = {
            "please": ["please", "kindly", "could you", "would you"],
            "want": ["want", "need", "would like", "require"],
            "help": ["help", "assist", "support", "aid"],
        }
    
    def augment(
        self,
        pattern: str,
        num_variations: int = 3,
    ) -> list[str]:
        """Generate simple variations of a pattern."""
        variations = [pattern]
        
        # Simple augmentation strategies
        # 1. Lowercase
        variations.append(pattern.lower())
        
        # 2. Capitalize
        variations.append(pattern.capitalize())
        
        # 3. Remove punctuation
        import re
        no_punct = re.sub(r'[^\w\s]', '', pattern)
        if no_punct != pattern:
            variations.append(no_punct)
        
        # 4. Add please prefix
        variations.append(f"Please {pattern.lower()}")
        
        return variations[:num_variations + 1]
    
    def generate_from_intents(
        self,
        intents: dict[str, list[str]],
        augment: bool = True,
    ) -> list[dict[str, Any]]:
        """Build training samples from raw intent->patterns mapping."""
        result = []
        
        for intent_name, patterns in intents.items():
            all_patterns = []
            
            for pattern in patterns:
                if augment:
                    all_patterns.extend(self.augment(pattern))
                else:
                    all_patterns.append(pattern)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_patterns = []
            for p in all_patterns:
                if p.lower() not in seen:
                    seen.add(p.lower())
                    unique_patterns.append(p)
            
            result.append({
                "tag": intent_name,
                "patterns": unique_patterns,
            })
        
        return result
