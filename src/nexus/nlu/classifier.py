"""
Intent classification using sentence-transformers and cosine similarity.
Pre-computes embeddings for all known intents at load time,
then matches new inputs against them.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from nexus.config import NLUConfig
from nexus.core.response import IntentMatch

logger = structlog.get_logger(__name__)


class IntentClassifier:
    """
    Classifies user input into one of the predefined intents using
    semantic similarity with sentence-BERT embeddings.
    """
    
    def __init__(self, config: NLUConfig) -> None:
        self.config = config
        self._model = None
        self._intent_embeddings: dict[str, np.ndarray] = {}
        self._intent_metadata: dict[str, dict[str, Any]] = {}
        self._loaded = False
    
    async def load(self) -> None:
        """Load the sentence-transformer model and compute intent embeddings."""
        if self._loaded:
            return
        
        logger.info("loading_intent_classifier", model=self.config.intent_model)
        
        try:
            # lazy import to keep startup fast
            from sentence_transformers import SentenceTransformer
            
            # load model in threadpool so we don't block the event loop
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(
                    self.config.intent_model,
                    device=self.config.device,
                )
            )
            
            # Load or compute intent embeddings
            await self._load_intent_embeddings()
            
            self._loaded = True
            logger.info("intent_classifier_loaded")
            
        except Exception as e:
            logger.error("intent_classifier_load_failed", error=str(e))
            raise
    
    async def _load_intent_embeddings(self) -> None:
        """Compute mean embedding for each intent from its training patterns."""
        from nexus.data.intents import get_intent_patterns
        
        intent_patterns = get_intent_patterns()
        
        loop = asyncio.get_event_loop()
        
        for intent_name, data in intent_patterns.items():
            patterns = data["patterns"]
            
            # Compute embeddings for all patterns
            embeddings = await loop.run_in_executor(
                None,
                lambda p=patterns: self._model.encode(p, convert_to_numpy=True)
            )
            
            # store the mean embedding as the intent's "prototype"
            self._intent_embeddings[intent_name] = np.mean(embeddings, axis=0)
            self._intent_metadata[intent_name] = {
                "description": data.get("description", ""),
                "priority": data.get("priority", 0),
            }
    
    async def classify(
        self,
        text: str,
        top_k: int = 3,
    ) -> IntentMatch:
        """Classify input text and return the best matching intent."""
        if not self._loaded:
            raise RuntimeError("Classifier not loaded. Call load() first.")
        
        loop = asyncio.get_event_loop()
        
        # Encode input text
        text_embedding = await loop.run_in_executor(
            None,
            lambda: self._model.encode(text, convert_to_numpy=True)
        )
        
        # Compute similarities with all intents
        similarities: list[tuple[str, float]] = []
        
        for intent_name, intent_embedding in self._intent_embeddings.items():
            similarity = self._cosine_similarity(text_embedding, intent_embedding)
            similarities.append((intent_name, float(similarity)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get best match
        best_intent, best_confidence = similarities[0]
        
        # Check if confidence is below fallback threshold
        is_fallback = best_confidence < self.config.fallback_threshold
        
        if is_fallback:
            return IntentMatch(
                name="fallback",
                confidence=best_confidence,
                is_fallback=True,
            )
        
        return IntentMatch(
            name=best_intent,
            confidence=best_confidence,
            is_fallback=False,
        )
    
    async def classify_multi(
        self,
        text: str,
        threshold: float = 0.3,
    ) -> list[IntentMatch]:
        """Get all intents above a confidence threshold (for multi-intent scenarios)."""
        if not self._loaded:
            raise RuntimeError("Classifier not loaded")
        
        loop = asyncio.get_event_loop()
        
        text_embedding = await loop.run_in_executor(
            None,
            lambda: self._model.encode(text, convert_to_numpy=True)
        )
        
        results: list[IntentMatch] = []
        
        for intent_name, intent_embedding in self._intent_embeddings.items():
            similarity = self._cosine_similarity(text_embedding, intent_embedding)
            
            if similarity >= threshold:
                results.append(IntentMatch(
                    name=intent_name,
                    confidence=float(similarity),
                    is_fallback=False,
                ))
        
        return sorted(results, key=lambda x: x.confidence, reverse=True)
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def __repr__(self) -> str:
        return f"IntentClassifier(loaded={self._loaded}, intents={len(self._intent_embeddings)})"
