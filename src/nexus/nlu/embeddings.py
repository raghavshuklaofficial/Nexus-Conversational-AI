"""
Sentence embedding engine using sentence-transformers.
Used for semantic similarity, finding similar texts, etc.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any, Sequence

import numpy as np
import structlog

from nexus.config import NLUConfig

logger = structlog.get_logger(__name__)


class EmbeddingEngine:
    """Generates sentence embeddings with optional caching and batch support."""
    
    def __init__(self, config: NLUConfig) -> None:
        self.config = config
        self._model = None
        self._loaded = False
        self._embedding_dim: int = 0
        
        # simple dict cache - good enough for our use case
        self._cache: dict[str, np.ndarray] = {}
        self._cache_max_size = config.embedding_model if hasattr(config, 'embedding_cache_size') else 10000
    
    async def load(self) -> None:
        """Load the embedding model."""
        if self._loaded:
            return
        
        logger.info("loading_embedding_engine", model=self.config.embedding_model)
        
        try:
            from sentence_transformers import SentenceTransformer
            
            loop = asyncio.get_event_loop()
            
            self._model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(
                    self.config.embedding_model,
                    device=self.config.device,
                )
            )
            
            # Get embedding dimension
            test_embedding = self._model.encode("test", convert_to_numpy=True)
            self._embedding_dim = len(test_embedding)
            
            self._loaded = True
            logger.info(
                "embedding_engine_loaded",
                dimension=self._embedding_dim,
            )
            
        except Exception as e:
            logger.error("embedding_engine_load_failed", error=str(e))
            raise
    
    async def embed(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Get embedding for a single text."""
        if not self._loaded:
            raise RuntimeError("Engine not loaded. Call load() first.")
        
        # Check cache
        if use_cache and text in self._cache:
            return self._cache[text]
        
        loop = asyncio.get_event_loop()
        
        embedding = await loop.run_in_executor(
            None,
            lambda: self._model.encode(text, convert_to_numpy=True)
        )
        
        # Update cache
        if use_cache:
            self._update_cache(text, embedding)
        
        return embedding
    
    async def embed_batch(
        self,
        texts: Sequence[str],
        use_cache: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Embed multiple texts at once. Returns (N, dim) array."""
        if not self._loaded:
            raise RuntimeError("Engine not loaded")
        
        # Check which texts need embedding
        embeddings = []
        texts_to_embed = []
        embed_indices = []
        
        for i, text in enumerate(texts):
            if use_cache and text in self._cache:
                embeddings.append((i, self._cache[text]))
            else:
                texts_to_embed.append(text)
                embed_indices.append(i)
        
        # Embed new texts
        if texts_to_embed:
            loop = asyncio.get_event_loop()
            
            new_embeddings = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    texts_to_embed,
                    convert_to_numpy=True,
                    batch_size=batch_size,
                    show_progress_bar=False,
                )
            )
            
            # Update cache and results
            for idx, (text, emb) in enumerate(zip(texts_to_embed, new_embeddings)):
                original_idx = embed_indices[idx]
                embeddings.append((original_idx, emb))
                if use_cache:
                    self._update_cache(text, emb)
        
        # Sort by original index and stack
        embeddings.sort(key=lambda x: x[0])
        return np.stack([emb for _, emb in embeddings])
    
    async def similarity(
        self,
        text1: str,
        text2: str,
        metric: str = "cosine",
    ) -> float:
        """Compute similarity between two texts."""
        emb1, emb2 = await asyncio.gather(
            self.embed(text1),
            self.embed(text2),
        )
        
        return self._compute_similarity(emb1, emb2, metric)
    
    async def find_similar(
        self,
        query: str,
        candidates: Sequence[str],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[tuple[int, str, float]]:
        """Find the top-k most similar texts from candidates."""
        query_emb = await self.embed(query)
        candidate_embs = await self.embed_batch(candidates)
        
        # Compute all similarities
        similarities = [
            self._compute_similarity(query_emb, cand_emb, "cosine")
            for cand_emb in candidate_embs
        ]
        
        # Sort and filter
        results = [
            (i, candidates[i], sim)
            for i, sim in enumerate(similarities)
            if sim >= threshold
        ]
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results[:top_k]
    
    def _compute_similarity(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
        metric: str,
    ) -> float:
        """Compute similarity between two embeddings."""
        if metric == "cosine":
            return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        elif metric == "euclidean":
            return float(1 / (1 + np.linalg.norm(emb1 - emb2)))
        elif metric == "dot":
            return float(np.dot(emb1, emb2))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _update_cache(self, key: str, value: np.ndarray) -> None:
        """Add to cache, evict oldest if full."""
        if len(self._cache) >= 10000:
            # TODO: proper LRU eviction, this is just FIFO for now
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = value
    
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self._embedding_dim
    
    def __repr__(self) -> str:
        return f"EmbeddingEngine(loaded={self._loaded}, dim={self._embedding_dim})"
