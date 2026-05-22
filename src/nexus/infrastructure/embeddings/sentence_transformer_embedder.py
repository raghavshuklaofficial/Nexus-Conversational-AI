"""
SentenceTransformer embedding adapter.

Implements EmbeddingPort using sentence-transformers.
CPU/model-bound calls run in a bounded thread-pool executor
to avoid blocking the async event loop.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import numpy as np
import structlog

from nexus.domain.ports import EmbeddingPort

logger = structlog.get_logger(__name__)

# Bounded executor to limit concurrent embedding work
_EMBED_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embed")


class SentenceTransformerEmbedder(EmbeddingPort):
    """Embedding adapter using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = None
        self._dimension: int = 0
        self._semaphore = asyncio.Semaphore(4)  # max concurrent embed requests

    async def initialize(self) -> None:
        if self._model is not None:
            return

        logger.info("loading_embedding_model", model=self._model_name, device=self._device)

        loop = asyncio.get_event_loop()

        def _load():
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self._model_name, device=self._device)
            dim = len(model.encode("test", convert_to_numpy=True))
            return model, dim

        self._model, self._dimension = await loop.run_in_executor(_EMBED_EXECUTOR, _load)
        logger.info("embedding_model_loaded", dimension=self._dimension)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self._model:
            raise RuntimeError("Embedding model not initialized")

        async with self._semaphore:
            loop = asyncio.get_event_loop()

            def _encode():
                embeddings = self._model.encode(
                    texts,
                    convert_to_numpy=True,
                    batch_size=self._batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                return embeddings.tolist()

            return await loop.run_in_executor(_EMBED_EXECUTOR, _encode)

    async def embed_query(self, text: str) -> list[float]:
        results = await self.embed_texts([text])
        return results[0]

    def get_dimension(self) -> int:
        return self._dimension
