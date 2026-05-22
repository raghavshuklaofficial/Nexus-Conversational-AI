"""
NLU Service — wraps the existing NLU pipeline components behind an
application-layer service with concurrent execution.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import structlog

from nexus.core.response import Entity, IntentMatch, Sentiment
from nexus.nlu.classifier import IntentClassifier
from nexus.nlu.extractor import EntityExtractor
from nexus.nlu.sentiment import SentimentAnalyzer

logger = structlog.get_logger(__name__)


@dataclass
class NLUResult:
    intent: IntentMatch
    entities: list[Entity]
    sentiment: Sentiment
    sentiment_score: float
    latency_ms: float


class NLUService:
    """Orchestrates intent classification, NER, and sentiment concurrently."""

    def __init__(self, classifier: IntentClassifier, extractor: EntityExtractor, analyzer: SentimentAnalyzer):
        self._classifier = classifier
        self._extractor = extractor
        self._analyzer = analyzer
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        await asyncio.gather(
            self._classifier.load(),
            self._extractor.load(),
            self._analyzer.load(),
        )
        self._initialized = True
        logger.info("nlu_service_initialized")

    async def analyze(self, text: str) -> NLUResult:
        start = time.time()
        intent, entities, sentiment_result = await asyncio.gather(
            self._classifier.classify(text),
            self._extractor.extract(text),
            self._analyzer.analyze(text),
        )
        sentiment, score = sentiment_result
        latency = (time.time() - start) * 1000
        return NLUResult(
            intent=intent,
            entities=entities,
            sentiment=sentiment,
            sentiment_score=score,
            latency_ms=latency,
        )

    @property
    def is_ready(self) -> bool:
        return self._initialized
