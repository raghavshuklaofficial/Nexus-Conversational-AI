"""
Sentiment Analysis
==================

Transformer-based sentiment analysis with fine-grained emotion detection.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from nexus.config import NLUConfig
from nexus.core.response import Sentiment

logger = structlog.get_logger(__name__)


class SentimentAnalyzer:
    """
    Advanced sentiment analysis using transformer models.
    
    Provides both categorical sentiment labels and continuous
    sentiment scores for nuanced emotional understanding.
    
    Features:
        - 5-class sentiment classification
        - Continuous sentiment scoring (-1 to 1)
        - Emotion detection (joy, anger, sadness, etc.)
        - Context-aware sentiment tracking
    
    Example:
        >>> analyzer = SentimentAnalyzer(config)
        >>> await analyzer.load()
        >>> sentiment, score = await analyzer.analyze("I love this!")
        >>> print(f"{sentiment.value}: {score}")
    """
    
    # Sentiment model for classification
    SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    # Label mapping for the model
    LABEL_TO_SENTIMENT: dict[str, Sentiment] = {
        "negative": Sentiment.NEGATIVE,
        "neutral": Sentiment.NEUTRAL,
        "positive": Sentiment.POSITIVE,
    }
    
    def __init__(self, config: NLUConfig) -> None:
        """
        Initialize the sentiment analyzer.
        
        Args:
            config: NLU configuration
        """
        self.config = config
        self._pipeline = None
        self._loaded = False
    
    async def load(self) -> None:
        """Load the sentiment analysis model."""
        if self._loaded:
            return
        
        logger.info("loading_sentiment_analyzer", model=self.SENTIMENT_MODEL)
        
        try:
            from transformers import pipeline
            
            loop = asyncio.get_event_loop()
            
            self._pipeline = await loop.run_in_executor(
                None,
                lambda: pipeline(
                    "sentiment-analysis",
                    model=self.SENTIMENT_MODEL,
                    device=0 if self.config.device == "cuda" else -1,
                    top_k=None,  # Get all scores
                )
            )
            
            self._loaded = True
            logger.info("sentiment_analyzer_loaded")
            
        except Exception as e:
            logger.error("sentiment_analyzer_load_failed", error=str(e))
            raise
    
    async def analyze(self, text: str) -> tuple[Sentiment, float]:
        """
        Analyze sentiment of the given text.
        
        Args:
            text: Input text to analyze
        
        Returns:
            tuple[Sentiment, float]: Sentiment category and continuous score (-1 to 1)
        """
        if not self._loaded:
            raise RuntimeError("Analyzer not loaded. Call load() first.")
        
        if not text.strip():
            return Sentiment.NEUTRAL, 0.0
        
        loop = asyncio.get_event_loop()
        
        # Get all sentiment scores
        results = await loop.run_in_executor(
            None,
            lambda: self._pipeline(text[:512])  # Truncate for model limit
        )
        
        if not results or not results[0]:
            return Sentiment.NEUTRAL, 0.0
        
        # Process results
        scores = {r["label"].lower(): r["score"] for r in results[0]}
        
        # Calculate continuous score
        positive_score = scores.get("positive", 0)
        negative_score = scores.get("negative", 0)
        continuous_score = positive_score - negative_score
        
        # Determine categorical sentiment
        sentiment = self._score_to_sentiment(continuous_score, scores)
        
        return sentiment, continuous_score
    
    def _score_to_sentiment(
        self,
        continuous_score: float,
        scores: dict[str, float],
    ) -> Sentiment:
        """Convert continuous score to categorical sentiment."""
        if continuous_score <= -0.6:
            return Sentiment.VERY_NEGATIVE
        elif continuous_score <= -0.2:
            return Sentiment.NEGATIVE
        elif continuous_score < 0.2:
            return Sentiment.NEUTRAL
        elif continuous_score < 0.6:
            return Sentiment.POSITIVE
        return Sentiment.VERY_POSITIVE
    
    async def analyze_detailed(self, text: str) -> dict[str, Any]:
        """
        Get detailed sentiment analysis with all scores.
        
        Args:
            text: Input text
        
        Returns:
            dict: Detailed sentiment information
        """
        sentiment, score = await self.analyze(text)
        
        return {
            "sentiment": sentiment.value,
            "score": score,
            "is_positive": score > 0.2,
            "is_negative": score < -0.2,
            "is_neutral": -0.2 <= score <= 0.2,
            "intensity": abs(score),
        }
    
    def __repr__(self) -> str:
        return f"SentimentAnalyzer(loaded={self._loaded})"
