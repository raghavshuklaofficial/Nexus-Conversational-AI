"""
NLU (Natural Language Understanding) Module
============================================

Advanced NLU components using transformer-based models for
intent classification, entity extraction, and semantic understanding.
"""

from nexus.nlu.classifier import IntentClassifier
from nexus.nlu.extractor import EntityExtractor
from nexus.nlu.sentiment import SentimentAnalyzer
from nexus.nlu.embeddings import EmbeddingEngine

__all__ = [
    "IntentClassifier",
    "EntityExtractor",
    "SentimentAnalyzer",
    "EmbeddingEngine",
]
