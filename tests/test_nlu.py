# Tests for NLU components (classifier, extractor, sentiment, embeddings)

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nexus.core.response import Entity, IntentMatch, Sentiment


class TestIntentClassifier:
    """Tests for the IntentClassifier."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock NLU config."""
        config = MagicMock()
        config.intent_model = "sentence-transformers/all-MiniLM-L6-v2"
        config.device = "cpu"
        config.fallback_threshold = 0.4
        return config
    
    @pytest.mark.asyncio
    async def test_classifier_initialization(self, mock_config):
        """Test that classifier initializes correctly."""
        from nexus.nlu.classifier import IntentClassifier
        
        classifier = IntentClassifier(mock_config)
        assert classifier._loaded is False
        assert classifier._model is None
    
    @pytest.mark.asyncio
    async def test_classify_returns_intent_match(self, mock_config):
        """Test that classify returns IntentMatch object."""
        from nexus.nlu.classifier import IntentClassifier
        
        classifier = IntentClassifier(mock_config)
        classifier._loaded = True
        classifier._intent_embeddings = {"greeting": MagicMock()}
        
        # Mock the model
        with patch.object(classifier, '_model') as mock_model:
            mock_model.encode.return_value = MagicMock()
            
            with patch.object(classifier, '_cosine_similarity', return_value=0.85):
                result = await classifier.classify("hello")
                
                assert isinstance(result, IntentMatch)
                assert result.confidence >= 0 and result.confidence <= 1


class TestEntityExtractor:
    """Tests for the EntityExtractor."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock NLU config."""
        config = MagicMock()
        config.entity_model = "dslim/bert-base-NER"
        config.device = "cpu"
        return config
    
    def test_regex_email_extraction(self, mock_config):
        """Test email extraction via regex."""
        from nexus.nlu.extractor import EntityExtractor
        
        extractor = EntityExtractor(mock_config)
        
        # Test email pattern
        text = "Contact me at test@example.com please"
        pattern = extractor._compiled_patterns["EMAIL"]
        matches = pattern.findall(text)
        
        assert len(matches) == 1
        assert "test@example.com" in matches
    
    def test_regex_phone_extraction(self, mock_config):
        """Test phone number extraction via regex."""
        from nexus.nlu.extractor import EntityExtractor
        
        extractor = EntityExtractor(mock_config)
        
        text = "Call me at 555-123-4567"
        pattern = extractor._compiled_patterns["PHONE"]
        matches = pattern.findall(text)
        
        assert len(matches) >= 1
    
    def test_normalize_email(self, mock_config):
        """Test email normalization."""
        from nexus.nlu.extractor import EntityExtractor
        
        extractor = EntityExtractor(mock_config)
        
        result = extractor._normalize_value("EMAIL", "Test@Example.COM")
        assert result == "test@example.com"


class TestSentimentAnalyzer:
    """Tests for the SentimentAnalyzer."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock NLU config."""
        config = MagicMock()
        config.device = "cpu"
        return config
    
    def test_score_to_sentiment_mapping(self, mock_config):
        """Test sentiment score to category mapping."""
        from nexus.nlu.sentiment import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer(mock_config)
        
        scores = {"positive": 0.9, "neutral": 0.05, "negative": 0.05}
        sentiment = analyzer._score_to_sentiment(0.85, scores)
        
        assert sentiment == Sentiment.VERY_POSITIVE
    
    def test_negative_sentiment_detection(self, mock_config):
        """Test negative sentiment detection."""
        from nexus.nlu.sentiment import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer(mock_config)
        
        scores = {"positive": 0.05, "neutral": 0.1, "negative": 0.85}
        sentiment = analyzer._score_to_sentiment(-0.8, scores)
        
        assert sentiment in [Sentiment.NEGATIVE, Sentiment.VERY_NEGATIVE]


class TestEmbeddingEngine:
    """Tests for the EmbeddingEngine."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock NLU config."""
        config = MagicMock()
        config.embedding_model = "sentence-transformers/all-mpnet-base-v2"
        config.device = "cpu"
        return config
    
    def test_cosine_similarity_calculation(self, mock_config):
        """Test cosine similarity calculation."""
        import numpy as np
        from nexus.nlu.embeddings import EmbeddingEngine
        
        engine = EmbeddingEngine(mock_config)
        
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        
        similarity = engine._compute_similarity(vec1, vec2, "cosine")
        assert abs(similarity - 1.0) < 0.001
    
    def test_orthogonal_vectors_similarity(self, mock_config):
        """Test similarity of orthogonal vectors."""
        import numpy as np
        from nexus.nlu.embeddings import EmbeddingEngine
        
        engine = EmbeddingEngine(mock_config)
        
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        
        similarity = engine._compute_similarity(vec1, vec2, "cosine")
        assert abs(similarity) < 0.001


class TestDialogueState:
    """Tests for DialogueState."""
    
    def test_state_creation(self):
        """Test dialogue state creation."""
        from nexus.dialogue.state import DialogueState
        
        intent = IntentMatch(name="greeting", confidence=0.9, is_fallback=False)
        
        state = DialogueState(
            user_input="Hello",
            intent=intent,
        )
        
        assert state.user_input == "Hello"
        assert state.intent.name == "greeting"
    
    def test_entity_access(self):
        """Test entity access methods."""
        from nexus.dialogue.state import DialogueState
        
        intent = IntentMatch(name="booking", confidence=0.8, is_fallback=False)
        entities = [
            Entity(text="Paris", type="LOCATION", confidence=0.95, start_pos=10, end_pos=15),
            Entity(text="tomorrow", type="DATE", confidence=0.9, start_pos=20, end_pos=28),
        ]
        
        state = DialogueState(
            user_input="Flight to Paris tomorrow",
            intent=intent,
            entities=entities,
        )
        
        assert state.has_entities is True
        assert state.get_entity("LOCATION").text == "Paris"
        assert len(state.get_entities("DATE")) == 1


class TestIntentData:
    """Tests for intent data."""
    
    def test_get_intent_patterns(self):
        """Test retrieval of intent patterns."""
        from nexus.data.intents import get_intent_patterns
        
        patterns = get_intent_patterns()
        
        assert "greeting" in patterns
        assert "goodbye" in patterns
        assert len(patterns["greeting"]["patterns"]) > 0
    
    def test_get_intent_responses(self):
        """Test retrieval of intent responses."""
        from nexus.data.intents import get_intent_responses
        
        responses = get_intent_responses()
        
        assert "greeting" in responses
        assert len(responses["greeting"]) > 0
    
    def test_intent_data_coverage(self):
        """Test that all intents have both patterns and responses."""
        from nexus.data.intents import get_intent_patterns, get_intent_responses
        
        patterns = get_intent_patterns()
        responses = get_intent_responses()
        
        for intent_name in patterns:
            assert intent_name in responses, f"Missing responses for {intent_name}"


class TestResponseTypes:
    """Tests for response types and models."""
    
    def test_response_creation(self):
        """Test Response model creation."""
        from nexus.core.response import Response, ResponseMetadata, ResponseType
        
        metadata = ResponseMetadata(processing_time_ms=50.0)
        
        response = Response(
            text="Hello! How can I help?",
            type=ResponseType.STANDARD,
            metadata=metadata,
        )
        
        assert response.text == "Hello! How can I help?"
        assert response.type == ResponseType.STANDARD
    
    def test_response_to_simple(self):
        """Test simplified response output."""
        from nexus.core.response import Response, ResponseMetadata, ResponseType, IntentMatch
        
        intent = IntentMatch(name="greeting", confidence=0.9, is_fallback=False)
        metadata = ResponseMetadata(
            processing_time_ms=50.0,
            detected_intent=intent,
        )
        
        response = Response(
            text="Hello!",
            type=ResponseType.STANDARD,
            metadata=metadata,
        )
        
        simple = response.to_simple()
        
        assert "text" in simple
        assert "confidence" in simple
        assert simple["confidence"] == 0.9
