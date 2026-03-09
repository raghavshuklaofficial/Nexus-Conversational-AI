"""
Entity extraction using BERT NER + regex patterns for
things like emails, phone numbers, dates, etc.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import structlog

from nexus.config import NLUConfig
from nexus.core.response import Entity

logger = structlog.get_logger(__name__)


class EntityExtractor:
    """
    Extracts named entities from text using a transformer NER model
    plus custom regex patterns for structured entities.
    """
    
    # regex patterns for common structured entities
    CUSTOM_PATTERNS: dict[str, str] = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE": r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
        "URL": r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)",
        "TIME": r"\b(?:1[0-2]|0?[1-9])(?::[0-5][0-9])?\s*(?:am|pm|AM|PM)\b|\b(?:[01]?[0-9]|2[0-3]):[0-5][0-9]\b",
        "DATE": r"\b(?:today|tomorrow|yesterday|next\s+(?:week|month|year)|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,?\s+\d{4})?)\b",
        "CURRENCY": r"\$\s*[\d,]+(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD|EUR|GBP)\b",
        "NUMBER": r"\b\d+(?:,\d{3})*(?:\.\d+)?\b",
    }
    
    # mapping from various NER label schemes to our standard types
    LABEL_MAPPING: dict[str, str] = {
        "PER": "PERSON",
        "PERSON": "PERSON",
        "ORG": "ORGANIZATION",
        "ORGANIZATION": "ORGANIZATION",
        "LOC": "LOCATION",
        "LOCATION": "LOCATION",
        "GPE": "LOCATION",
        "MISC": "MISCELLANEOUS",
        "DATE": "DATE",
        "TIME": "TIME",
        "MONEY": "CURRENCY",
        "PERCENT": "PERCENTAGE",
        "CARDINAL": "NUMBER",
        "ORDINAL": "ORDINAL",
    }
    
    def __init__(self, config: NLUConfig) -> None:
        self.config = config
        self._pipeline = None
        self._loaded = False
        
        # pre-compile the regex patterns once
        self._compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.CUSTOM_PATTERNS.items()
        }
    
    async def load(self) -> None:
        """Load the NER model."""
        if self._loaded:
            return
        
        logger.info("loading_entity_extractor", model=self.config.entity_model)
        
        try:
            from transformers import pipeline
            
            loop = asyncio.get_event_loop()
            
            # Load NER pipeline
            self._pipeline = await loop.run_in_executor(
                None,
                lambda: pipeline(
                    "ner",
                    model=self.config.entity_model,
                    device=0 if self.config.device == "cuda" else -1,
                    aggregation_strategy="simple",
                )
            )
            
            self._loaded = True
            logger.info("entity_extractor_loaded")
            
        except Exception as e:
            logger.error("entity_extractor_load_failed", error=str(e))
            raise
    
    async def extract(self, text: str) -> list[Entity]:
        """Extract entities from text using both NER model and regex."""
        if not self._loaded:
            raise RuntimeError("Extractor not loaded. Call load() first.")
        
        entities: list[Entity] = []
        seen_spans: set[tuple[int, int]] = set()
        
        # Run transformer NER
        loop = asyncio.get_event_loop()
        ner_results = await loop.run_in_executor(
            None,
            lambda: self._pipeline(text)
        )
        
        # Process NER results
        for result in ner_results:
            entity_type = self._normalize_label(result["entity_group"])
            start = result["start"]
            end = result["end"]
            
            # Skip if overlapping with existing entity
            if self._overlaps(start, end, seen_spans):
                continue
            
            seen_spans.add((start, end))
            
            entities.append(Entity(
                text=result["word"],
                type=entity_type,
                confidence=float(result["score"]),
                start_pos=start,
                end_pos=end,
            ))
        
        # Extract regex-based entities
        regex_entities = self._extract_regex_entities(text, seen_spans)
        entities.extend(regex_entities)
        
        # Sort by position
        entities.sort(key=lambda e: e.start_pos)
        
        return entities
    
    def _extract_regex_entities(
        self,
        text: str,
        seen_spans: set[tuple[int, int]],
    ) -> list[Entity]:
        """Extract entities using regex patterns."""
        entities: list[Entity] = []
        
        for entity_type, pattern in self._compiled_patterns.items():
            for match in pattern.finditer(text):
                start, end = match.span()
                
                if self._overlaps(start, end, seen_spans):
                    continue
                
                seen_spans.add((start, end))
                
                entities.append(Entity(
                    text=match.group(),
                    type=entity_type,
                    value=self._normalize_value(entity_type, match.group()),
                    confidence=1.0,  # regex match = certain
                    start_pos=start,
                    end_pos=end,
                ))
        
        return entities
    
    def _normalize_label(self, label: str) -> str:
        """Normalize NER labels to standard types."""
        # Remove B-, I- prefixes
        label = label.replace("B-", "").replace("I-", "")
        return self.LABEL_MAPPING.get(label.upper(), label.upper())
    
    def _normalize_value(self, entity_type: str, text: str) -> str | None:
        """Normalize entity value based on type."""
        if entity_type == "PHONE":
            # Remove non-digits
            return re.sub(r"\D", "", text)
        elif entity_type == "EMAIL":
            return text.lower()
        elif entity_type == "CURRENCY":
            # Extract numeric value
            match = re.search(r"[\d,]+(?:\.\d+)?", text)
            return match.group().replace(",", "") if match else text
        return None
    
    @staticmethod
    def _overlaps(start: int, end: int, spans: set[tuple[int, int]]) -> bool:
        """Check if span overlaps with any existing spans."""
        for s_start, s_end in spans:
            if start < s_end and end > s_start:
                return True
        return False
    
    def __repr__(self) -> str:
        return f"EntityExtractor(loaded={self._loaded})"
