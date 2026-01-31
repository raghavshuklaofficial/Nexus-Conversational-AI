"""
Configuration Management
========================

Centralized configuration using Pydantic settings with environment variable support,
validation, and type safety.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment modes."""
    
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class ModelProvider(str, Enum):
    """Supported model providers for NLU."""
    
    LOCAL = "local"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    CUSTOM = "custom"


class NLUConfig(BaseSettings):
    """Natural Language Understanding configuration."""
    
    model_config = SettingsConfigDict(env_prefix="NEXUS_NLU_")
    
    # Model settings
    intent_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Transformer model for intent classification"
    )
    entity_model: str = Field(
        default="dslim/bert-base-NER",
        description="Model for named entity recognition"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="Model for semantic embeddings"
    )
    
    # Processing settings
    max_sequence_length: int = Field(default=512, ge=32, le=4096)
    confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    fallback_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    
    # Device configuration
    device: str = Field(default="auto", description="Device: 'cpu', 'cuda', 'mps', or 'auto'")
    use_fp16: bool = Field(default=True, description="Use FP16 for inference")
    
    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate and resolve device setting."""
        if v == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return v


class DialogueConfig(BaseSettings):
    """Dialogue management configuration."""
    
    model_config = SettingsConfigDict(env_prefix="NEXUS_DIALOGUE_")
    
    # Context management
    max_history_turns: int = Field(default=10, ge=1, le=100)
    context_window_size: int = Field(default=5, ge=1, le=50)
    session_timeout_minutes: int = Field(default=30, ge=1, le=1440)
    
    # Response generation
    enable_sentiment_analysis: bool = Field(default=True)
    enable_entity_memory: bool = Field(default=True)
    enable_topic_tracking: bool = Field(default=True)
    
    # Fallback handling
    max_clarification_attempts: int = Field(default=3, ge=1, le=10)
    enable_smart_fallback: bool = Field(default=True)


class APIConfig(BaseSettings):
    """API server configuration."""
    
    model_config = SettingsConfigDict(env_prefix="NEXUS_API_")
    
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1024, le=65535)
    workers: int = Field(default=1, ge=1, le=32)
    
    # Security
    enable_cors: bool = Field(default=True)
    allowed_origins: list[str] = Field(default=["*"])
    api_key_header: str = Field(default="X-API-Key")
    rate_limit_per_minute: int = Field(default=60, ge=1)
    
    # WebSocket
    ws_heartbeat_interval: int = Field(default=30, ge=5, le=300)
    max_connections_per_ip: int = Field(default=10, ge=1, le=100)


class DatabaseConfig(BaseSettings):
    """Database configuration."""
    
    model_config = SettingsConfigDict(env_prefix="NEXUS_DB_")
    
    url: str = Field(default="sqlite+aiosqlite:///./nexus.db")
    pool_size: int = Field(default=5, ge=1, le=100)
    max_overflow: int = Field(default=10, ge=0, le=100)
    echo: bool = Field(default=False)


class CacheConfig(BaseSettings):
    """Caching configuration."""
    
    model_config = SettingsConfigDict(env_prefix="NEXUS_CACHE_")
    
    backend: str = Field(default="memory", description="'memory' or 'redis'")
    redis_url: str = Field(default="redis://localhost:6379/0")
    default_ttl: int = Field(default=3600, ge=60)
    embedding_cache_size: int = Field(default=10000, ge=100)


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    
    model_config = SettingsConfigDict(env_prefix="NEXUS_LOG_")
    
    level: str = Field(default="INFO")
    format: str = Field(default="json", description="'json' or 'console'")
    include_timestamp: bool = Field(default=True)
    include_caller: bool = Field(default=True)


class MetricsConfig(BaseSettings):
    """Observability and metrics configuration."""
    
    model_config = SettingsConfigDict(env_prefix="NEXUS_METRICS_")
    
    enabled: bool = Field(default=True)
    port: int = Field(default=9090, ge=1024, le=65535)
    include_latency_histograms: bool = Field(default=True)
    include_model_metrics: bool = Field(default=True)


class NexusConfig(BaseSettings):
    """
    Main application configuration.
    
    Aggregates all sub-configurations and provides global settings.
    Environment variables override default values with prefix NEXUS_.
    """
    
    model_config = SettingsConfigDict(
        env_prefix="NEXUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Global settings
    app_name: str = Field(default="Nexus Conversational AI")
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False)
    
    # Data paths
    data_dir: Path = Field(default=Path("data"))
    models_dir: Path = Field(default=Path("models"))
    
    # Sub-configurations
    nlu: NLUConfig = Field(default_factory=NLUConfig)
    dialogue: DialogueConfig = Field(default_factory=DialogueConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    
    @model_validator(mode="after")
    def setup_directories(self) -> "NexusConfig":
        """Ensure required directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        return self
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == Environment.DEVELOPMENT


@lru_cache(maxsize=1)
def get_config() -> NexusConfig:
    """
    Get the application configuration singleton.
    
    Returns:
        NexusConfig: The application configuration instance.
    """
    return NexusConfig()


def reload_config() -> NexusConfig:
    """
    Reload the configuration (clears cache).
    
    Returns:
        NexusConfig: Fresh configuration instance.
    """
    get_config.cache_clear()
    return get_config()
