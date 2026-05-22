"""
Prometheus metrics for the Nexus platform.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest, CONTENT_TYPE_LATEST

APP_INFO = Info("nexus", "Nexus Conversational AI Platform")

REQUEST_LATENCY = Histogram(
    "nexus_request_latency_seconds", "Request latency",
    ["method", "endpoint", "status"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
REQUEST_COUNT = Counter("nexus_requests_total", "Total requests", ["method", "endpoint", "status"])
MODEL_LATENCY = Histogram(
    "nexus_model_latency_seconds", "Model inference latency",
    ["model_type"], buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
RETRIEVAL_LATENCY = Histogram(
    "nexus_retrieval_latency_seconds", "Vector retrieval latency",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)
CACHE_HITS = Counter("nexus_cache_hits_total", "Cache hits", ["cache_type"])
CACHE_MISSES = Counter("nexus_cache_misses_total", "Cache misses", ["cache_type"])
KAFKA_MESSAGES_PUBLISHED = Counter("nexus_kafka_messages_published_total", "Kafka messages published", ["topic"])
KAFKA_MESSAGES_CONSUMED = Counter("nexus_kafka_messages_consumed_total", "Kafka messages consumed", ["topic", "status"])
INGESTION_TOTAL = Counter("nexus_ingestion_total", "Documents ingested", ["status"])
INGESTION_CHUNKS = Counter("nexus_ingestion_chunks_total", "Chunks created")
ACTIVE_SESSIONS = Gauge("nexus_active_sessions", "Active sessions")


def get_metrics() -> bytes:
    return generate_latest()

def get_content_type() -> str:
    return CONTENT_TYPE_LATEST
