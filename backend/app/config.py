"""
Backend configuration settings for Person 2 (Backend & Systems Lead).
Manages FastAPI server parameters, Kafka/Redpanda connection parameters,
database configuration, and drift monitoring defaults.
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field


class BackendConfig(BaseModel):
    """Configuration settings for streaming drift backend."""
    app_name: str = "Label-Free Streaming Drift Monitor API"
    app_version: str = "1.0.0"
    debug: bool = False

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Storage settings
    db_url: str = Field(
        default=f"sqlite:///{Path(__file__).parent.parent / 'storage' / 'monitor.db'}"
    )

    # Kafka / Redpanda streaming settings
    kafka_bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    kafka_topic_transactions: str = "transaction-stream"
    kafka_topic_alerts: str = "drift-alerts"
    kafka_group_id: str = "drift-monitor-backend"
    kafka_enabled: bool = os.getenv("KAFKA_ENABLED", "false").lower() == "true"

    # Streaming simulation settings
    default_stream_rate_per_sec: float = 50.0
    monitor_window_size: int = 300
    monitor_check_interval: int = 50


settings = BackendConfig()
