"""
Streaming pipeline module for Person 2 (Backend & Systems Lead).
Provides Kafka producer, consumer, and streaming simulator.
"""

from backend.app.streaming.consumer import StreamingEventConsumer
from backend.app.streaming.producer import StreamingEventProducer

__all__ = ["StreamingEventProducer", "StreamingEventConsumer"]
