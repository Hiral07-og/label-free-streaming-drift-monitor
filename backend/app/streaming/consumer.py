"""
Streaming Event Consumer for Person 2 (Backend & Systems Lead).
Consumes transaction events from Apache Kafka / Redpanda topics or an in-memory
asynchronous processing queue, routing them to DriftService for bounded-memory
sketch updates, streaming ADWIN detection, and performance tracking.
"""

import json
import logging
import queue
import threading
import time
from typing import Any, Dict, Optional

from backend.app.config import settings
from backend.app.schemas.events import TransactionEvent
from backend.app.services.drift_service import DriftService

logger = logging.getLogger("drift_backend.streaming.consumer")


class StreamingEventConsumer:
    """
    Consumer that receives events from Kafka or an in-memory queue
    and forwards them to the DriftService monitoring pipeline.
    """

    def __init__(
        self,
        drift_service: DriftService,
        bootstrap_servers: Optional[str] = None,
        topic: Optional[str] = None,
        group_id: Optional[str] = None,
        enabled: Optional[bool] = None,
    ):
        self.drift_service = drift_service
        self.bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self.topic = topic or settings.kafka_topic_transactions
        self.group_id = group_id or settings.kafka_group_id
        self.enabled = enabled if enabled is not None else settings.kafka_enabled

        self.queue: queue.Queue = queue.Queue(maxsize=10000)
        self.is_running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._kafka_consumer = None

        self.consumed_count = 0
        self.error_count = 0
        self.last_consumed_at: Optional[float] = None

    def start(self) -> None:
        """Starts background consumption thread."""
        if self.is_running:
            return

        self.is_running = True
        if self.enabled:
            self._init_kafka_consumer()

        self._worker_thread = threading.Thread(
            target=self._run_consumer_loop,
            daemon=True,
            name="StreamConsumerWorker",
        )
        self._worker_thread.start()
        logger.info("StreamingEventConsumer worker started (Kafka enabled: %s)", self.enabled)

    def stop(self) -> None:
        """Stops background consumption worker cleanly."""
        self.is_running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        if self._kafka_consumer and hasattr(self._kafka_consumer, "close"):
            try:
                self._kafka_consumer.close()
            except Exception:
                pass
        logger.info("StreamingEventConsumer stopped.")

    def enqueue(self, event: TransactionEvent) -> bool:
        """Submits an event to the local consumer queue for asynchronous processing."""
        try:
            self.queue.put_nowait(event)
            return True
        except queue.Full:
            logger.warning("Consumer queue is full; dropping event %s", event.record_id)
            return False

    def _init_kafka_consumer(self) -> None:
        """Attempts to initialize Kafka consumer connection."""
        try:
            from kafka import KafkaConsumer  # type: ignore

            self._kafka_consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest",
                enable_auto_commit=True,
                consumer_timeout_ms=1000,
            )
            logger.info("Connected Kafka consumer to topic %s", self.topic)
        except Exception as e:
            logger.warning(
                "Could not initialize Kafka consumer (%s). Operating on in-memory queue.", e
            )
            self.enabled = False

    def _run_consumer_loop(self) -> None:
        """Main consumption loop."""
        while self.is_running:
            processed_any = False

            # 1. Poll Kafka if active
            if self.enabled and self._kafka_consumer:
                try:
                    records = self._kafka_consumer.poll(timeout_ms=500)
                    for topic_partition, msgs in records.items():
                        for msg in msgs:
                            self._process_raw_payload(msg.value)
                            processed_any = True
                except Exception as e:
                    logger.error("Error polling from Kafka: %s", e)
                    self.error_count += 1

            # 2. Process in-memory queue
            try:
                while True:
                    event = self.queue.get_nowait()
                    self.drift_service.process_event(event)
                    self.consumed_count += 1
                    self.last_consumed_at = time.time()
                    self.queue.task_done()
                    processed_any = True
            except queue.Empty:
                pass

            if not processed_any:
                time.sleep(0.05)

    def _process_raw_payload(self, raw_data: Dict[str, Any]) -> None:
        """Parses raw payload into TransactionEvent and routes to DriftService."""
        try:
            event = TransactionEvent(**raw_data)
            self.drift_service.process_event(event)
            self.consumed_count += 1
            self.last_consumed_at = time.time()
        except Exception as e:
            logger.error("Error parsing or monitoring streaming event: %s", e)
            self.error_count += 1

    def get_status(self) -> Dict[str, Any]:
        """Returns consumer runtime statistics."""
        return {
            "is_running": self.is_running,
            "kafka_enabled": self.enabled,
            "topic": self.topic if self.enabled else "in_memory_queue",
            "consumed_count": self.consumed_count,
            "queue_depth": self.queue.qsize(),
            "error_count": self.error_count,
            "last_consumed_at": self.last_consumed_at,
        }
