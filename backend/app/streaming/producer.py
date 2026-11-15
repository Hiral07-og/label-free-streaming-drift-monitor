"""
Streaming Event Producer for Person 2 (Backend & Systems Lead).
Supports publishing transaction events to Apache Kafka / Redpanda topics,
with a robust fallback event stream generator simulator for local development.
"""

import json
import logging
import random
import time
from typing import Any, Dict, Generator, Optional

from backend.app.config import settings
from backend.app.schemas.events import TransactionEvent

logger = logging.getLogger("drift_backend.streaming.producer")


class StreamingEventProducer:
    """
    Producer capable of dispatching events to Kafka/Redpanda broker
    or generating realistic continuous transaction streams for local testing.
    """

    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        topic: Optional[str] = None,
        enabled: Optional[bool] = None,
    ):
        self.bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self.topic = topic or settings.kafka_topic_transactions
        self.enabled = enabled if enabled is not None else settings.kafka_enabled
        self._kafka_producer = None

        if self.enabled:
            self._init_kafka_producer()

    def _init_kafka_producer(self) -> None:
        """Attempts to initialize Kafka producer if client library is present."""
        try:
            from kafka import KafkaProducer  # type: ignore

            self._kafka_producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: str(k).encode("utf-8") if k else None,
                retries=3,
            )
            logger.info("Connected to Kafka producer at %s", self.bootstrap_servers)
        except ImportError:
            try:
                import confluent_kafka  # type: ignore

                conf = {
                    "bootstrap.servers": self.bootstrap_servers,
                    "client.id": "drift-producer",
                }
                self._kafka_producer = confluent_kafka.Producer(conf)
                logger.info("Connected via confluent_kafka to %s", self.bootstrap_servers)
            except ImportError:
                logger.warning(
                    "Kafka client libraries not installed. Producer running in local simulation mode."
                )
                self.enabled = False
        except Exception as e:
            logger.warning("Could not connect to Kafka broker (%s). Falling back to simulation.", e)
            self.enabled = False

    def send_event(self, event: TransactionEvent) -> bool:
        """
        Publishes a transaction event to Kafka or logs locally.
        Returns True if sent successfully.
        """
        payload = event.model_dump()
        record_id = payload.get("record_id") or f"tx_{int(time.time() * 1000)}"

        if self.enabled and self._kafka_producer:
            try:
                if hasattr(self._kafka_producer, "send"):
                    self._kafka_producer.send(self.topic, key=record_id, value=payload)
                elif hasattr(self._kafka_producer, "produce"):
                    self._kafka_producer.produce(
                        self.topic, key=record_id, value=json.dumps(payload)
                    )
                    self._kafka_producer.poll(0)
                return True
            except Exception as e:
                logger.error("Failed to send event to Kafka: %s", e)
                return False
        else:
            # Local in-memory acknowledgment
            return True

    def generate_event(
        self,
        drift_mode: str = "none",
        record_idx: int = 1,
    ) -> TransactionEvent:
        """
        Generates a synthetic transaction event:
        - drift_mode="none": normal stationary baseline
        - drift_mode="abrupt": sudden high transaction amount and high risk
        - drift_mode="gradual": shifted mean and elevated entropy
        """
        now = time.time()
        rec_id = f"tx_sim_{record_idx:06d}"

        if drift_mode == "abrupt":
            # Sudden high fraud surge
            amount = float(np_random_amount := max(10.0, random.gauss(550.0, 180.0)))
            sender_count = float(max(1.0, random.gauss(18.0, 5.0)))
            risk_score = float(min(1.0, max(0.0, random.betavariate(5, 2))))
            loc_entropy = float(min(1.0, max(0.0, random.uniform(0.65, 0.99))))
            novelty = float(min(1.0, max(0.0, random.gauss(0.85, 0.1))))
            time_delta = float(max(2.0, random.expovariate(1 / 45.0)))
            degree = float(max(1.0, random.gauss(8.0, 2.5)))
            device = random.choices(["mobile_app", "tor_proxy", "bot_client"], weights=[0.2, 0.4, 0.4])[0]
            merchant = random.choices(["crypto_exchange", "luxury_goods", "electronics"], weights=[0.5, 0.3, 0.2])[0]
            is_fraud = 1 if random.random() < 0.80 else 0
        elif drift_mode == "gradual":
            # Gradual shift
            factor = min(2.5, 1.0 + (record_idx % 200) * 0.007)
            amount = float(max(5.0, random.gauss(120.0 * factor, 40.0)))
            sender_count = float(max(1.0, random.gauss(3.5 * factor, 1.5)))
            risk_score = float(min(1.0, max(0.0, random.betavariate(2, 5) * factor)))
            loc_entropy = float(min(1.0, max(0.0, random.uniform(0.2, 0.5) * factor)))
            novelty = float(min(1.0, max(0.0, random.gauss(0.4 * factor, 0.15))))
            time_delta = float(max(5.0, random.expovariate(1 / 200.0)))
            degree = float(max(1.0, random.gauss(3.0, 1.2)))
            device = random.choices(["mobile_app", "web_browser", "api_gateway"], weights=[0.4, 0.4, 0.2])[0]
            merchant = random.choices(["groceries", "electronics", "online_service"], weights=[0.4, 0.3, 0.3])[0]
            is_fraud = 1 if random.random() < (0.10 * factor) else 0
        else:
            # Stationary baseline
            amount = float(max(5.0, random.gauss(75.0, 25.0)))
            sender_count = float(max(1.0, random.gauss(1.8, 0.9)))
            risk_score = float(min(1.0, max(0.0, random.betavariate(1.5, 8))))
            loc_entropy = float(min(1.0, max(0.0, random.uniform(0.1, 0.35))))
            novelty = float(min(1.0, max(0.0, random.gauss(0.25, 0.1))))
            time_delta = float(max(10.0, random.expovariate(1 / 350.0)))
            degree = float(max(1.0, random.gauss(2.0, 0.8)))
            device = random.choices(["web_browser", "mobile_app", "pos_terminal"], weights=[0.5, 0.4, 0.1])[0]
            merchant = random.choices(["groceries", "restaurants", "online_service"], weights=[0.5, 0.3, 0.2])[0]
            is_fraud = 1 if random.random() < 0.05 else 0

        return TransactionEvent(
            record_id=rec_id,
            timestamp=now,
            tx_amount=round(amount, 2),
            sender_tx_count_1h=round(sender_count, 1),
            sender_risk_score=round(risk_score, 4),
            location_entropy=round(loc_entropy, 4),
            dest_novelty_score=round(novelty, 4),
            time_delta_last_tx=round(time_delta, 1),
            network_degree=round(degree, 1),
            device_type=device,
            merchant_category=merchant,
            is_fraud=is_fraud,
        )

    def stream_generator(
        self,
        count: int = 100,
        drift_mode: str = "none",
        delay_seconds: float = 0.01,
    ) -> Generator[TransactionEvent, None, None]:
        """Yields a sequence of streaming events with optional delay."""
        for i in range(1, count + 1):
            event = self.generate_event(drift_mode=drift_mode, record_idx=i)
            if self.enabled:
                self.send_event(event)
            yield event
            if delay_seconds > 0:
                time.sleep(delay_seconds)
