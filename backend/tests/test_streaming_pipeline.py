"""
Unit Tests for Streaming Pipeline (Producer, Consumer, Simulation).
Person 2 (Backend & Systems Lead)
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.app.schemas.events import TransactionEvent
from backend.app.services.drift_service import DriftService
from backend.app.streaming.consumer import StreamingEventConsumer
from backend.app.streaming.producer import StreamingEventProducer


class TestStreamingPipeline(unittest.TestCase):
    """Test suite for event producer and consumer."""

    def setUp(self):
        self.producer = StreamingEventProducer(enabled=False)
        self.service = DriftService.get_instance()
        self.consumer = StreamingEventConsumer(drift_service=self.service, enabled=False)

    def test_01_producer_generate_event(self):
        """Verify synthetic transaction event generation across drift regimes."""
        event_none = self.producer.generate_event(drift_mode="none", record_idx=1)
        self.assertIsInstance(event_none, TransactionEvent)
        self.assertGreater(event_none.tx_amount, 0)

        event_abrupt = self.producer.generate_event(drift_mode="abrupt", record_idx=2)
        self.assertIsInstance(event_abrupt, TransactionEvent)

        event_gradual = self.producer.generate_event(drift_mode="gradual", record_idx=3)
        self.assertIsInstance(event_gradual, TransactionEvent)

    def test_02_producer_stream_generator(self):
        """Verify stream generator yields requested count of events."""
        events = list(self.producer.stream_generator(count=10, drift_mode="none", delay_seconds=0))
        self.assertEqual(len(events), 10)
        for ev in events:
            self.assertIsInstance(ev, TransactionEvent)

    def test_03_consumer_queue_and_status(self):
        """Verify consumer worker lifecycle and queue processing."""
        self.consumer.start()
        self.assertTrue(self.consumer.is_running)

        event = self.producer.generate_event(drift_mode="none", record_idx=999)
        enqueued = self.consumer.enqueue(event)
        self.assertTrue(enqueued)

        status = self.consumer.get_status()
        self.assertTrue(status["is_running"])
        self.assertFalse(status["kafka_enabled"])

        self.consumer.stop()
        self.assertFalse(self.consumer.is_running)


if __name__ == "__main__":
    unittest.main()
