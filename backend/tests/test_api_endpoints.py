"""
Unit and Integration Tests for Person 2 FastAPI Endpoints.
Verifies all REST routes, Prometheus exporter, schema validation, and health checks.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi.testclient import TestClient
from backend.app.main import app


class TestAPIEndpoints(unittest.TestCase):
    """Test suite for FastAPI endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_check(self):
        """Verify health check returns UP."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "UP")
        self.assertIn("service", data)

    def test_02_ingest_single_record(self):
        """Verify synchronous record ingestion."""
        payload = {
            "record_id": "tx_test_001",
            "tx_amount": 120.50,
            "sender_tx_count_1h": 2.0,
            "sender_risk_score": 0.15,
            "location_entropy": 0.25,
            "dest_novelty_score": 0.40,
            "time_delta_last_tx": 240.0,
            "network_degree": 2.0,
            "device_type": "web_browser",
            "merchant_category": "groceries",
            "is_fraud": 0,
        }
        response = self.client.post("/api/v1/stream/record", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["record_id"], "tx_test_001")
        self.assertIn("prediction", data)
        self.assertIn("probability", data)
        self.assertIn("streaming_drift_detected", data)

    def test_03_ingest_batch(self):
        """Verify batch ingestion and throughput measurement."""
        records = [
            {
                "record_id": f"tx_batch_{i}",
                "tx_amount": 50.0 + i,
                "sender_tx_count_1h": 1.0,
                "sender_risk_score": 0.05,
                "location_entropy": 0.1,
                "dest_novelty_score": 0.2,
                "time_delta_last_tx": 300.0,
                "network_degree": 1.0,
                "device_type": "mobile_app",
                "merchant_category": "groceries",
            }
            for i in range(10)
        ]
        response = self.client.post("/api/v1/stream/batch", json={"records": records})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["processed_count"], 10)
        self.assertGreater(data["throughput_rec_per_sec"], 0)

    def test_04_simulate_scenario(self):
        """Verify stream simulator injects records and returns report."""
        response = self.client.post(
            "/api/v1/stream/simulate",
            json={"num_records": 30, "drift_type": "none"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["simulated_records"], 30)

    def test_05_drift_status(self):
        """Verify drift status summary endpoint."""
        response = self.client.get("/api/v1/drift/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "HEALTHY")
        self.assertIn("memory", data)
        self.assertIn("alert", data)
        self.assertIn("retraining_recommendation", data)
        self.assertGreaterEqual(data["records_processed"], 0)

    def test_06_univariate_metrics(self):
        """Verify univariate feature metrics endpoint."""
        response = self.client.get("/api/v1/drift/univariate")
        self.assertEqual(response.status_code, 200)
        features = response.json()
        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)
        first = features[0]
        self.assertIn("feature_name", first)
        self.assertIn("psi", first)
        self.assertIn("ks_p_value", first)
        self.assertIn("tdigest_median", first)

    def test_07_multivariate_metrics(self):
        """Verify multivariate drift endpoint."""
        response = self.client.get("/api/v1/drift/multivariate")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("multivariate_drift_confirmed", data)

    def test_08_performance_estimate(self):
        """Verify CBPE estimated performance metrics."""
        response = self.client.get("/api/v1/performance/estimated")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("estimated_f1", data)
        self.assertIn("baseline_f1", data)

    def test_09_delayed_reconciliation(self):
        """Verify delayed ground truth reconciliation."""
        labels = [
            {"record_id": f"tx_batch_{i}", "is_fraud": 0}
            for i in range(5)
        ]
        response = self.client.post("/api/v1/performance/labels", json={"labels": labels})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["reconciled_count"], 5)

    def test_10_prometheus_metrics(self):
        """Verify standard Prometheus plain-text exporter output."""
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        text = response.text
        self.assertIn("drift_records_processed_total", text)
        self.assertIn("drift_throughput_rec_per_sec", text)
        self.assertIn("drift_sketches_memory_bytes", text)
        self.assertIn("drift_alert_severity", text)


if __name__ == "__main__":
    unittest.main()
