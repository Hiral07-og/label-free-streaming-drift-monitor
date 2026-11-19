"""
Unified End-to-End Integration Test Suite.
Verifies the complete connected system spanning:
- ML Engine (Sketches, ADWIN, Page-Hinkley, KS, PSI, MMD, C2ST, CBPE, Shadow Evaluator)
- Streaming Pipeline (Producer -> Consumer -> DriftService)
- REST API Routers (stream, drift, performance, alerts, retrain, benchmarks)
- Persistent SQLite Storage, Prometheus Exporter, and Web Dashboard UI
"""

import os
import sys
import time
import unittest
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.drift_service import DriftService


class TestUnifiedEndToEndIntegration(unittest.TestCase):
    """End-to-end integration tests verifying Person 1 + Person 2 unified pipeline."""

    @classmethod
    def setUpClass(cls):
        cls._ctx = TestClient(app)
        cls.client = cls._ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def test_01_live_producer_consumer_pipeline(self):
        """Verify continuous Producer -> Consumer -> DriftService pipeline works end-to-end."""
        start_res = self.client.post(
            "/api/v1/stream/auto-stream/start",
            json={"drift_mode": "none", "rate_per_sec": 25.0},
        )
        self.assertEqual(start_res.status_code, 200)
        self.assertTrue(start_res.json()["auto_stream_active"])

        time.sleep(0.5)

        stop_res = self.client.post("/api/v1/stream/auto-stream/stop")
        self.assertEqual(stop_res.status_code, 200)
        self.assertFalse(stop_res.json()["auto_stream_active"])

        time.sleep(0.2)
        status = self.client.get("/api/v1/stream/consumer-status").json()
        self.assertGreater(status["auto_stream_emitted"], 0)
        self.assertGreater(status["consumer"]["consumed_count"], 0)
        self.assertEqual(status["consumer"]["error_count"], 0)

    def test_02_all_five_drift_scenarios_connected(self):
        """Verify all 5 SyntheticDriftInjector scenarios work via REST API."""
        for scenario in ["none", "abrupt", "gradual", "recurring", "feature_specific"]:
            res = self.client.post(
                "/api/v1/stream/simulate",
                json={"num_records": 30, "drift_type": scenario},
            )
            self.assertEqual(res.status_code, 200, f"Scenario '{scenario}' failed")
            data = res.json()
            self.assertEqual(data["scenario"], scenario)
            self.assertEqual(data["simulated_records"], 30)

    def test_03_all_five_sketches_exposed_via_api(self):
        """Verify Welford, t-digest, Count-Min Sketch, HyperLogLog, and Reservoir are active."""
        status = self.client.get("/api/v1/drift/status").json()
        self.assertGreater(status["memory"]["sketches_total_bytes"], 0)
        self.assertIn("device_type", status["categorical_sketches"])
        self.assertIn("merchant_category", status["categorical_sketches"])
        self.assertGreater(status["categorical_sketches"]["device_type"]["total_counts"], 0)

        uni = self.client.get("/api/v1/drift/univariate").json()
        self.assertGreater(len(uni), 0)
        self.assertIn("running_mean", uni[0])
        self.assertIn("tdigest_median", uni[0])
        self.assertIn("tdigest_p95", uni[0])

    def test_04_benchmarks_api_connected_to_ml_engine(self):
        """Verify ML Engine benchmark reports are directly accessible via FastAPI."""
        res = self.client.get("/api/v1/benchmarks/results")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "AVAILABLE")
        self.assertGreaterEqual(len(data["detection_delay_benchmarks"]), 4)
        self.assertGreaterEqual(len(data["tdigest_quantile_accuracy"]), 5)
        self.assertIn("Experimental Benchmarking Report", data["markdown_report"])

    def test_05_full_governance_and_dashboard_verification(self):
        """Verify shadow retraining token governance, Prometheus exporter, and unified Dashboard."""
        service = DriftService.get_instance()
        token = f"e2e_token_{uuid.uuid4().hex[:8]}"
        service.db.log_retrain_audit(
            trigger_reason="E2E integrated drift verification",
            urgency="HIGH",
            active_f1=0.70,
            candidate_f1=0.85,
            f1_delta=0.15,
            auc_delta=0.09,
            decision="PROMOTE_TO_PRODUCTION",
            approval_token=token,
            status="PENDING_APPROVAL",
        )
        promo = self.client.post(
            "/api/v1/retrain/approve",
            json={"approval_token": token, "approved_by": "Unified System Reviewer"},
        )
        self.assertEqual(promo.status_code, 200)
        self.assertEqual(promo.json()["status"], "APPROVED")

        # Verify Prometheus metrics and Dashboard HTML include unified tabs
        prom = self.client.get("/metrics")
        self.assertEqual(prom.status_code, 200)
        self.assertIn("drift_records_processed_total", prom.text)

        dash = self.client.get("/dashboard")
        self.assertEqual(dash.status_code, 200)
        self.assertIn("Benchmarks & Eval", dash.text)
        self.assertIn("Count-Min Sketch & HyperLogLog", dash.text)
        self.assertIn("Continuous Streaming Pipeline", dash.text)
        self.assertIn("Upload & Report", dash.text)

    def test_06_csv_upload_and_results_generation(self):
        """Verify CSV dataset upload, sketch streaming, drift evaluation, and sample downloading."""
        sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_drifted_transactions.csv")
        self.assertTrue(os.path.exists(sample_path), "Sample CSV file should exist")

        with open(sample_path, "rb") as f:
            res = self.client.post(
                "/api/v1/stream/upload-csv",
                files={"file": ("drifted_test.csv", f, "text/csv")},
            )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["records_analyzed"], 150)
        self.assertTrue(data["drift_detected"])
        self.assertEqual(data["alert_severity"], "CRITICAL")
        self.assertIsNotNone(data["cbpe_estimated_accuracy"])
        self.assertGreater(len(data["features_analysis"]), 0)

        # Test sample CSV download endpoint
        sample_res = self.client.get("/api/v1/stream/sample-csv/normal")
        self.assertEqual(sample_res.status_code, 200)
        self.assertIn("text/csv", sample_res.headers["content-type"])


if __name__ == "__main__":
    unittest.main()
