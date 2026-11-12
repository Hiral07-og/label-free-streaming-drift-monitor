"""
End-to-End integration test for the master StreamingDriftMonitor:
- Reference initialization
- Record-by-record streaming ingestion
- Periodic multi-signal drift detection
- Label-free performance estimation
- Delayed ground-truth reconciliation
- Monitoring status summary generation for backend APIs
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd

from ml_engine.drift_monitor.config import MonitorConfig
from ml_engine.drift_monitor.dataset import StreamingFinancialBenchmark
from ml_engine.drift_monitor.monitor import StreamingDriftMonitor


class TestStreamingDriftMonitor(unittest.TestCase):
    """Integration test suite for StreamingDriftMonitor."""

    def setUp(self):
        self.bench = StreamingFinancialBenchmark(random_state=42)
        self.split = self.bench.generate(n_reference=400, n_stream=600)
        self.config = MonitorConfig(window_size=150, check_interval=50)
        self.monitor = StreamingDriftMonitor(config=self.config)

    def test_end_to_end_streaming_pipeline(self):
        """Validates full life-cycle from setup to record streaming and delayed evaluation."""
        # 1. Setup reference
        base_metrics = self.monitor.setup_reference(self.split.reference_df)
        self.assertIsNotNone(base_metrics)
        self.assertTrue(self.monitor.is_initialized)

        # 2. Process records one by one
        records = self.split.stream_df.iloc[:200].to_dict(orient="records")
        for rec in records:
            res = self.monitor.process_record(rec)
            self.assertIn(res.prediction, [0, 1])
            self.assertGreaterEqual(res.probability, 0.0)
            self.assertLessEqual(res.probability, 1.0)

        self.assertEqual(self.monitor.records_processed, 200)

        # 3. Process batch
        batch_df = self.split.stream_df.iloc[200:300]
        batch_results = self.monitor.process_batch(batch_df)
        self.assertEqual(len(batch_results), 100)
        self.assertEqual(self.monitor.records_processed, 300)

        # 4. Ingest delayed ground-truth labels
        delayed_labels = {
            r["record_id"]: int(r["is_fraud"])
            for r in records[:150]
        }
        delayed_res = self.monitor.receive_delayed_labels(delayed_labels)
        self.assertEqual(delayed_res["matched_labels"], 150)
        self.assertIsNotNone(delayed_res["latest_comparison"])

        # 5. Extract monitoring summary for Person 2 (Backend / FastAPI)
        summary = self.monitor.get_monitoring_summary()
        self.assertEqual(summary["status"], "HEALTHY")
        self.assertEqual(summary["records_processed"], 300)
        self.assertIn("memory", summary)
        self.assertIn("alert", summary)
        self.assertIn("retraining_recommendation", summary)
        self.assertIn("performance_estimation", summary)
        self.assertIn("univariate_drift", summary)
        self.assertIn("multivariate_drift", summary)


if __name__ == "__main__":
    unittest.main()
