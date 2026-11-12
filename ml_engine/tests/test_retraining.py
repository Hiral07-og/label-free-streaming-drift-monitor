"""
Unit tests for Alerting and Retraining Governance:
- Alert severity levels (NORMAL, WARNING, CRITICAL)
- Retraining recommendations and triggers
- Shadow model evaluation and approval token generation
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd

from ml_engine.drift_monitor.config import AlertSeverity, RetrainConfig, RetrainUrgency
from ml_engine.drift_monitor.dataset import StreamingFinancialBenchmark
from ml_engine.drift_monitor.detectors.mmd import MMDResult
from ml_engine.drift_monitor.model import BaselineModelWrapper
from ml_engine.drift_monitor.performance_estimation.cbpe import CBPEPerformanceEstimate
from ml_engine.drift_monitor.retraining.alert_engine import DriftAlertEngine
from ml_engine.drift_monitor.retraining.retrain_trigger import RetrainingTrigger
from ml_engine.drift_monitor.retraining.shadow_evaluator import ShadowModelEvaluator


class TestRetraining(unittest.TestCase):
    """Test suite for alert engine and retraining governance."""

    def test_alert_engine_normal_vs_critical(self):
        """Validates alert severity assignment."""
        engine = DriftAlertEngine()

        # 1. Normal state
        univariate_stable = {
            "feat1": {"is_significant": False, "drift_detected": False},
            "feat2": {"is_significant": False, "drift_detected": False},
        }
        status_normal = engine.evaluate(univariate_results=univariate_stable)
        self.assertEqual(status_normal.severity, AlertSeverity.NORMAL)

        # 2. Critical state: severe multivariate drift + performance drop
        mmd_critical = MMDResult(
            mmd_squared=0.08,
            p_value=0.001,
            drift_detected=True,
            gamma=0.5,
            alpha=0.05,
            sample_size_ref=200,
            sample_size_cur=200,
        )
        cbpe_degraded = CBPEPerformanceEstimate(
            sample_size=300,
            estimated_accuracy=0.72,
            estimated_precision=0.60,
            estimated_recall=0.50,
            estimated_f1=0.54,
            estimated_roc_auc=0.68,
            estimated_brier_score=0.18,
            expected_positive_rate=0.10,
            mean_confidence=0.70,
            decision_threshold=0.5,
        )
        baseline = {"accuracy": 0.88, "f1": 0.78, "roc_auc": 0.85}

        status_critical = engine.evaluate(
            univariate_results=univariate_stable,
            multivariate_mmd=mmd_critical,
            cbpe_estimate=cbpe_degraded,
            baseline_metrics=baseline,
        )
        self.assertEqual(status_critical.severity, AlertSeverity.CRITICAL)
        self.assertTrue(status_critical.performance_degraded)

    def test_retraining_trigger(self):
        """Validates retraining trigger logic and minimum window guardrail."""
        trigger = RetrainingTrigger()

        # Insufficient volume guardrail
        status = DriftAlertEngine().evaluate(univariate_results={})
        rec_small = trigger.evaluate(status, recent_records_count=50)
        self.assertFalse(rec_small.should_retrain)
        self.assertEqual(rec_small.urgency, RetrainUrgency.NONE)

    def test_shadow_model_evaluation(self):
        """Validates candidate shadow model training and comparison."""
        bench = StreamingFinancialBenchmark(random_state=42)
        split = bench.generate(n_reference=500, n_stream=500)

        active_model = BaselineModelWrapper(
            numerical_features=split.numerical_features,
            categorical_features=split.categorical_features,
        )
        active_model.train_and_calibrate(split.reference_df)

        evaluator = ShadowModelEvaluator(config=RetrainConfig(shadow_val_split=0.3))
        res, candidate = evaluator.evaluate_candidate(
            active_model=active_model,
            recent_data_with_labels=split.stream_df.iloc[:200],
            target_column="is_fraud",
        )

        self.assertIsNotNone(res)
        self.assertIn(res.decision, ["PROMOTE_TO_PRODUCTION", "KEEP_ACTIVE", "NEEDS_HUMAN_REVIEW"])
        self.assertIsNotNone(candidate)


if __name__ == "__main__":
    unittest.main()
