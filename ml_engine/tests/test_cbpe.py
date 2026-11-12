"""
Unit tests for Confidence-Based Performance Estimation (CBPE) and delayed evaluation:
- Probabilistic expectations (Accuracy, F1, ROC-AUC, Brier)
- Error metrics vs delayed ground-truth arrival (MAE, RMSE, Bias)
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from ml_engine.drift_monitor.performance_estimation.cbpe import (
    ConfidenceBasedPerformanceEstimator,
)
from ml_engine.drift_monitor.performance_estimation.delayed_evaluator import (
    DelayedGroundTruthEvaluator,
)


class TestCBPE(unittest.TestCase):
    """Test suite for label-free performance estimation."""

    def test_cbpe_unbiased_expectations(self):
        """
        Under exact calibration (Y ~ Bernoulli(p)), CBPE estimates
        should converge to true empirical sample metrics.
        """
        rng = np.random.default_rng(42)
        n = 3000

        # Generate well-calibrated probabilities
        probs = rng.beta(a=2.0, b=5.0, size=n)
        # Generate true labels from these exact probabilities
        y_true = (rng.uniform(0.0, 1.0, size=n) < probs).astype(int)
        preds = (probs >= 0.5).astype(int)

        # True metrics
        actual_acc = accuracy_score(y_true, preds)
        actual_f1 = f1_score(y_true, preds, zero_division=0)
        actual_auc = roc_auc_score(y_true, probs)

        # CBPE estimates without seeing y_true!
        cbpe = ConfidenceBasedPerformanceEstimator(decision_threshold=0.5)
        est = cbpe.estimate(probs)

        # Difference should be very small (within Monte Carlo sampling error ~2-3%)
        self.assertLess(abs(est.estimated_accuracy - actual_acc), 0.03)
        self.assertLess(abs(est.estimated_f1 - actual_f1), 0.04)
        self.assertLess(abs(est.estimated_roc_auc - actual_auc), 0.04)

    def test_delayed_evaluator_matching(self):
        """Validates record matching and continuous estimation error tracking."""
        evaluator = DelayedGroundTruthEvaluator()

        for i in range(100):
            evaluator.record_prediction(
                record_id=f"rec_{i}",
                probability=0.85 if i % 2 == 0 else 0.15,
                prediction=1 if i % 2 == 0 else 0,
            )

        # Send delayed ground truth for 60 records
        labels = {f"rec_{i}": (1 if i % 2 == 0 else 0) for i in range(60)}
        matched = evaluator.receive_labels_batch(labels)
        self.assertEqual(matched, 60)

        report = evaluator.evaluate_recent_window(window_size=50)
        self.assertIsNotNone(report)
        self.assertEqual(report.sample_size, 50)
        self.assertEqual(report.actual_accuracy, 1.0)

        agg = evaluator.get_aggregate_estimation_errors()
        self.assertEqual(agg["n_evaluations"], 1)
        self.assertIn("mae_accuracy", agg)


if __name__ == "__main__":
    unittest.main()
