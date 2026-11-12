"""
Delayed ground-truth evaluator and estimation error tracker.
Matches incoming predictions with delayed labels when they arrive,
calculates actual performance metrics, and measures estimation error
(MAE, RMSE, bias) of CBPE.
"""

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml_engine.drift_monitor.performance_estimation.cbpe import (
    CBPEPerformanceEstimate,
    ConfidenceBasedPerformanceEstimator,
)


@dataclass
class PerformanceComparisonReport:
    """Comparison between CBPE label-free estimate and actual ground truth."""
    sample_size: int
    estimated: CBPEPerformanceEstimate
    actual_accuracy: float
    actual_precision: float
    actual_recall: float
    actual_f1: float
    actual_roc_auc: float
    actual_brier_score: float
    error_accuracy: float
    error_precision: float
    error_recall: float
    error_f1: float
    error_roc_auc: float
    error_brier: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "estimated": self.estimated.to_dict(),
            "actual": {
                "accuracy": round(self.actual_accuracy, 4),
                "precision": round(self.actual_precision, 4),
                "recall": round(self.actual_recall, 4),
                "f1": round(self.actual_f1, 4),
                "roc_auc": round(self.actual_roc_auc, 4),
                "brier_score": round(self.actual_brier_score, 4),
            },
            "error_mae": {
                "accuracy": round(abs(self.error_accuracy), 4),
                "precision": round(abs(self.error_precision), 4),
                "recall": round(abs(self.error_recall), 4),
                "f1": round(abs(self.error_f1), 4),
                "roc_auc": round(abs(self.error_roc_auc), 4),
                "brier": round(abs(self.error_brier), 4),
            },
            "error_signed_bias": {
                "accuracy": round(self.error_accuracy, 4),
                "f1": round(self.error_f1, 4),
                "roc_auc": round(self.error_roc_auc, 4),
            },
        }


class DelayedGroundTruthEvaluator:
    """
    Manages pending records waiting for delayed labels and computes
    continuous estimation error metrics.
    """

    def __init__(
        self,
        cbpe_estimator: Optional[ConfidenceBasedPerformanceEstimator] = None,
        max_buffer_size: int = 50000,
    ):
        self.cbpe = cbpe_estimator or ConfidenceBasedPerformanceEstimator()
        self.max_buffer_size = max_buffer_size

        # Buffer: record_id -> {"prob": float, "pred": int, "timestamp": Any}
        self.pending_records: Dict[str, Dict[str, Any]] = {}
        # Completed pairs: (prob, pred, actual_label)
        self.matched_history: Deque[Tuple[float, int, int]] = deque(maxlen=max_buffer_size)

        # Historical comparison reports
        self.comparison_history: List[PerformanceComparisonReport] = []

    def record_prediction(
        self,
        record_id: str,
        probability: float,
        prediction: int,
        timestamp: Optional[Any] = None,
    ) -> None:
        """Stores streaming prediction awaiting delayed ground truth."""
        if len(self.pending_records) >= self.max_buffer_size:
            # Drop oldest key
            oldest = next(iter(self.pending_records))
            del self.pending_records[oldest]

        self.pending_records[str(record_id)] = {
            "prob": float(probability),
            "pred": int(prediction),
            "timestamp": timestamp,
        }

    def receive_label(self, record_id: str, actual_label: int) -> bool:
        """
        Receives ground-truth label for a specific record ID.
        Returns True if matching prediction was found and paired, False otherwise.
        """
        rec_id = str(record_id)
        if rec_id in self.pending_records:
            info = self.pending_records.pop(rec_id)
            self.matched_history.append((info["prob"], info["pred"], int(actual_label)))
            return True
        return False

    def receive_labels_batch(self, ground_truth_map: Dict[str, int]) -> int:
        """Pairs multiple delayed labels in batch."""
        matched = 0
        for rid, label in ground_truth_map.items():
            if self.receive_label(rid, label):
                matched += 1
        return matched

    def evaluate_recent_window(self, window_size: Optional[int] = None) -> Optional[PerformanceComparisonReport]:
        """
        Calculates comparison between CBPE estimate and actual performance
        over the recent window of matched records.
        """
        if not self.matched_history:
            return None

        data = list(self.matched_history)
        if window_size is not None and window_size > 0:
            data = data[-window_size:]

        n = len(data)
        if n < 5:
            return None

        probs = np.array([x[0] for x in data], dtype=np.float64)
        preds = np.array([x[1] for x in data], dtype=int)
        actuals = np.array([x[2] for x in data], dtype=int)

        # 1. Label-free CBPE estimate
        estimate = self.cbpe.estimate(probs)

        # 2. Actual ground-truth metrics
        act_acc = float(accuracy_score(actuals, preds))
        act_prec = float(precision_score(actuals, preds, zero_division=0))
        act_rec = float(recall_score(actuals, preds, zero_division=0))
        act_f1 = float(f1_score(actuals, preds, zero_division=0))
        act_brier = float(brier_score_loss(actuals, probs))

        if len(np.unique(actuals)) > 1:
            act_auc = float(roc_auc_score(actuals, probs))
        else:
            act_auc = 0.5

        # 3. Compute errors
        err_acc = estimate.estimated_accuracy - act_acc
        err_prec = estimate.estimated_precision - act_prec
        err_rec = estimate.estimated_recall - act_rec
        err_f1 = estimate.estimated_f1 - act_f1
        err_auc = estimate.estimated_roc_auc - act_auc
        err_brier = estimate.estimated_brier_score - act_brier

        report = PerformanceComparisonReport(
            sample_size=n,
            estimated=estimate,
            actual_accuracy=act_acc,
            actual_precision=act_prec,
            actual_recall=act_rec,
            actual_f1=act_f1,
            actual_roc_auc=act_auc,
            actual_brier_score=act_brier,
            error_accuracy=err_acc,
            error_precision=err_prec,
            error_recall=err_rec,
            error_f1=err_f1,
            error_roc_auc=err_auc,
            error_brier=err_brier,
        )

        self.comparison_history.append(report)
        return report

    def get_aggregate_estimation_errors(self) -> Dict[str, Any]:
        """Computes aggregate MAE, RMSE, and bias across all historical reports."""
        if not self.comparison_history:
            return {
                "n_evaluations": 0,
                "mae_accuracy": 0.0,
                "mae_f1": 0.0,
                "mae_roc_auc": 0.0,
                "bias_accuracy": 0.0,
            }

        acc_errs = [r.error_accuracy for r in self.comparison_history]
        f1_errs = [r.error_f1 for r in self.comparison_history]
        auc_errs = [r.error_roc_auc for r in self.comparison_history]

        return {
            "n_evaluations": len(self.comparison_history),
            "mae_accuracy": round(float(np.mean(np.abs(acc_errs))), 4),
            "rmse_accuracy": round(float(np.sqrt(np.mean(np.square(acc_errs)))), 4),
            "bias_accuracy": round(float(np.mean(acc_errs)), 4),
            "mae_f1": round(float(np.mean(np.abs(f1_errs))), 4),
            "mae_roc_auc": round(float(np.mean(np.abs(auc_errs))), 4),
        }
