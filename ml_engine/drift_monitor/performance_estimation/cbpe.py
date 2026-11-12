"""
Confidence-Based Performance Estimation (CBPE) module.
Estimates model accuracy, precision, recall, F1-score, ROC-AUC, and Brier score
on incoming unlabeled streaming predictions without requiring ground-truth labels,
under the Covariate Shift assumption and probability calibration.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class CBPEPerformanceEstimate:
    """Estimated performance metrics produced without ground-truth labels."""
    sample_size: int
    estimated_accuracy: float
    estimated_precision: float
    estimated_recall: float
    estimated_f1: float
    estimated_roc_auc: float
    estimated_brier_score: float
    expected_positive_rate: float
    mean_confidence: float
    decision_threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "estimated_accuracy": round(self.estimated_accuracy, 4),
            "estimated_precision": round(self.estimated_precision, 4),
            "estimated_recall": round(self.estimated_recall, 4),
            "estimated_f1": round(self.estimated_f1, 4),
            "estimated_roc_auc": round(self.estimated_roc_auc, 4),
            "estimated_brier_score": round(self.estimated_brier_score, 4),
            "expected_positive_rate": round(self.expected_positive_rate, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "decision_threshold": self.decision_threshold,
        }


class ConfidenceBasedPerformanceEstimator:
    """
    Confidence-Based Performance Estimator (CBPE) for binary classification.
    Requires well-calibrated output probabilities: p(x) ~ P(Y=1 | X=x).
    
    Theoretical Assumptions:
    1. Model probabilities are well-calibrated (calibrated via Platt/Isotonic scaling on reference).
    2. Covariate Shift holds: P_stream(Y|X) = P_ref(Y|X).
       If concept drift occurs purely in P(Y|X) with identical feature distributions,
       label-free estimation cannot observe this without delayed labels.
    """

    def __init__(
        self,
        decision_threshold: float = 0.5,
        clip_bounds: Tuple[float, float] = (1e-6, 1.0 - 1e-6),
    ):
        self.decision_threshold = decision_threshold
        self.clip_min, self.clip_max = clip_bounds

    def estimate(
        self,
        probabilities: Union[List[float], np.ndarray],
        subsample_auc: int = 500,
    ) -> CBPEPerformanceEstimate:
        """
        Estimates evaluation metrics from a collection of predicted probabilities.
        """
        probs = np.asarray(probabilities, dtype=np.float64).flatten()
        probs = probs[~np.isnan(probs)]
        n = len(probs)

        if n == 0:
            return CBPEPerformanceEstimate(
                sample_size=0,
                estimated_accuracy=0.0,
                estimated_precision=0.0,
                estimated_recall=0.0,
                estimated_f1=0.0,
                estimated_roc_auc=0.5,
                estimated_brier_score=0.0,
                expected_positive_rate=0.0,
                mean_confidence=0.0,
                decision_threshold=self.decision_threshold,
            )

        # Clip probabilities for numerical stability
        p = np.clip(probs, self.clip_min, self.clip_max)
        preds = (p >= self.decision_threshold).astype(int)

        # Expected confusion matrix components
        # E[TP] = sum_{y_hat=1} p_i
        e_tp = float(np.sum(p[preds == 1])) if np.any(preds == 1) else 0.0
        # E[FP] = sum_{y_hat=1} (1 - p_i)
        e_fp = float(np.sum(1.0 - p[preds == 1])) if np.any(preds == 1) else 0.0
        # E[TN] = sum_{y_hat=0} (1 - p_i)
        e_tn = float(np.sum(1.0 - p[preds == 0])) if np.any(preds == 0) else 0.0
        # E[FN] = sum_{y_hat=0} p_i
        e_fn = float(np.sum(p[preds == 0])) if np.any(preds == 0) else 0.0

        # Expected Accuracy: (E[TP] + E[TN]) / N
        est_acc = (e_tp + e_tn) / n

        # Expected Precision: E[TP] / (E[TP] + E[FP]) = sum_{y_hat=1} p_i / sum(y_hat)
        pred_positives = float(np.sum(preds))
        est_prec = (e_tp / pred_positives) if pred_positives > 0 else 0.0

        # Expected Recall: E[TP] / (E[TP] + E[FN]) = sum_{y_hat=1} p_i / sum(p_i)
        total_expected_positives = float(np.sum(p))
        est_rec = (e_tp / total_expected_positives) if total_expected_positives > 0 else 0.0

        # Expected F1: harmonic mean of estimated precision and recall
        if (est_prec + est_rec) > 0:
            est_f1 = 2.0 * (est_prec * est_rec) / (est_prec + est_rec)
        else:
            est_f1 = 0.0

        # Expected Brier score: E[(Y - p)^2] = p * (1 - p) under calibration
        est_brier = float(np.mean(p * (1.0 - p)))

        # Expected ROC-AUC via Wilcoxon-Mann-Whitney probabilistic expectation
        est_auc = self._estimate_expected_roc_auc(p, max_samples=subsample_auc)

        # Confidence: max(p, 1 - p)
        confidence = np.maximum(p, 1.0 - p)
        mean_conf = float(np.mean(confidence))
        exp_pos_rate = float(np.mean(p))

        return CBPEPerformanceEstimate(
            sample_size=n,
            estimated_accuracy=float(est_acc),
            estimated_precision=float(est_prec),
            estimated_recall=float(est_rec),
            estimated_f1=float(est_f1),
            estimated_roc_auc=float(est_auc),
            estimated_brier_score=float(est_brier),
            expected_positive_rate=exp_pos_rate,
            mean_confidence=mean_conf,
            decision_threshold=self.decision_threshold,
        )

    def _estimate_expected_roc_auc(self, p: np.ndarray, max_samples: int = 500) -> float:
        """
        Estimates expected ROC-AUC using pairwise probability expectation:
        E[AUC] = sum_{i != j} I(p_i > p_j) * p_i * (1 - p_j) / sum_{i != j} p_i * (1 - p_j)
        """
        n = len(p)
        if n < 4:
            return 0.5

        # Subsample if n is large for O(k^2) efficiency
        if n > max_samples:
            idx = np.random.default_rng(42).choice(n, max_samples, replace=False)
            p_sub = p[idx]
        else:
            p_sub = p

        m = len(p_sub)
        p_row = p_sub.reshape(1, m)
        p_col = p_sub.reshape(m, 1)

        # Pairwise indicator: 1 if p_i > p_j, 0.5 if tie
        greater_matrix = (p_col > p_row).astype(np.float64) + 0.5 * (p_col == p_row).astype(np.float64)
        # Weight = p_i * (1 - p_j) representing P(Y_i=1 and Y_j=0)
        weight_matrix = p_col * (1.0 - p_row)

        # Zero out diagonal
        np.fill_diagonal(greater_matrix, 0.0)
        np.fill_diagonal(weight_matrix, 0.0)

        total_weight = np.sum(weight_matrix)
        if total_weight <= 1e-9:
            return 0.5

        expected_auc = np.sum(greater_matrix * weight_matrix) / total_weight
        return float(np.clip(expected_auc, 0.0, 1.0))
