"""
Alert Engine for evaluating multi-signal drift evidence and assigning severity levels:
- NORMAL: System operating in-distribution with expected statistical stability.
- WARNING: Early drift signals detected (moderate univariate shift or slight degradation).
- CRITICAL: Severe multivariate or multi-feature drift confirmed with performance deterioration.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
from ml_engine.drift_monitor.config import AlertSeverity, DriftType, RetrainConfig
from ml_engine.drift_monitor.detectors.c2st import C2STResult
from ml_engine.drift_monitor.detectors.distribution_tests import KSTestResult, PSIResult
from ml_engine.drift_monitor.detectors.mmd import MMDResult
from ml_engine.drift_monitor.performance_estimation.cbpe import CBPEPerformanceEstimate


@dataclass
class AlertStatus:
    """Consolidated alert evaluation result across all monitoring signals."""
    severity: AlertSeverity
    drift_type: DriftType
    composite_score: float  # 0.0 to 1.0
    summary_message: str
    drifted_features: List[str]
    univariate_drift_ratio: float
    multivariate_drift_detected: bool
    performance_degraded: bool
    estimated_metrics_drop: Dict[str, float] = field(default_factory=dict)
    detailed_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "drift_type": self.drift_type.value,
            "composite_score": round(self.composite_score, 4),
            "summary_message": self.summary_message,
            "drifted_features": self.drifted_features,
            "univariate_drift_ratio": round(self.univariate_drift_ratio, 4),
            "multivariate_drift_detected": self.multivariate_drift_detected,
            "performance_degraded": self.performance_degraded,
            "estimated_metrics_drop": {k: round(v, 4) for k, v in self.estimated_metrics_drop.items()},
            "detailed_reasons": self.detailed_reasons,
        }


class DriftAlertEngine:
    """
    Evaluates univariate tests, multivariate tests, and CBPE performance estimates
    to produce composite alerts with severity classifications.
    """

    def __init__(self, config: Optional[RetrainConfig] = None):
        self.config = config or RetrainConfig()

    def evaluate(
        self,
        univariate_results: Dict[str, Dict[str, Any]],
        multivariate_mmd: Optional[MMDResult] = None,
        multivariate_c2st: Optional[C2STResult] = None,
        cbpe_estimate: Optional[CBPEPerformanceEstimate] = None,
        baseline_metrics: Optional[Dict[str, float]] = None,
    ) -> AlertStatus:
        """
        Combines all monitoring signals to determine severity and composite risk.
        """
        reasons: List[str] = []
        drifted_features: List[str] = []

        total_features = max(1, len(univariate_results))
        for feat, res in univariate_results.items():
            if res.get("is_significant", False) or res.get("drift_detected", False):
                drifted_features.append(feat)

        drift_ratio = len(drifted_features) / total_features

        # 1. Multivariate drift evaluation
        mmd_drift = multivariate_mmd.drift_detected if multivariate_mmd is not None else False
        c2st_drift = multivariate_c2st.drift_detected if multivariate_c2st is not None else False
        multi_drift = mmd_drift or c2st_drift

        if mmd_drift:
            reasons.append(f"MMD multivariate drift detected (p={multivariate_mmd.p_value:.4f})")
        if c2st_drift:
            reasons.append(f"C2ST domain discriminator detected distribution shift (AUC={multivariate_c2st.roc_auc:.3f})")

        # 2. Performance degradation evaluation
        perf_degraded = False
        drops: Dict[str, float] = {}

        if cbpe_estimate is not None and baseline_metrics is not None:
            base_acc = baseline_metrics.get("accuracy", 0.0)
            base_f1 = baseline_metrics.get("f1", 0.0)
            base_auc = baseline_metrics.get("roc_auc", 0.0)

            acc_drop = base_acc - cbpe_estimate.estimated_accuracy
            f1_drop = base_f1 - cbpe_estimate.estimated_f1
            auc_drop = base_auc - cbpe_estimate.estimated_roc_auc

            drops["accuracy_drop"] = float(acc_drop)
            drops["f1_drop"] = float(f1_drop)
            drops["auc_drop"] = float(auc_drop)

            if acc_drop >= self.config.max_estimated_accuracy_drop:
                perf_degraded = True
                reasons.append(f"Estimated accuracy dropped by {acc_drop:.1%} (limit {self.config.max_estimated_accuracy_drop:.1%})")
            if f1_drop >= self.config.max_estimated_f1_drop:
                perf_degraded = True
                reasons.append(f"Estimated F1-score dropped by {f1_drop:.1%} (limit {self.config.max_estimated_f1_drop:.1%})")
            if auc_drop >= self.config.max_estimated_auc_drop:
                perf_degraded = True
                reasons.append(f"Estimated ROC-AUC dropped by {auc_drop:.1%} (limit {self.config.max_estimated_auc_drop:.1%})")

        # 3. Compute composite risk score [0.0, 1.0]
        # Weighting: univariate ratio (35%), multivariate (35%), performance drop (30%)
        multi_component = 1.0 if (mmd_drift and c2st_drift) else (0.6 if multi_drift else 0.0)
        perf_component = 1.0 if perf_degraded else max(0.0, min(1.0, drops.get("f1_drop", 0.0) / 0.10))
        composite_score = float(np.clip(
            0.35 * drift_ratio + 0.35 * multi_component + 0.30 * perf_component,
            0.0,
            1.0,
        ))

        # 4. Severity classification
        if (
            (drift_ratio >= self.config.critical_drift_ratio and perf_degraded)
            or (multi_drift and perf_degraded)
            or (drift_ratio >= 0.50)
            or (drops.get("f1_drop", 0.0) >= 0.15)
        ):
            severity = AlertSeverity.CRITICAL
            drift_type = DriftType.COMBINED if (multi_drift and perf_degraded) else DriftType.MULTIVARIATE
            summary = f"CRITICAL alert: Significant drift across {len(drifted_features)} features with confirmed performance degradation."
        elif (
            drift_ratio >= self.config.warning_drift_ratio
            or multi_drift
            or perf_degraded
        ):
            severity = AlertSeverity.WARNING
            if perf_degraded and not (drift_ratio > 0 or multi_drift):
                drift_type = DriftType.PERFORMANCE
            elif multi_drift:
                drift_type = DriftType.MULTIVARIATE
            else:
                drift_type = DriftType.UNIVARIATE
            summary = f"WARNING alert: Emerging drift detected ({len(drifted_features)} features drifted, multi={multi_drift})."
        else:
            severity = AlertSeverity.NORMAL
            drift_type = DriftType.NONE
            summary = "NORMAL: Streaming distribution remains aligned with baseline reference."

        if not reasons:
            reasons.append("All monitoring metrics are within acceptable baseline bounds.")

        return AlertStatus(
            severity=severity,
            drift_type=drift_type,
            composite_score=composite_score,
            summary_message=summary,
            drifted_features=drifted_features,
            univariate_drift_ratio=drift_ratio,
            multivariate_drift_detected=multi_drift,
            performance_degraded=perf_degraded,
            estimated_metrics_drop=drops,
            detailed_reasons=reasons,
        )
