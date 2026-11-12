"""
Automated Retraining Trigger and Recommendation Generator.
Translates alert states and performance signals into governed retraining recommendations
without automatically deploying untested models to production.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from ml_engine.drift_monitor.config import AlertSeverity, RetrainConfig, RetrainUrgency
from ml_engine.drift_monitor.retraining.alert_engine import AlertStatus


@dataclass
class RetrainingRecommendation:
    """Formal recommendation for model retraining and shadow testing."""
    should_retrain: bool
    urgency: RetrainUrgency
    severity: AlertSeverity
    trigger_reason: str
    affected_features: List[str]
    estimated_impact: Dict[str, float]
    recommended_action: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_retrain": self.should_retrain,
            "urgency": self.urgency.value,
            "severity": self.severity.value,
            "trigger_reason": self.trigger_reason,
            "affected_features": self.affected_features,
            "estimated_impact": {k: round(v, 4) for k, v in self.estimated_impact.items()},
            "recommended_action": self.recommended_action,
        }


class RetrainingTrigger:
    """
    Decides whether a retraining workflow should be scheduled and with what urgency.
    """

    def __init__(self, config: Optional[RetrainConfig] = None):
        self.config = config or RetrainConfig()

    def evaluate(
        self,
        alert_status: AlertStatus,
        recent_records_count: int,
    ) -> RetrainingRecommendation:
        """
        Evaluates alert status and window volume to produce governed retraining decision.
        """
        if recent_records_count < self.config.min_window_for_retrain:
            return RetrainingRecommendation(
                should_retrain=False,
                urgency=RetrainUrgency.NONE,
                severity=alert_status.severity,
                trigger_reason=f"Insufficient streaming volume ({recent_records_count} < {self.config.min_window_for_retrain} required).",
                affected_features=alert_status.drifted_features,
                estimated_impact=alert_status.estimated_metrics_drop,
                recommended_action="Continue accumulating streaming records before initiating retraining evaluation.",
            )

        if alert_status.severity == AlertSeverity.CRITICAL:
            return RetrainingRecommendation(
                should_retrain=True,
                urgency=RetrainUrgency.HIGH,
                severity=AlertSeverity.CRITICAL,
                trigger_reason="CRITICAL drift confirmed with estimated performance deterioration.",
                affected_features=alert_status.drifted_features,
                estimated_impact=alert_status.estimated_metrics_drop,
                recommended_action="Initiate shadow model training immediately on recent streaming data and validate against active model.",
            )

        elif alert_status.severity == AlertSeverity.WARNING:
            if alert_status.performance_degraded or alert_status.multivariate_drift_detected:
                return RetrainingRecommendation(
                    should_retrain=True,
                    urgency=RetrainUrgency.MEDIUM,
                    severity=AlertSeverity.WARNING,
                    trigger_reason="WARNING: Emerging multivariate drift or slight performance dip observed.",
                    affected_features=alert_status.drifted_features,
                    estimated_impact=alert_status.estimated_metrics_drop,
                    recommended_action="Prepare candidate model in shadow mode; monitor next evaluation interval.",
                )
            else:
                return RetrainingRecommendation(
                    should_retrain=False,
                    urgency=RetrainUrgency.LOW,
                    severity=AlertSeverity.WARNING,
                    trigger_reason="WARNING: Mild univariate drift detected with stable performance confidence.",
                    affected_features=alert_status.drifted_features,
                    estimated_impact=alert_status.estimated_metrics_drop,
                    recommended_action="Inspect data pipeline for affected features. Retraining not currently necessary.",
                )

        else:
            return RetrainingRecommendation(
                should_retrain=False,
                urgency=RetrainUrgency.NONE,
                severity=AlertSeverity.NORMAL,
                trigger_reason="Data distribution and performance are stable.",
                affected_features=[],
                estimated_impact={},
                recommended_action="Maintain standard monitoring frequency.",
            )
