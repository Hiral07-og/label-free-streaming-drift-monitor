"""
Alert Service for Person 2 (Backend & Systems Lead).
Manages alert state retrieval, active alert summaries, notification dispatching,
and historical alert audits from SQLite.
"""

from typing import Any, Dict, List, Optional
from backend.app.schemas.alerts import AlertDetailResponse, RetrainRecommendationResponse
from backend.app.storage.database import DatabaseManager


class AlertService:
    """Service handling alert state, notifications, and historical queries."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def format_alert_detail(self, alert_data: Dict[str, Any]) -> AlertDetailResponse:
        """Converts raw monitor alert dictionary into a typed AlertDetailResponse."""
        return AlertDetailResponse(
            severity=alert_data.get("severity", "NORMAL"),
            drift_type=alert_data.get("drift_type", "NONE"),
            composite_score=float(alert_data.get("composite_score", 0.0)),
            summary_message=alert_data.get("summary_message", "All signals stationary."),
            drifted_features=alert_data.get("drifted_features", []),
            univariate_drift_ratio=float(alert_data.get("univariate_drift_ratio", 0.0)),
            multivariate_drift_detected=bool(alert_data.get("multivariate_drift_detected", False)),
            performance_degraded=bool(alert_data.get("performance_degraded", False)),
            estimated_metrics_drop=alert_data.get("estimated_metrics_drop", {}),
            detailed_reasons=alert_data.get("detailed_reasons", []),
        )

    def format_retrain_recommendation(
        self, rec_data: Dict[str, Any]
    ) -> RetrainRecommendationResponse:
        """Converts raw retraining recommendation dict into a typed response."""
        return RetrainRecommendationResponse(
            should_retrain=bool(rec_data.get("should_retrain", False)),
            urgency=rec_data.get("urgency", "NONE"),
            severity=rec_data.get("severity", "NORMAL"),
            trigger_reason=rec_data.get("trigger_reason", "No significant drift detected."),
            affected_features=rec_data.get("affected_features", []),
            estimated_impact=rec_data.get("estimated_impact", {}),
            recommended_action=rec_data.get("recommended_action", "Continue standard monitoring."),
        )

    def get_historical_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetches historical drift events logged in database."""
        return self.db.get_drift_events(limit=limit)

    def get_latest_alert(self) -> Optional[Dict[str, Any]]:
        """Retrieves most recent alert from persistent database."""
        return self.db.get_latest_alert()
