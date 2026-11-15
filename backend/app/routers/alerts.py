"""
Alerts & Notifications Router for Person 2 (Backend & Systems Lead).
Provides endpoints to retrieve active alert status, retraining recommendations,
and SQLite historical alert records.
"""

from typing import Any, Dict, List
from fastapi import APIRouter, Depends, Query

from backend.app.schemas.alerts import AlertDetailResponse, RetrainRecommendationResponse
from backend.app.services.alert_service import AlertService
from backend.app.services.drift_service import DriftService

router = APIRouter(prefix="/api/v1/alerts", tags=["Alerts & Notifications"])


def get_drift_service() -> DriftService:
    return DriftService.get_instance()


def get_alert_service(
    service: DriftService = Depends(get_drift_service),
) -> AlertService:
    return AlertService(db_manager=service.db)


@router.get("/active", response_model=AlertDetailResponse, summary="Get Active Alert State")
def get_active_alert(
    service: DriftService = Depends(get_drift_service),
    alerts: AlertService = Depends(get_alert_service),
) -> AlertDetailResponse:
    """
    Returns current active alert status, including composite score,
    severity (NORMAL, WARNING, CRITICAL), and drifted features list.
    """
    raw_alert = service.monitor.latest_alert_status.to_dict()
    return alerts.format_alert_detail(raw_alert)


@router.get("/recommendation", response_model=RetrainRecommendationResponse, summary="Get Retraining Recommendation")
def get_retrain_recommendation(
    service: DriftService = Depends(get_drift_service),
    alerts: AlertService = Depends(get_alert_service),
) -> RetrainRecommendationResponse:
    """
    Returns the formal retraining recommendation:
    urgency level (NONE, LOW, MEDIUM, HIGH), triggered reason, and recommended actions.
    """
    raw_rec = service.monitor.latest_retrain_recommendation.to_dict()
    return alerts.format_retrain_recommendation(raw_rec)


@router.get("/history", summary="Get Historical Alerts")
def get_alert_history(
    limit: int = Query(default=50, ge=1, le=200),
    alerts: AlertService = Depends(get_alert_service),
) -> List[Dict[str, Any]]:
    """
    Returns historical drift alarm log from persistent SQLite storage.
    """
    return alerts.get_historical_alerts(limit=limit)
