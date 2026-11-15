"""
Performance Estimation & Ground Truth Reconciliation Router for Person 2.
Exposes Confidence-Based Performance Estimation (CBPE) predictions and
delayed ground truth label reconciliation endpoints.
"""

from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.schemas.performance import (
    DelayedLabelsRequest,
    DelayedReconciliationResponse,
    PerformanceEstimateResponse,
)
from backend.app.services.drift_service import DriftService

router = APIRouter(prefix="/api/v1/performance", tags=["Performance Estimation"])


def get_drift_service() -> DriftService:
    return DriftService.get_instance()


@router.get("/estimated", response_model=PerformanceEstimateResponse, summary="Get CBPE Estimated Performance")
def get_performance_estimate(
    service: DriftService = Depends(get_drift_service),
) -> PerformanceEstimateResponse:
    """
    Returns estimated accuracy, precision, recall, F1-score, and ROC-AUC
    computed without ground truth labels using model confidence distributions.
    """
    return service.get_performance_estimate()


@router.post("/labels", response_model=DelayedReconciliationResponse, summary="Ingest Delayed Ground Truth")
def reconcile_delayed_labels(
    request: DelayedLabelsRequest,
    service: DriftService = Depends(get_drift_service),
) -> DelayedReconciliationResponse:
    """
    Ingests delayed actual labels (ground truth), reconciles them against earlier
    CBPE confidence estimates, measures estimation error, and logs audit record.
    """
    try:
        if isinstance(request.labels, list):
            raw_items = [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in request.labels
            ]
        elif isinstance(request.labels, dict):
            raw_items = request.labels
        else:
            raw_items = request.labels
        return service.reconcile_delayed_labels(raw_items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reconcile delayed labels: {str(e)}")


@router.get("/history", summary="Get Historical Performance Evaluations")
def get_performance_history(
    limit: int = Query(default=50, ge=1, le=200),
    service: DriftService = Depends(get_drift_service),
) -> List[Dict[str, Any]]:
    """
    Returns historical comparison evaluations between CBPE estimates
    and delayed actual ground truth metrics stored in SQLite.
    """
    return service.db.get_performance_evaluations(limit=limit)
