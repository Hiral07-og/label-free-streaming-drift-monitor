"""
Drift Monitoring Router for Person 2 (Backend & Systems Lead).
Provides REST endpoints for overall drift status, univariate sketch statistics
(PSI, KS p-value, FDR correction, t-digest quantiles), multivariate tests (MMD, C2ST),
and historical alarm audit events.
"""

from typing import Any, Dict, List
from fastapi import APIRouter, Depends, Query

from backend.app.schemas.drift import (
    DriftStatusResponse,
    MultivariateDriftResponse,
    UnivariateFeatureMetrics,
)
from backend.app.services.drift_service import DriftService

router = APIRouter(prefix="/api/v1/drift", tags=["Drift Monitoring"])


def get_drift_service() -> DriftService:
    return DriftService.get_instance()


@router.get("/status", response_model=DriftStatusResponse, summary="Get Full Drift Status")
def get_drift_status(
    service: DriftService = Depends(get_drift_service),
) -> DriftStatusResponse:
    """
    Returns comprehensive system monitoring state:
    throughput, records processed, sketch memory compression,
    active alert status, and retraining recommendation.
    """
    return service.get_drift_status()


@router.get("/univariate", response_model=List[UnivariateFeatureMetrics], summary="Get Univariate Feature Metrics")
def get_univariate_metrics(
    service: DriftService = Depends(get_drift_service),
) -> List[UnivariateFeatureMetrics]:
    """
    Returns statistical metrics for all monitored numerical features:
    - Population Stability Index (PSI)
    - Two-sample Kolmogorov-Smirnov p-value
    - Benjamini-Hochberg FDR significance flag
    - Running Welford mean
    - Bounded-memory t-digest median and 95th percentile
    """
    return service.get_univariate_metrics()


@router.get("/multivariate", response_model=MultivariateDriftResponse, summary="Get Multivariate Drift Tests")
def get_multivariate_metrics(
    service: DriftService = Depends(get_drift_service),
) -> MultivariateDriftResponse:
    """
    Returns non-parametric Maximum Mean Discrepancy (MMD) with permutation test
    and Classifier Two-Sample Test (C2ST) ROC-AUC over reservoir sample tuples.
    """
    return service.get_multivariate_metrics()


@router.get("/events", summary="Get Historical Drift Alarm Events")
def get_drift_events(
    limit: int = Query(default=50, ge=1, le=200),
    service: DriftService = Depends(get_drift_service),
) -> List[Dict[str, Any]]:
    """
    Retrieves historical drift alarms logged in persistent SQLite storage.
    """
    return service.db.get_drift_events(limit=limit)
