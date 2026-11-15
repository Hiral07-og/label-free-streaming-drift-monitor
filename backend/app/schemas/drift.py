"""
Pydantic schemas for univariate and multivariate drift monitoring responses.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class UnivariateFeatureMetrics(BaseModel):
    """Drift metrics for an individual monitored feature."""
    feature_name: str
    psi: float = Field(..., description="Population Stability Index")
    ks_p_value: float = Field(..., description="Kolmogorov-Smirnov p-value")
    is_significant: bool = Field(..., description="Significant drift after Benjamini-Hochberg FDR correction")
    running_mean: float = Field(..., description="Welford running mean")
    tdigest_median: float = Field(..., description="t-digest estimated 50th percentile")
    tdigest_p95: float = Field(..., description="t-digest estimated 95th percentile")


class MultivariateMMDMetrics(BaseModel):
    """Maximum Mean Discrepancy (MMD) multivariate test metrics."""
    mmd_squared: float
    p_value: float
    drift_detected: bool
    gamma: float
    alpha: float


class MultivariateC2STMetrics(BaseModel):
    """Classifier Two-Sample Test (C2ST) metrics."""
    roc_auc: float
    p_value: float
    drift_detected: bool
    top_contributing_features: List[List[Any]] = Field(default_factory=list)


class MultivariateDriftResponse(BaseModel):
    """Consolidated multivariate drift summary."""
    status: str = "HEALTHY"
    mmd: Optional[MultivariateMMDMetrics] = None
    c2st: Optional[MultivariateC2STMetrics] = None
    multivariate_drift_confirmed: bool = False


class DriftEventItem(BaseModel):
    """Historical drift alarm event."""
    record_position: int
    timestamp: float
    severity: str
    summary: str
    affected_features: List[str]
    recommendation: Dict[str, Any]


class DriftStatusResponse(BaseModel):
    """Comprehensive drift status and system health summary for dashboard."""
    status: str = "HEALTHY"
    records_processed: int
    throughput_rec_per_sec: float
    uptime_seconds: float
    memory: Dict[str, Any]
    alert: Dict[str, Any]
    retraining_recommendation: Dict[str, Any]
    performance_estimation: Optional[Dict[str, Any]] = None
    baseline_metrics: Optional[Dict[str, Any]] = None
    multivariate_drift: Dict[str, Any]
    univariate_drift: Dict[str, Any]
    categorical_sketches: Dict[str, Any]
    alert_history_count: int
