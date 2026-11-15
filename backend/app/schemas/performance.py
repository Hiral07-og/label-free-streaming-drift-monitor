"""
Pydantic schemas for Confidence-Based Performance Estimation (CBPE)
and delayed ground-truth reconciliation.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class PerformanceEstimateResponse(BaseModel):
    """CBPE label-free performance metrics."""
    sample_size: int = 0
    estimated_accuracy: float = 0.0
    estimated_precision: float = 0.0
    estimated_recall: float = 0.0
    estimated_f1: float = 0.0
    estimated_roc_auc: float = 0.0
    estimated_brier_score: float = 0.0
    expected_positive_rate: float = 0.0
    mean_confidence: float = 1.0
    decision_threshold: float = 0.5
    baseline_accuracy: Optional[float] = 0.0
    baseline_f1: Optional[float] = 0.0
    baseline_roc_auc: Optional[float] = 0.0
    accuracy_drop: Optional[float] = 0.0
    f1_drop: Optional[float] = 0.0
    roc_auc_drop: Optional[float] = 0.0
    performance_degraded: Optional[bool] = False


class DelayedLabelItem(BaseModel):
    record_id: str
    is_fraud: int


class DelayedLabelsRequest(BaseModel):
    """Batch of delayed ground-truth labels as a list or mapping."""
    labels: Union[List[DelayedLabelItem], Dict[str, int], List[Dict[str, Any]]] = Field(
        ..., description="List of items or mapping of record_id -> label"
    )


class ComparisonMetricItem(BaseModel):
    """Pair of estimated vs actual metric with error."""
    estimated: float
    actual: float
    absolute_error: float


class DelayedReconciliationResponse(BaseModel):
    """Reconciliation result comparing CBPE estimates with delayed truth."""
    matched_labels: int = 0
    reconciled_count: int = 0
    sample_size: Optional[int] = 0
    estimated_metrics: Optional[Dict[str, Any]] = None
    actual_metrics: Optional[Dict[str, Any]] = None
    estimation_errors: Optional[Dict[str, Any]] = None
    within_tolerance: Optional[bool] = True
    latest_comparison: Optional[Dict[str, Any]] = None
    aggregate_errors: Optional[Dict[str, Any]] = None
