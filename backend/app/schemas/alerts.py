"""
Pydantic schemas for alerting, retrain triggers, and shadow model governance.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AlertDetailResponse(BaseModel):
    """Current active alert state and classification."""
    severity: str  # NORMAL, WARNING, CRITICAL
    drift_type: str
    composite_score: float
    summary_message: str
    drifted_features: List[str]
    univariate_drift_ratio: float
    multivariate_drift_detected: bool
    performance_degraded: bool
    estimated_metrics_drop: Dict[str, float]
    detailed_reasons: List[str]


class RetrainRecommendationResponse(BaseModel):
    """Formal retraining recommendation and urgency."""
    should_retrain: bool
    urgency: str  # NONE, LOW, MEDIUM, HIGH
    severity: str
    trigger_reason: str
    affected_features: List[str]
    estimated_impact: Dict[str, float]
    recommended_action: str


class CandidateEvaluationRequest(BaseModel):
    """Request payload to train and benchmark candidate shadow model."""
    sample_size: int = Field(default=300, description="Number of recent labeled records to train candidate model")


class CandidateEvaluationResponse(BaseModel):
    """Candidate model benchmark results on hold-out validation."""
    active_metrics: Dict[str, Any]
    candidate_metrics: Dict[str, Any]
    deltas: Dict[str, float]
    decision: str  # PROMOTE_TO_PRODUCTION, KEEP_ACTIVE, NEEDS_HUMAN_REVIEW
    approval_token: Optional[str]
    justification: str


class PromotionApprovalRequest(BaseModel):
    """Approval request for promoting candidate shadow model to active production."""
    approval_token: str = Field(..., description="Approval token issued by ShadowModelEvaluator")
    approved_by: str = Field(default="MLOps Engineer", description="Identity or role of approver")
    notes: Optional[str] = Field(default=None, description="Optional sign-off notes")


class PromotionApprovalResponse(BaseModel):
    """Approval confirmation."""
    status: str = "APPROVED"
    approval_token: str
    promoted_model: str = "candidate_v2"
    timestamp: float
    message: str
