"""
Retraining Governance & Shadow Evaluation Router for Person 2.
Enforces safe MLOps deployment practices:
- Shadow-model evaluation on held-out streaming validation data
- Approval token generation when candidate outperforms active model
- Human-in-the-loop promotion verification with tamper-evident audit logging
"""

from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.schemas.alerts import (
    CandidateEvaluationRequest,
    CandidateEvaluationResponse,
    PromotionApprovalRequest,
    PromotionApprovalResponse,
    RetrainRecommendationResponse,
)
from backend.app.services.drift_service import DriftService

router = APIRouter(prefix="/api/v1/retrain", tags=["Retraining Governance"])


def get_drift_service() -> DriftService:
    return DriftService.get_instance()


@router.get("/status", summary="Get Retraining Status & Audit History")
def get_retrain_status(
    limit: int = Query(default=20, ge=1, le=100),
    service: DriftService = Depends(get_drift_service),
) -> Dict[str, Any]:
    """
    Returns current retraining trigger recommendation and past
    shadow evaluation audits.
    """
    rec = service.monitor.latest_retrain_recommendation.to_dict()
    audits = service.db.get_retrain_audits(limit=limit)
    return {
        "recommendation": rec,
        "recent_audits": audits,
    }


@router.post(
    "/evaluate-candidate",
    response_model=CandidateEvaluationResponse,
    summary="Evaluate Candidate Shadow Model",
)
def evaluate_candidate(
    request: CandidateEvaluationRequest,
    service: DriftService = Depends(get_drift_service),
) -> CandidateEvaluationResponse:
    """
    Trains a candidate shadow model on recent streaming data and benchmarks it
    against the active model on hold-out validation.
    Issues an approval token ONLY if the candidate achieves superior performance.
    """
    try:
        return service.evaluate_candidate_model(sample_size=request.sample_size)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to evaluate candidate shadow model: {str(e)}"
        )


@router.post(
    "/approve",
    response_model=PromotionApprovalResponse,
    summary="Approve and Promote Candidate Model",
)
def approve_candidate_promotion(
    request: PromotionApprovalRequest,
    service: DriftService = Depends(get_drift_service),
) -> PromotionApprovalResponse:
    """
    Promotes an evaluated candidate model to active production upon verification
    of an authorized approval token. Enforces governance audit trail.
    """
    try:
        return service.approve_promotion(
            approval_token=request.approval_token,
            approved_by=request.approved_by,
            notes=request.notes,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to approve model: {str(e)}")
