"""
Stream Ingestion Router for Person 2 (Backend & Systems Lead).
Handles synchronous record ingestion, batch ingestion, streaming simulation,
and streaming consumer health diagnostics.
"""

from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend.app.schemas.events import (
    BatchIngestRequest,
    BatchIngestResponse,
    RecordIngestResponse,
    TransactionEvent,
)
from backend.app.services.drift_service import DriftService

router = APIRouter(prefix="/api/v1/stream", tags=["Streaming Ingestion"])


class SimulationRequest(BaseModel):
    num_records: int = Field(default=150, ge=1, le=2000, description="Number of synthetic records to stream")
    drift_type: str = Field(
        default="none",
        description="Scenario to inject: 'none' (stationary), 'abrupt' (fraud wave), or 'gradual' (inflation shift)",
    )


def get_drift_service() -> DriftService:
    return DriftService.get_instance()


@router.post("/record", response_model=RecordIngestResponse, summary="Ingest Single Transaction")
def ingest_record(
    event: TransactionEvent,
    service: DriftService = Depends(get_drift_service),
) -> RecordIngestResponse:
    """
    Ingests and processes a single transaction event through the streaming monitor.
    Updates running sketches (Welford, t-digest, Count-Min, HyperLogLog, Reservoir)
    and checks streaming ADWIN/Page-Hinkley detectors.
    """
    try:
        return service.process_event(event)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process record: {str(e)}")


@router.post("/batch", response_model=BatchIngestResponse, summary="Ingest Batch of Transactions")
def ingest_batch(
    request: BatchIngestRequest,
    service: DriftService = Depends(get_drift_service),
) -> BatchIngestResponse:
    """
    Ingests an ordered batch of streaming transactions.
    Computes throughput and returns detected alarms count and latest alert severity.
    """
    try:
        return service.process_batch(request.records)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process batch: {str(e)}")


@router.post("/simulate", summary="Simulate Streaming Scenario")
def simulate_scenario(
    request: SimulationRequest,
    service: DriftService = Depends(get_drift_service),
) -> Dict[str, Any]:
    """
    Simulates a streaming workload with configurable drift injection:
    - **none**: Stationary standard baseline traffic
    - **abrupt**: Sudden fraud wave with elevated amounts and risk scores
    - **gradual**: Stepwise inflation shift affecting quantiles and feature entropy
    """
    try:
        return service.simulate_stream(
            num_records=request.num_records,
            drift_type=request.drift_type,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")
