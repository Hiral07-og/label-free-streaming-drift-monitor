"""
Stream Ingestion Router for Person 2 (Backend & Systems Lead).
Handles synchronous record ingestion, batch ingestion, streaming simulation,
and streaming consumer health diagnostics.
"""

import os
from typing import Any, Dict
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
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
        description="Scenario to inject: 'none', 'abrupt', 'gradual', 'recurring', or 'feature_specific'",
    )


class AutoStreamRequest(BaseModel):
    drift_mode: str = Field(
        default="none",
        description="Continuous stream mode: 'none', 'abrupt', or 'gradual'",
    )
    rate_per_sec: float = Field(
        default=5.0, ge=0.5, le=100.0, description="Target emission rate in records per second"
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


@router.get("/consumer-status", summary="Get Streaming Pipeline & Consumer Status")
def get_consumer_status(
    service: DriftService = Depends(get_drift_service),
) -> Dict[str, Any]:
    """
    Returns live runtime metrics for the connected Producer -> Consumer streaming pipeline.
    """
    return service.get_pipeline_status()


@router.post("/auto-stream/start", summary="Start Continuous Background Auto-Streaming")
def start_auto_stream(
    request: AutoStreamRequest,
    service: DriftService = Depends(get_drift_service),
) -> Dict[str, Any]:
    """
    Starts continuous real-time event emission through StreamingEventProducer ->
    StreamingEventConsumer -> StreamingDriftMonitor.
    """
    return service.start_auto_stream(
        drift_mode=request.drift_mode,
        rate_per_sec=request.rate_per_sec,
    )


@router.post("/auto-stream/stop", summary="Stop Continuous Background Auto-Streaming")
def stop_auto_stream(
    service: DriftService = Depends(get_drift_service),
) -> Dict[str, Any]:
    """
    Cleanly stops background event emission.
    """
    return service.stop_auto_stream()


@router.post("/upload-csv", summary="Upload CSV Dataset for Drift & Performance Analysis")
async def upload_csv_dataset(
    file: UploadFile = File(..., description="CSV file of streaming transactions to analyze"),
    service: DriftService = Depends(get_drift_service),
) -> Dict[str, Any]:
    """
    Uploads a CSV file containing transactions.
    Streams each row through bounded sketches (Welford, t-digest, Count-Min, HyperLogLog, Reservoir),
    evaluates univariate & multivariate drift, estimates model accuracy via CBPE,
    and returns a full diagnostic report card.
    """
    try:
        content = await file.read()
        return service.process_uploaded_csv(content, filename=file.filename or "upload.csv")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/sample-csv/{sample_type}", summary="Download Sample CSV Dataset")
def get_sample_csv(sample_type: str = "normal"):
    """
    Returns sample test CSV data for immediate demonstration:
    - 'normal': Stationary in-distribution transactions
    - 'drifted': Abruptly drifted out-of-distribution transactions
    """
    file_map = {
        "normal": os.path.join("data", "sample_normal_transactions.csv"),
        "drifted": os.path.join("data", "sample_drifted_transactions.csv"),
    }
    path = file_map.get(sample_type)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Sample dataset not found")
    return FileResponse(path, media_type="text/csv", filename=f"sample_{sample_type}_transactions.csv")

