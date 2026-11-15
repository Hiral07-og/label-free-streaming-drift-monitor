"""
Pydantic schemas for streaming transaction ingestion and validation.
"""

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TransactionEvent(BaseModel):
    """Incoming transaction schema with strong type coercion and validation."""
    record_id: Optional[str] = Field(default=None, description="Unique transaction ID")
    timestamp: Optional[float] = Field(default_factory=time.time, description="Unix timestamp")
    
    # Numerical features
    tx_amount: float = Field(..., description="Transaction dollar amount")
    sender_tx_count_1h: float = Field(default=1.0, description="Transactions by sender in past 1 hour")
    sender_risk_score: float = Field(default=0.1, description="Internal sender risk score [0, 1]")
    location_entropy: float = Field(default=0.2, description="Location novelty entropy [0, 1]")
    dest_novelty_score: float = Field(default=0.5, description="Destination novelty score")
    time_delta_last_tx: float = Field(default=300.0, description="Seconds since sender's last transaction")
    network_degree: float = Field(default=1.0, description="Entity graph degree")

    # Categorical features
    device_type: str = Field(default="web_browser", description="Client device type")
    merchant_category: str = Field(default="groceries", description="Merchant category")

    # Optional delayed ground truth (for testing)
    is_fraud: Optional[int] = Field(default=None, description="Ground truth label if available")


class RecordIngestResponse(BaseModel):
    """Response returned upon processing a streaming record."""
    status: str = "SUCCESS"
    record_id: str
    timestamp: float
    prediction: int
    probability: float
    streaming_drift_detected: bool
    affected_features: List[str]
    alert_severity: str
    is_periodic_check_step: bool


class BatchIngestRequest(BaseModel):
    """Batch ingestion request."""
    records: List[TransactionEvent]


class BatchIngestResponse(BaseModel):
    """Batch ingestion response summary."""
    status: str = "SUCCESS"
    processed_count: int
    streaming_alarms_count: int
    latest_severity: str
    throughput_rec_per_sec: float
