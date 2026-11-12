# Person 2 Backend & Streaming Integration Guide

This guide specifies the exact interface contracts, code patterns, and JSON payloads for **Person 2 (Backend / Systems Lead)** to connect Kafka streaming and FastAPI endpoints directly to the **Streaming Drift Monitor (Person 1 ML Engine)**.

---

## 1. Quick Import & Initialization

```python
from ml_engine.drift_monitor import (
    StreamingDriftMonitor,
    MonitorConfig,
    StreamingFinancialBenchmark,
    BaselineModelWrapper,
    AlertSeverity,
)

# 1. Initialize configuration (or customize via MonitorConfig)
config = MonitorConfig(window_size=300, check_interval=50)

# 2. Instantiate master monitor
monitor = StreamingDriftMonitor(config=config)

# 3. Initialize reference distribution and baseline model
benchmark = StreamingFinancialBenchmark(random_state=42)
split = benchmark.generate(n_reference=3000, n_stream=4000)

monitor.setup_reference(
    reference_df=split.reference_df,
    target_column="is_fraud",
    train_model_if_missing=True,
)
```

---

## 2. Kafka Consumer Integration Pattern

Person 2 can ingest records directly from the Kafka consumer loop into `monitor.process_record`:

```python
from confluent_kafka import Consumer
import json

consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'drift-monitor-group',
    'auto.offset.reset': 'earliest'
})
consumer.subscribe(['transaction-events'])

while True:
    msg = consumer.poll(1.0)
    if msg is None:
        continue
    if msg.error():
        continue

    record_dict = json.loads(msg.value().decode('utf-8'))
    
    # Process record through ML Monitoring Engine
    result = monitor.process_record(record_dict)
    
    # 1. Immediate streaming alert check (ADWIN / Page-Hinkley)
    if result.streaming_drift_detected:
        logger.warning(
            f"Streaming drift on record {result.record_id}: {result.affected_features}"
        )
    
    # 2. Periodic consolidated alert check (Severity: NORMAL, WARNING, CRITICAL)
    if result.is_periodic_check_step:
        alert_status = monitor.latest_alert_status
        if alert_status.severity == AlertSeverity.CRITICAL:
            trigger_slack_pagerduty_alert(alert_status)
```

---

## 3. FastAPI Endpoint Implementations

Person 2 can plug `StreamingDriftMonitor` into FastAPI routers as follows:

### A. Health & Status Summary: `GET /api/v1/drift/status`
```python
@router.get("/api/v1/drift/status")
async def get_drift_status():
    """Returns real-time drift, sketch memory, and model health summary."""
    return monitor.get_monitoring_summary()
```

**Sample JSON Output:**
```json
{
  "status": "HEALTHY",
  "records_processed": 1450,
  "throughput_rec_per_sec": 48.2,
  "uptime_seconds": 30.1,
  "memory": {
    "sketches_total_bytes": 118416,
    "raw_window_bytes": 50431,
    "memory_compression_ratio": 2.35
  },
  "alert": {
    "severity": "CRITICAL",
    "drift_type": "COMBINED",
    "composite_score": 0.75,
    "summary_message": "CRITICAL alert: Significant drift across 2 features with confirmed performance degradation.",
    "drifted_features": ["tx_amount", "sender_risk_score"]
  },
  "retraining_recommendation": {
    "should_retrain": true,
    "urgency": "HIGH",
    "recommended_action": "Initiate shadow model training immediately on recent streaming data."
  },
  "performance_estimation": {
    "estimated_accuracy": 0.7873,
    "estimated_f1": 0.4021,
    "estimated_roc_auc": 0.7760,
    "estimated_brier_score": 0.1506
  }
}
```

### B. Ingest Single Transaction: `POST /api/v1/stream/record`
```python
@router.post("/api/v1/stream/record")
async def ingest_record(payload: dict):
    """Processes single incoming record from external source."""
    result = monitor.process_record(payload)
    return result.to_dict()
```

### C. Ingest Delayed Ground Truth: `POST /api/v1/feedback/labels`
```python
@router.post("/api/v1/feedback/labels")
async def receive_delayed_labels(ground_truth_map: dict[str, int]):
    """Receives delayed ground truth labels, calculates actual performance vs CBPE."""
    verification = monitor.receive_delayed_labels(ground_truth_map)
    return verification
```

### D. Shadow Retraining Evaluation: `POST /api/v1/retrain/evaluate-candidate`
```python
@router.post("/api/v1/retrain/evaluate-candidate")
async def evaluate_shadow_model(recent_batch_df: list[dict]):
    """
    Evaluates candidate model against active production model.
    Enforces governance rule: No automatic production promotion without approval token!
    """
    df = pd.DataFrame(recent_batch_df)
    shadow_result, candidate_model = monitor.trigger_shadow_retraining(df)
    return shadow_result.to_dict()
```

---

## 4. Prometheus Metrics Exporter Mapping

Person 2 can map `monitor.get_monitoring_summary()` directly to Prometheus gauges:

| Metric Name | Type | Value from Monitor |
| :--- | :---: | :--- |
| `drift_records_processed_total` | Counter | `summary['records_processed']` |
| `drift_composite_score` | Gauge | `summary['alert']['composite_score']` |
| `drift_severity_level` | Gauge | `0` (Normal), `1` (Warning), `2` (Critical) |
| `model_cbpe_estimated_accuracy` | Gauge | `summary['performance_estimation']['estimated_accuracy']` |
| `model_cbpe_estimated_f1` | Gauge | `summary['performance_estimation']['estimated_f1']` |
| `model_cbpe_estimated_auc` | Gauge | `summary['performance_estimation']['estimated_roc_auc']` |
| `sketches_memory_bytes` | Gauge | `summary['memory']['sketches_total_bytes']` |
| `multivariate_mmd_p_value` | Gauge | `summary['multivariate_drift']['mmd']['p_value']` |
| `multivariate_c2st_auc` | Gauge | `summary['multivariate_drift']['c2st']['roc_auc']` |

---

## 5. Verification Commands

Person 2 can verify the entire ML engine locally:
```bash
# Run unit test suite
.venv\Scripts\python.exe -m unittest discover ml_engine/tests/

# Run complete turn-by-turn demo across all 8 weeks
.venv\Scripts\python.exe run_person1_demo.py
```
