"""
FastAPI Main Application for Person 2 (Backend & Systems Lead).
Integrates streaming event ingestion, drift monitoring routers,
Prometheus text-format metrics exporter, and interactive web dashboard.
"""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
import time
from typing import AsyncGenerator

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.app.config import settings
from backend.app.routers import (
    alerts_router,
    drift_router,
    performance_router,
    retrain_router,
    stream_router,
)
from backend.app.services.drift_service import DriftService
from backend.app.streaming.consumer import StreamingEventConsumer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("drift_backend")

consumer: StreamingEventConsumer = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle manager."""
    global consumer
    logger.info("Starting up %s v%s...", settings.app_name, settings.app_version)

    # Pre-initialize DriftService singleton
    service = DriftService.get_instance()
    logger.info(
        "DriftService warmed up (baseline accuracy: %.3f)",
        service.monitor.baseline_metrics.accuracy if service.monitor.baseline_metrics else 0.0,
    )

    # Initialize consumer worker
    consumer = StreamingEventConsumer(drift_service=service)
    consumer.start()

    yield

    logger.info("Shutting down %s...", settings.app_name)
    if consumer:
        consumer.stop()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-grade Label-Free Streaming Drift Monitor with Bounded-Memory Sketches, CBPE, and Retraining Triggers.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(stream_router)
app.include_router(drift_router)
app.include_router(performance_router)
app.include_router(alerts_router)
app.include_router(retrain_router)


@app.get("/health", tags=["Health"])
def health_check():
    """Service health probe."""
    return {
        "status": "UP",
        "service": settings.app_name,
        "version": settings.app_version,
        "timestamp": time.time(),
    }


@app.get("/metrics", tags=["Prometheus Metrics"], response_class=Response)
def prometheus_metrics():
    """
    Standard Prometheus text-format exporter (/metrics) exposing:
    - Ingestion counters and throughput gauges
    - Bounded-memory sketch bytes vs uncompressed window bytes
    - Active alert severity and composite score
    - Feature-level PSI and KS p-values
    - CBPE estimated accuracy, F1, and ROC-AUC
    """
    service = DriftService.get_instance()
    summary = service.monitor.get_monitoring_summary()
    mem = summary["memory"]
    alert = summary["alert"]
    cbpe = summary.get("performance_estimation") or {}
    univariate = summary.get("univariate_drift") or {}

    sev_map = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}
    sev_val = sev_map.get(alert.get("severity", "NORMAL"), 0)

    lines = [
        "# HELP drift_records_processed_total Total records processed by the streaming monitor",
        "# TYPE drift_records_processed_total counter",
        f"drift_records_processed_total {summary['records_processed']}",
        "",
        "# HELP drift_throughput_rec_per_sec Current streaming throughput in records per second",
        "# TYPE drift_throughput_rec_per_sec gauge",
        f"drift_throughput_rec_per_sec {summary['throughput_rec_per_sec']}",
        "",
        "# HELP drift_sketches_memory_bytes Total memory in bytes consumed by bounded-memory sketches",
        "# TYPE drift_sketches_memory_bytes gauge",
        f"drift_sketches_memory_bytes {mem['sketches_total_bytes']}",
        "",
        "# HELP drift_raw_window_memory_bytes Memory in bytes required to store uncompressed sliding window",
        "# TYPE drift_raw_window_memory_bytes gauge",
        f"drift_raw_window_memory_bytes {mem['raw_window_bytes']}",
        "",
        "# HELP drift_memory_compression_ratio Memory compression ratio achieved by sketches",
        "# TYPE drift_memory_compression_ratio gauge",
        f"drift_memory_compression_ratio {mem['memory_compression_ratio']}",
        "",
        "# HELP drift_alert_severity Active alert severity: 0=NORMAL, 1=WARNING, 2=CRITICAL",
        "# TYPE drift_alert_severity gauge",
        f"drift_alert_severity {sev_val}",
        "",
        "# HELP drift_composite_score Weighted composite drift score [0, 1]",
        "# TYPE drift_composite_score gauge",
        f"drift_composite_score {alert.get('composite_score', 0.0)}",
        "",
        "# HELP drift_estimated_f1 CBPE label-free estimated F1 score",
        "# TYPE drift_estimated_f1 gauge",
        f"drift_estimated_f1 {cbpe.get('estimated_f1', 0.0)}",
        "",
        "# HELP drift_estimated_accuracy CBPE label-free estimated accuracy",
        "# TYPE drift_estimated_accuracy gauge",
        f"drift_estimated_accuracy {cbpe.get('estimated_accuracy', 0.0)}",
        "",
        "# HELP drift_estimated_roc_auc CBPE label-free estimated ROC-AUC",
        "# TYPE drift_estimated_roc_auc gauge",
        f"drift_estimated_roc_auc {cbpe.get('estimated_roc_auc', 0.0)}",
        "",
    ]

    for feat_name, feat_data in univariate.items():
        psi = feat_data.get("psi", 0.0)
        ks_p = feat_data.get("ks_p_value", 1.0)
        sig = 1 if feat_data.get("is_significant", False) else 0
        lines.append(f'drift_feature_psi{{feature="{feat_name}"}} {psi}')
        lines.append(f'drift_feature_ks_pvalue{{feature="{feat_name}"}} {ks_p}')
        lines.append(f'drift_feature_significant{{feature="{feat_name}"}} {sig}')

    content = "\n".join(lines) + "\n"
    return Response(content=content, media_type="text/plain; version=0.0.4; charset=utf-8")


# Static files & Dashboard mount
static_dir = Path(__file__).parent / "dashboard" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", include_in_schema=False)
@app.get("/dashboard", include_in_schema=False)
def serve_dashboard():
    """Serves the interactive single-page dashboard HTML."""
    html_file = static_dir / "index.html"
    if html_file.exists():
        return FileResponse(str(html_file))
    return HTMLResponse("<h2>Streaming Drift Monitor Dashboard static files not found.</h2>")
