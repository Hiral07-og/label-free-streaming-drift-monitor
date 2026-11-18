"""
Unified API Routers for Label-Free Streaming Drift Monitor.
"""

from backend.app.routers.alerts import router as alerts_router
from backend.app.routers.benchmarks import router as benchmarks_router
from backend.app.routers.drift import router as drift_router
from backend.app.routers.performance import router as performance_router
from backend.app.routers.retrain import router as retrain_router
from backend.app.routers.stream import router as stream_router

__all__ = [
    "stream_router",
    "drift_router",
    "performance_router",
    "alerts_router",
    "retrain_router",
    "benchmarks_router",
]
