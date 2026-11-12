"""
Benchmarks and experiment suite package:
- SyntheticDriftInjector & DriftExperimentDataset
- DriftBenchmarkSuite
- generate_benchmark_reports
"""

from ml_engine.drift_monitor.benchmarks.drift_generator import (
    DriftExperimentDataset,
    SyntheticDriftInjector,
)
from ml_engine.drift_monitor.benchmarks.generate_report import (
    generate_benchmark_reports,
)
from ml_engine.drift_monitor.benchmarks.run_benchmarks import (
    DetectionBenchmarkResult,
    DriftBenchmarkSuite,
)

__all__ = [
    "SyntheticDriftInjector",
    "DriftExperimentDataset",
    "DriftBenchmarkSuite",
    "DetectionBenchmarkResult",
    "generate_benchmark_reports",
]
