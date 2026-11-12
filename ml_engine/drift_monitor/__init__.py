"""
Streaming Drift Monitor Package:
Label-Free Streaming Drift Monitor with Bounded-Memory Sketches and Automated Retrain Triggers.
Person 1 (ML / Research Lead) Main Deliverable.
"""

from ml_engine.drift_monitor.config import (
    AlertSeverity,
    CBPEConfig,
    DriftConfig,
    DriftType,
    MonitorConfig,
    RetrainConfig,
    RetrainUrgency,
    SketchConfig,
)
from ml_engine.drift_monitor.dataset import (
    StreamingDatasetSplit,
    StreamingFinancialBenchmark,
    load_elliptic_dataset,
)
from ml_engine.drift_monitor.detectors import (
    ADWIN,
    C2STResult,
    ClassifierTwoSampleTest,
    KSTestResult,
    MaximumMeanDiscrepancy,
    MMDResult,
    PageHinkley,
    PSIResult,
    benjamini_hochberg_correction,
    bonferroni_correction,
    calculate_psi,
    categorical_chi2_test,
    kolmogorov_smirnov_test,
)
from ml_engine.drift_monitor.model import (
    BaselineEvaluationMetrics,
    BaselineModelWrapper,
)
from ml_engine.drift_monitor.monitor import (
    RecordMonitoringResult,
    StreamingDriftMonitor,
)
from ml_engine.drift_monitor.performance_estimation import (
    CBPEPerformanceEstimate,
    ConfidenceBasedPerformanceEstimator,
    DelayedGroundTruthEvaluator,
    PerformanceComparisonReport,
)
from ml_engine.drift_monitor.retraining import (
    AlertStatus,
    DriftAlertEngine,
    RetrainingRecommendation,
    RetrainingTrigger,
    ShadowEvaluationResult,
    ShadowModelEvaluator,
)
from ml_engine.drift_monitor.sketches import (
    Centroid,
    CountMinSketch,
    HyperLogLog,
    ReservoirSampler,
    RunningStats,
    TDigest,
)

__version__ = "1.0.0"

__all__ = [
    # Master Orchestrator
    "StreamingDriftMonitor",
    "RecordMonitoringResult",
    # Configurations
    "MonitorConfig",
    "SketchConfig",
    "DriftConfig",
    "CBPEConfig",
    "RetrainConfig",
    "AlertSeverity",
    "DriftType",
    "RetrainUrgency",
    # Dataset & Model
    "StreamingFinancialBenchmark",
    "StreamingDatasetSplit",
    "load_elliptic_dataset",
    "BaselineModelWrapper",
    "BaselineEvaluationMetrics",
    # Sketches
    "RunningStats",
    "TDigest",
    "Centroid",
    "CountMinSketch",
    "HyperLogLog",
    "ReservoirSampler",
    # Detectors
    "ADWIN",
    "PageHinkley",
    "kolmogorov_smirnov_test",
    "calculate_psi",
    "categorical_chi2_test",
    "KSTestResult",
    "PSIResult",
    "MaximumMeanDiscrepancy",
    "MMDResult",
    "ClassifierTwoSampleTest",
    "C2STResult",
    "bonferroni_correction",
    "benjamini_hochberg_correction",
    # Performance Estimation
    "ConfidenceBasedPerformanceEstimator",
    "CBPEPerformanceEstimate",
    "DelayedGroundTruthEvaluator",
    "PerformanceComparisonReport",
    # Retraining & Alerts
    "DriftAlertEngine",
    "AlertStatus",
    "RetrainingTrigger",
    "RetrainingRecommendation",
    "ShadowModelEvaluator",
    "ShadowEvaluationResult",
]
