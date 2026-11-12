"""
Configuration schemas and data structures for the Streaming Drift Monitor.
Provides strongly typed dataclasses for sketches, detectors, CBPE, alerting,
and retraining triggers.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class AlertSeverity(str, Enum):
    """Alert severity levels for drift and performance degradation."""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class DriftType(str, Enum):
    """Classification of detected drift."""
    NONE = "NONE"
    UNIVARIATE = "UNIVARIATE"
    MULTIVARIATE = "MULTIVARIATE"
    PERFORMANCE = "PERFORMANCE"
    COMBINED = "COMBINED"


class RetrainUrgency(str, Enum):
    """Urgency level for automated model retraining."""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class SketchConfig:
    """Configuration for bounded-memory sketch algorithms."""
    # t-digest compression parameter delta (higher = more centroids, better accuracy, more memory)
    tdigest_compression: float = 100.0
    # Count-Min Sketch error epsilon (w = ceil(e / epsilon))
    cms_epsilon: float = 0.005
    # Count-Min Sketch failure probability delta (d = ceil(ln(1 / delta)))
    cms_delta: float = 0.01
    # Count-Min Sketch conservative update flag
    cms_conservative_update: bool = True
    # HyperLogLog precision bits b (m = 2^b registers, e.g. 10 -> 1024 registers)
    hll_precision: int = 10
    # Fixed-capacity Reservoir sampling size K for multivariate tests
    reservoir_capacity: int = 1000


@dataclass
class DriftConfig:
    """Configuration for univariate and multivariate drift detectors."""
    # ADWIN confidence parameter delta
    adwin_delta: float = 0.002
    # Page-Hinkley threshold lambda
    ph_threshold: float = 50.0
    # Page-Hinkley tolerance delta
    ph_tolerance: float = 0.01
    # Page-Hinkley alpha forgetting factor (1.0 = no decay)
    ph_alpha: float = 0.999
    # Kolmogorov-Smirnov significance level alpha
    ks_alpha: float = 0.05
    # Population Stability Index thresholds
    psi_warning: float = 0.10
    psi_critical: float = 0.25
    psi_bins: int = 10
    # Multiple testing correction method: "benjamini-hochberg" or "bonferroni"
    multiple_testing_correction: str = "benjamini-hochberg"
    # MMD kernel parameters
    mmd_kernel: str = "rbf"
    mmd_gamma: Optional[float] = None  # None = median heuristic
    mmd_permutations: int = 100
    mmd_alpha: float = 0.05
    # Classifier Two-Sample Test (C2ST) parameters
    c2st_cv_folds: int = 3
    c2st_auc_threshold: float = 0.65
    c2st_alpha: float = 0.05


@dataclass
class CBPEConfig:
    """Configuration for Confidence-Based Performance Estimation."""
    # Binary classification decision threshold
    decision_threshold: float = 0.5
    # Probability clipping bounds to prevent extreme log-loss or inf
    clip_min: float = 1e-6
    clip_max: float = 1.0 - 1e-6
    # Maximum allowed delay in records for ground truth arrival
    max_delayed_records: int = 50000


@dataclass
class RetrainConfig:
    """Configuration for alerting and automated retraining triggers."""
    # Ratio of drifted numerical features to trigger Warning (e.g. 15%)
    warning_drift_ratio: float = 0.15
    # Ratio of drifted numerical features to trigger Critical (e.g. 35%)
    critical_drift_ratio: float = 0.35
    # Estimated accuracy drop threshold relative to baseline
    max_estimated_accuracy_drop: float = 0.06
    # Estimated F1 drop threshold relative to baseline
    max_estimated_f1_drop: float = 0.08
    # Estimated ROC-AUC drop threshold relative to baseline
    max_estimated_auc_drop: float = 0.07
    # Minimum window size before raising critical retraining recommendations
    min_window_for_retrain: int = 200
    # Validation split fraction for candidate model evaluation
    shadow_val_split: float = 0.25
    # Minimum performance gain required to recommend candidate model promotion
    candidate_min_f1_gain: float = 0.02


@dataclass
class MonitorConfig:
    """Master configuration for the Streaming Drift Monitor engine."""
    sketches: SketchConfig = field(default_factory=SketchConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    cbpe: CBPEConfig = field(default_factory=CBPEConfig)
    retrain: RetrainConfig = field(default_factory=RetrainConfig)
    # Sliding window size of recent records for windowed distribution checks
    window_size: int = 500
    # Sliding window step size for periodic checks
    check_interval: int = 100
    # Monitored numerical and categorical feature names
    numerical_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    id_column: str = "record_id"
    timestamp_column: str = "timestamp"
    prediction_column: str = "prediction"
    probability_column: str = "probability"
