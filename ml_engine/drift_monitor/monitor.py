"""
Unified Streaming Drift Monitor Engine.
The primary Python deliverable for Person 1, providing a cohesive, production-grade
interface for continuous streaming ingestion, bounded-memory sketches,
multi-signal drift detection, CBPE label-free performance estimation,
severity alerting, and candidate model evaluation.
"""

from collections import deque
from dataclasses import asdict, dataclass, field
import time
from typing import Any, Deque, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from ml_engine.drift_monitor.config import (
    AlertSeverity,
    DriftConfig,
    DriftType,
    MonitorConfig,
    RetrainUrgency,
    SketchConfig,
)
from ml_engine.drift_monitor.detectors.adwin import ADWIN
from ml_engine.drift_monitor.detectors.c2st import C2STResult, ClassifierTwoSampleTest
from ml_engine.drift_monitor.detectors.corrections import (
    benjamini_hochberg_correction,
    bonferroni_correction,
)
from ml_engine.drift_monitor.detectors.distribution_tests import (
    calculate_psi,
    categorical_chi2_test,
    kolmogorov_smirnov_test,
)
from ml_engine.drift_monitor.detectors.mmd import (
    MaximumMeanDiscrepancy,
    MMDResult,
)
from ml_engine.drift_monitor.detectors.page_hinkley import PageHinkley
from ml_engine.drift_monitor.model import BaselineEvaluationMetrics, BaselineModelWrapper
from ml_engine.drift_monitor.performance_estimation.cbpe import (
    CBPEPerformanceEstimate,
    ConfidenceBasedPerformanceEstimator,
)
from ml_engine.drift_monitor.performance_estimation.delayed_evaluator import (
    DelayedGroundTruthEvaluator,
    PerformanceComparisonReport,
)
from ml_engine.drift_monitor.retraining.alert_engine import (
    AlertStatus,
    DriftAlertEngine,
)
from ml_engine.drift_monitor.retraining.retrain_trigger import (
    RetrainingRecommendation,
    RetrainingTrigger,
)
from ml_engine.drift_monitor.retraining.shadow_evaluator import (
    ShadowEvaluationResult,
    ShadowModelEvaluator,
)
from ml_engine.drift_monitor.sketches.count_min_sketch import CountMinSketch
from ml_engine.drift_monitor.sketches.hyperloglog import HyperLogLog
from ml_engine.drift_monitor.sketches.reservoir import ReservoirSampler
from ml_engine.drift_monitor.sketches.running_stats import RunningStats
from ml_engine.drift_monitor.sketches.tdigest import TDigest


@dataclass
class RecordMonitoringResult:
    """Immediate result returned upon processing a single streaming record."""
    record_id: str
    timestamp: Any
    prediction: int
    probability: float
    streaming_drift_detected: bool
    affected_features: List[str]
    alert_severity: AlertSeverity
    is_periodic_check_step: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "prediction": self.prediction,
            "probability": round(self.probability, 4),
            "streaming_drift_detected": self.streaming_drift_detected,
            "affected_features": self.affected_features,
            "alert_severity": self.alert_severity.value,
            "is_periodic_check_step": self.is_periodic_check_step,
        }


class StreamingDriftMonitor:
    """
    Central monitoring engine orchestrating bounded-memory sketches,
    streaming and windowed drift detectors, CBPE performance estimation,
    alert severity generation, and candidate retraining governance.
    """

    def __init__(
        self,
        config: Optional[MonitorConfig] = None,
        model: Optional[BaselineModelWrapper] = None,
    ):
        self.config = config or MonitorConfig()
        self.model = model

        # Internal state
        self.is_initialized: bool = False
        self.records_processed: int = 0
        self.start_time: float = time.time()

        # Feature schemas
        self.numerical_features: List[str] = list(self.config.numerical_features)
        self.categorical_features: List[str] = list(self.config.categorical_features)
        self.target_column: str = "is_fraud"

        # Reference data store & reference sample
        self.reference_df: Optional[pd.DataFrame] = None
        self.reference_matrix: Optional[np.ndarray] = None
        self.reference_means: Dict[str, float] = {}
        self.reference_stds: Dict[str, float] = {}
        self.reference_cat_counts: Dict[str, Dict[str, int]] = {}
        self.baseline_metrics: Optional[BaselineEvaluationMetrics] = None

        # Sketches per numerical feature
        self.running_stats: Dict[str, RunningStats] = {}
        self.tdigests: Dict[str, TDigest] = {}

        # Sketches per categorical feature
        self.cms_sketches: Dict[str, CountMinSketch] = {}
        self.hll_sketches: Dict[str, HyperLogLog] = {}

        # Streaming detectors per numerical feature
        self.adwin_detectors: Dict[str, ADWIN] = {}
        self.ph_detectors: Dict[str, PageHinkley] = {}

        # Multivariate Reservoir Sampler (bounded sample of stream tuples)
        self.reservoir: ReservoirSampler = ReservoirSampler(
            capacity=self.config.sketches.reservoir_capacity,
            random_state=42,
            name="stream_reservoir",
        )

        # Sliding window buffer of recent raw records for windowed distribution checks
        self.recent_window: Deque[Dict[str, Any]] = deque(maxlen=self.config.window_size)
        self.recent_probabilities: Deque[float] = deque(maxlen=self.config.window_size)

        # Detectors and engines
        self.mmd_detector = MaximumMeanDiscrepancy(
            n_permutations=self.config.drift.mmd_permutations,
            alpha=self.config.drift.mmd_alpha,
        )
        self.c2st_detector = ClassifierTwoSampleTest(
            n_splits=self.config.drift.c2st_cv_folds,
            auc_threshold=self.config.drift.c2st_auc_threshold,
            alpha=self.config.drift.c2st_alpha,
        )
        self.cbpe_estimator = ConfidenceBasedPerformanceEstimator(
            decision_threshold=self.config.cbpe.decision_threshold,
        )
        self.delayed_evaluator = DelayedGroundTruthEvaluator(
            cbpe_estimator=self.cbpe_estimator,
            max_buffer_size=self.config.cbpe.max_delayed_records,
        )
        self.alert_engine = DriftAlertEngine(config=self.config.retrain)
        self.retrain_trigger = RetrainingTrigger(config=self.config.retrain)
        self.shadow_evaluator = ShadowModelEvaluator(config=self.config.retrain)

        # Cached latest periodic monitoring results
        self.latest_univariate_results: Dict[str, Dict[str, Any]] = {}
        self.latest_mmd_result: Optional[MMDResult] = None
        self.latest_c2st_result: Optional[C2STResult] = None
        self.latest_cbpe_estimate: Optional[CBPEPerformanceEstimate] = None
        self.latest_alert_status: AlertStatus = AlertStatus(
            severity=AlertSeverity.NORMAL,
            drift_type=DriftType.NONE,
            composite_score=0.0,
            summary_message="Monitor initialized. Awaiting stream data.",
            drifted_features=[],
            univariate_drift_ratio=0.0,
            multivariate_drift_detected=False,
            performance_degraded=False,
        )
        self.latest_retrain_recommendation: RetrainingRecommendation = RetrainingRecommendation(
            should_retrain=False,
            urgency=RetrainUrgency.NONE,
            severity=AlertSeverity.NORMAL,
            trigger_reason="Awaiting data",
            affected_features=[],
            estimated_impact={},
            recommended_action="Monitor stream",
        )
        self.alert_history: List[Dict[str, Any]] = []

    def setup_reference(
        self,
        reference_df: pd.DataFrame,
        target_column: str = "is_fraud",
        train_model_if_missing: bool = True,
    ) -> BaselineEvaluationMetrics:
        """
        Initializes reference baseline distributions and trains/calibrates model if needed.
        """
        self.reference_df = reference_df.copy()
        self.target_column = target_column

        # Infer feature names if not set
        if not self.numerical_features:
            self.numerical_features = [
                c for c in reference_df.columns
                if pd.api.types.is_numeric_dtype(reference_df[c])
                and c not in [self.config.id_column, self.config.timestamp_column, target_column]
            ]
        if not self.categorical_features:
            self.categorical_features = [
                c for c in reference_df.columns
                if (pd.api.types.is_object_dtype(reference_df[c]) or isinstance(reference_df[c].dtype, pd.CategoricalDtype))
                and c not in [self.config.id_column, self.config.timestamp_column, target_column]
            ]

        # Setup or calibrate model
        if self.model is None or not self.model.is_fitted:
            if train_model_if_missing:
                self.model = BaselineModelWrapper(
                    numerical_features=self.numerical_features,
                    categorical_features=self.categorical_features,
                )
                self.baseline_metrics = self.model.train_and_calibrate(
                    train_df=self.reference_df,
                    target_col=target_column,
                )
        else:
            self.baseline_metrics = self.model.baseline_metrics

        # Initialize sketches & detectors per feature
        for feat in self.numerical_features:
            self.reference_means[feat] = float(reference_df[feat].mean())
            self.reference_stds[feat] = max(1e-4, float(reference_df[feat].std()))
            self.running_stats[feat] = RunningStats(name=feat)
            self.tdigests[feat] = TDigest(compression=self.config.sketches.tdigest_compression)
            self.adwin_detectors[feat] = ADWIN(delta=self.config.drift.adwin_delta, name=f"adwin_{feat}")
            self.ph_detectors[feat] = PageHinkley(
                threshold=self.config.drift.ph_threshold,
                delta=self.config.drift.ph_tolerance,
                alpha=self.config.drift.ph_alpha,
                name=f"ph_{feat}",
            )

        for cat in self.categorical_features:
            self.cms_sketches[cat] = CountMinSketch(
                epsilon=self.config.sketches.cms_epsilon,
                delta=self.config.sketches.cms_delta,
                conservative_update=self.config.sketches.cms_conservative_update,
                name=f"cms_{cat}",
            )
            self.hll_sketches[cat] = HyperLogLog(
                precision_bits=self.config.sketches.hll_precision,
                name=f"hll_{cat}",
            )
            # Store reference category counts
            self.reference_cat_counts[cat] = reference_df[cat].value_counts().to_dict()

        # Cache reference numerical matrix for MMD/C2ST
        self.reference_matrix = reference_df[self.numerical_features].values.astype(np.float64)

        self.is_initialized = True
        return self.baseline_metrics

    def process_record(self, record: Dict[str, Any]) -> RecordMonitoringResult:
        """
        Processes a single incoming streaming record in real time:
        1. Predicts class and calibrated probability.
        2. Updates running statistics and bounded-memory sketches.
        3. Updates streaming ADWIN and Page-Hinkley detectors.
        4. Adds record to multivariate reservoir sampler and recent window buffer.
        5. Runs periodic multivariate, distribution, and CBPE evaluations when interval matches.
        """
        if not self.is_initialized:
            raise RuntimeError("StreamingDriftMonitor must be initialized with setup_reference() before processing records.")

        self.records_processed += 1
        rec_id = str(record.get(self.config.id_column, f"rec_{self.records_processed}"))
        ts = record.get(self.config.timestamp_column, time.time())

        # 1. Model inference
        pred, prob = self.model.predict_record(record)

        # 2. Update delayed evaluator pending queue
        self.delayed_evaluator.record_prediction(
            record_id=rec_id,
            probability=prob,
            prediction=pred,
            timestamp=ts,
        )

        # 3. Update numerical sketches & streaming detectors
        adwin_alarms: List[str] = []
        for feat in self.numerical_features:
            val = float(record.get(feat, 0.0))
            self.running_stats[feat].update(val)
            self.tdigests[feat].update(val)

            # Streaming change detection using standardized z-score
            mean_ref = self.reference_means.get(feat, 0.0)
            std_ref = self.reference_stds.get(feat, 1.0)
            z_val = (val - mean_ref) / std_ref

            adwin_drift, _ = self.adwin_detectors[feat].update(z_val)
            ph_drift, _ = self.ph_detectors[feat].update(z_val)

            if adwin_drift or ph_drift:
                adwin_alarms.append(feat)

        # 4. Update categorical sketches
        for cat in self.categorical_features:
            cval = str(record.get(cat, "unknown"))
            self.cms_sketches[cat].update(cval)
            self.hll_sketches[cat].add(cval)

        # 5. Reservoir & recent buffer update
        self.reservoir.add(record)
        self.recent_window.append(record)
        self.recent_probabilities.append(prob)

        # 6. Periodic batch check for multivariate, distribution, & CBPE analysis
        is_check_step = (self.records_processed % self.config.check_interval == 0) and (len(self.recent_window) >= 50)
        if is_check_step:
            self._run_periodic_evaluation()

        return RecordMonitoringResult(
            record_id=rec_id,
            timestamp=ts,
            prediction=pred,
            probability=prob,
            streaming_drift_detected=(len(adwin_alarms) > 0),
            affected_features=adwin_alarms,
            alert_severity=self.latest_alert_status.severity,
            is_periodic_check_step=is_check_step,
        )

    def process_batch(self, df: pd.DataFrame) -> List[RecordMonitoringResult]:
        """Processes a batch of streaming records."""
        results = []
        records = df.to_dict(orient="records")
        for rec in records:
            res = self.process_record(rec)
            results.append(res)
        return results

    def _run_periodic_evaluation(self) -> None:
        """
        Executes distribution comparisons (KS/PSI with FDR correction),
        multivariate MMD/C2ST tests, CBPE performance estimation,
        alert classification, and retrain trigger checks.
        """
        recent_df = pd.DataFrame(list(self.recent_window))

        # 1. Univariate KS and PSI tests across all numerical features
        raw_p_values: Dict[str, float] = {}
        univariate_details: Dict[str, Dict[str, Any]] = {}

        for feat in self.numerical_features:
            ref_feat = self.reference_df[feat].values
            cur_feat = recent_df[feat].values

            ks_res = kolmogorov_smirnov_test(
                reference_sample=ref_feat,
                current_sample=cur_feat,
                feature_name=feat,
                alpha=self.config.drift.ks_alpha,
            )
            psi_res = calculate_psi(
                reference_sample=ref_feat,
                current_sample=cur_feat,
                feature_name=feat,
                n_bins=self.config.drift.psi_bins,
                psi_warning=self.config.drift.psi_warning,
                psi_critical=self.config.drift.psi_critical,
            )

            raw_p_values[feat] = ks_res.p_value
            univariate_details[feat] = {
                "ks": ks_res.to_dict(),
                "psi": psi_res.to_dict(),
                "drift_detected": ks_res.drift_detected or psi_res.drift_detected,
            }

        # Apply Multiple Testing Correction (Benjamini-Hochberg)
        if self.config.drift.multiple_testing_correction == "bonferroni":
            corrected = bonferroni_correction(raw_p_values, alpha=self.config.drift.ks_alpha)
        else:
            corrected = benjamini_hochberg_correction(raw_p_values, alpha=self.config.drift.ks_alpha)

        for feat, corr_info in corrected.items():
            univariate_details[feat]["correction"] = corr_info
            univariate_details[feat]["is_significant"] = corr_info["is_significant"]

        self.latest_univariate_results = univariate_details

        # 2. Multivariate drift detection (MMD and C2ST using reservoir or window)
        cur_matrix = self.reservoir.get_feature_matrix(self.numerical_features)
        if len(cur_matrix) >= 30 and self.reference_matrix is not None:
            self.latest_mmd_result = self.mmd_detector.test(
                X_ref=self.reference_matrix,
                Y_cur=cur_matrix,
            )
            self.latest_c2st_result = self.c2st_detector.test(
                X_ref=self.reference_matrix,
                Y_cur=cur_matrix,
                feature_names=self.numerical_features,
            )

        # 3. Label-Free Performance Estimation (CBPE)
        if self.recent_probabilities:
            self.latest_cbpe_estimate = self.cbpe_estimator.estimate(list(self.recent_probabilities))

        # 4. Alert status evaluation
        baseline_dict = self.baseline_metrics.to_dict() if self.baseline_metrics else None
        self.latest_alert_status = self.alert_engine.evaluate(
            univariate_results=self.latest_univariate_results,
            multivariate_mmd=self.latest_mmd_result,
            multivariate_c2st=self.latest_c2st_result,
            cbpe_estimate=self.latest_cbpe_estimate,
            baseline_metrics=baseline_dict,
        )

        # 5. Retraining recommendation
        self.latest_retrain_recommendation = self.retrain_trigger.evaluate(
            alert_status=self.latest_alert_status,
            recent_records_count=len(self.recent_window),
        )

        # Record alert if not normal
        if self.latest_alert_status.severity != AlertSeverity.NORMAL:
            self.alert_history.append({
                "record_position": self.records_processed,
                "timestamp": time.time(),
                "severity": self.latest_alert_status.severity.value,
                "summary": self.latest_alert_status.summary_message,
                "affected_features": self.latest_alert_status.drifted_features,
                "recommendation": self.latest_retrain_recommendation.to_dict(),
            })

    def receive_delayed_labels(self, ground_truth_map: Dict[str, int]) -> Dict[str, Any]:
        """
        Receives delayed labels from investigative ground truth, computes actual
        metrics, and calculates estimation error against CBPE.
        """
        n_matched = self.delayed_evaluator.receive_labels_batch(ground_truth_map)
        report = self.delayed_evaluator.evaluate_recent_window(window_size=self.config.window_size)
        agg_err = self.delayed_evaluator.get_aggregate_estimation_errors()

        return {
            "matched_labels": n_matched,
            "latest_comparison": report.to_dict() if report else None,
            "aggregate_errors": agg_err,
        }

    def trigger_shadow_retraining(
        self,
        recent_data_with_labels: pd.DataFrame,
    ) -> Tuple[ShadowEvaluationResult, Optional[BaselineModelWrapper]]:
        """
        Trains and validates candidate model against active production model.
        Returns evaluation report with decision and approval token.
        """
        return self.shadow_evaluator.evaluate_candidate(
            active_model=self.model,
            recent_data_with_labels=recent_data_with_labels,
            target_column=self.target_column,
        )

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        Comprehensive JSON-serializable monitoring payload designed for Person 2's
        FastAPI endpoints (/drift/status, /metrics, dashboard cards).
        """
        uptime_seconds = time.time() - self.start_time
        throughput = self.records_processed / max(0.001, uptime_seconds)

        # Sketch memory summaries
        sketches_memory = sum(s.memory_bytes() for s in self.tdigests.values())
        sketches_memory += sum(c.memory_bytes() for c in self.cms_sketches.values())
        sketches_memory += sum(h.memory_bytes() for h in self.hll_sketches.values())
        sketches_memory += self.reservoir.memory_bytes()

        # Raw window memory comparison
        raw_window_bytes = (
            int(pd.DataFrame(list(self.recent_window)).memory_usage(deep=True).sum())
            if self.recent_window
            else 0
        )

        return {
            "status": "HEALTHY",
            "records_processed": self.records_processed,
            "throughput_rec_per_sec": round(throughput, 2),
            "uptime_seconds": round(uptime_seconds, 1),
            "memory": {
                "sketches_total_bytes": sketches_memory,
                "raw_window_bytes": raw_window_bytes,
                "memory_compression_ratio": round(raw_window_bytes / max(1, sketches_memory), 2)
                if raw_window_bytes > 0
                else 1.0,
            },
            "alert": self.latest_alert_status.to_dict(),
            "retraining_recommendation": self.latest_retrain_recommendation.to_dict(),
            "performance_estimation": (
                self.latest_cbpe_estimate.to_dict() if self.latest_cbpe_estimate else None
            ),
            "baseline_metrics": (
                self.baseline_metrics.to_dict() if self.baseline_metrics else None
            ),
            "multivariate_drift": {
                "mmd": self.latest_mmd_result.to_dict() if self.latest_mmd_result else None,
                "c2st": self.latest_c2st_result.to_dict() if self.latest_c2st_result else None,
            },
            "univariate_drift": {
                feat: {
                    "psi": self.latest_univariate_results[feat]["psi"]["psi_value"],
                    "ks_p_value": self.latest_univariate_results[feat]["ks"]["p_value"],
                    "is_significant": self.latest_univariate_results[feat].get("is_significant", False),
                    "running_mean": self.running_stats[feat].mean,
                    "tdigest_median": self.tdigests[feat].quantile(0.50),
                    "tdigest_p95": self.tdigests[feat].quantile(0.95),
                }
                for feat in self.numerical_features
                if feat in self.latest_univariate_results
            },
            "categorical_sketches": {
                cat: {
                    "estimated_cardinality": self.hll_sketches[cat].estimate_cardinality(),
                    "total_counts": self.cms_sketches[cat].total_count,
                }
                for cat in self.categorical_features
            },
            "alert_history_count": len(self.alert_history),
        }
