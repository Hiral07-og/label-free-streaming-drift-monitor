"""
Comprehensive Experimental Benchmark Suite for the Streaming Drift Monitor.
Executes rigorous evaluation across:
1. Detection Delay & Detection Rate (Abrupt, Gradual, Recurring, Feature-Specific)
2. False Alarm Rate (FAR) on stationary baseline stream
3. Sketch Approximation Accuracy & Bounded Memory Efficiency (t-digest, CMS, HLL)
4. Confidence-Based Performance Estimation (CBPE) error vs Delayed Ground Truth
5. Univariate vs Multivariate Sensitivity Comparison
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ml_engine.drift_monitor.benchmarks.drift_generator import (
    DriftExperimentDataset,
    SyntheticDriftInjector,
)
from ml_engine.drift_monitor.config import AlertSeverity, MonitorConfig
from ml_engine.drift_monitor.dataset import StreamingFinancialBenchmark
from ml_engine.drift_monitor.model import BaselineModelWrapper
from ml_engine.drift_monitor.monitor import StreamingDriftMonitor
from ml_engine.drift_monitor.sketches.count_min_sketch import CountMinSketch
from ml_engine.drift_monitor.sketches.hyperloglog import HyperLogLog
from ml_engine.drift_monitor.sketches.tdigest import TDigest


@dataclass
class DetectionBenchmarkResult:
    """Benchmark results for a specific drift scenario."""
    drift_type: str
    injection_index: int
    first_detected_index: Optional[int]
    detection_delay_records: Optional[int]
    detected_successfully: bool
    final_alert_severity: str
    detected_features: List[str]


class DriftBenchmarkSuite:
    """
    Automated benchmark runner evaluating quantitative metrics:
    Detection Delay, False Alarm Rate, Memory Savings, Quantile Errors, and CBPE Errors.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.injector = SyntheticDriftInjector(random_state=random_state)
        self.benchmark_dataset = StreamingFinancialBenchmark(random_state=random_state)

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Executes full suite of experiments and returns consolidated dictionary."""
        print("=== Step 1: Preparing Reference Data & Calibrated Model ===")
        split = self.benchmark_dataset.generate(n_reference=2500, n_stream=5000)

        model = BaselineModelWrapper(
            numerical_features=split.numerical_features,
            categorical_features=split.categorical_features,
            random_state=self.random_state,
        )
        base_metrics = model.train_and_calibrate(train_df=split.reference_df)

        results = {
            "timestamp": time.time(),
            "baseline_metrics": base_metrics.to_dict(),
            "experiment_1_detection_delay": self.benchmark_detection_delays(split, model),
            "experiment_2_false_alarm_rate": self.benchmark_false_alarm_rate(split, model),
            "experiment_3_sketches_accuracy_and_memory": self.benchmark_sketches(split.reference_df),
            "experiment_4_cbpe_estimation_error": self.benchmark_cbpe_accuracy(split, model),
            "experiment_5_multivariate_sensitivity": self.benchmark_multivariate_sensitivity(split, model),
        }

        return results

    def benchmark_detection_delays(
        self,
        split: Any,
        model: BaselineModelWrapper,
    ) -> List[Dict[str, Any]]:
        """
        Experiment 1: Measures detection delay across Abrupt, Gradual, Recurring,
        and Feature-Specific drift scenarios.
        """
        scenarios = [
            self.injector.inject_abrupt_drift(
                split.stream_df, injection_idx=600, affected_features=["tx_amount", "sender_risk_score"]
            ),
            self.injector.inject_gradual_drift(
                split.stream_df, start_idx=600, duration=800, affected_features=["tx_amount", "dest_novelty_score"]
            ),
            self.injector.inject_recurring_drift(
                split.stream_df, cycle_length=600, affected_features=["tx_amount", "sender_tx_count_1h"]
            ),
            self.injector.inject_feature_specific_drift(
                split.stream_df, injection_idx=600, shifts={"location_entropy": 2.5, "dest_novelty_score": 2.5}
            ),
        ]

        results = []
        for scenario in scenarios:
            config = MonitorConfig(window_size=200, check_interval=100)
            monitor = StreamingDriftMonitor(config=config, model=model)
            monitor.setup_reference(split.reference_df, train_model_if_missing=False)

            detected_idx = None
            detected_features: List[str] = []

            for i, rec in enumerate(scenario.stream_df.to_dict(orient="records")):
                res = monitor.process_record(rec)
                # Check for either streaming alarm (ADWIN/PH) or windowed alert (Warning/Critical)
                if i >= scenario.injection_start_idx:
                    if (res.streaming_drift_detected or res.alert_severity in [AlertSeverity.WARNING, AlertSeverity.CRITICAL]) and detected_idx is None:
                        detected_idx = i
                        detected_features = list(res.affected_features) or list(monitor.latest_alert_status.drifted_features)
                        break

            delay = (detected_idx - scenario.injection_start_idx) if detected_idx is not None else None
            results.append({
                "drift_type": scenario.drift_type,
                "injection_index": scenario.injection_start_idx,
                "first_detected_index": detected_idx,
                "detection_delay_records": delay,
                "detected_successfully": bool(detected_idx is not None),
                "final_alert_severity": monitor.latest_alert_status.severity.value,
                "detected_features": detected_features,
            })

        return results

    def benchmark_false_alarm_rate(
        self,
        split: Any,
        model: BaselineModelWrapper,
        n_stationary_records: int = 600,
    ) -> Dict[str, Any]:
        """
        Experiment 2: Measures False Alarm Rate (FAR) on a purely stationary,
        in-distribution stream.
        """
        config = MonitorConfig(window_size=200, check_interval=100)
        monitor = StreamingDriftMonitor(config=config, model=model)
        monitor.setup_reference(split.reference_df, train_model_if_missing=False)

        stationary_df = split.stream_df.iloc[:n_stationary_records]

        false_alarms_streaming = 0
        false_alarms_windowed = 0
        total_checks = 0

        for rec in stationary_df.to_dict(orient="records"):
            res = monitor.process_record(rec)
            if res.streaming_drift_detected:
                false_alarms_streaming += 1
            if res.is_periodic_check_step:
                total_checks += 1
                if res.alert_severity != AlertSeverity.NORMAL:
                    false_alarms_windowed += 1

        far_streaming = false_alarms_streaming / max(1, len(stationary_df))
        far_windowed = false_alarms_windowed / max(1, total_checks)

        return {
            "stationary_records_tested": len(stationary_df),
            "periodic_checks_conducted": total_checks,
            "streaming_false_alarms": false_alarms_streaming,
            "streaming_far": round(far_streaming, 5),
            "windowed_false_alarms": false_alarms_windowed,
            "windowed_far": round(far_windowed, 4),
            "fdr_correction_active": True,
        }

    def benchmark_sketches(self, reference_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Experiment 3: Compares sketch approximation against exact NumPy statistics
        and quantifies bounded memory consumption.
        """
        tx_data = reference_df["tx_amount"].values
        n = len(tx_data)

        # 1. t-digest evaluation
        td = TDigest(compression=100.0)
        for x in tx_data:
            td.update(x)
        td.compress()

        quantiles_to_test = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        quantile_errors = []

        for q in quantiles_to_test:
            exact_q = float(np.percentile(tx_data, q * 100))
            approx_q = float(td.quantile(q))
            abs_err = abs(approx_q - exact_q)
            rel_err = abs_err / max(1e-4, abs(exact_q))
            quantile_errors.append({
                "quantile": q,
                "exact": round(exact_q, 3),
                "approx": round(approx_q, 3),
                "absolute_error": round(abs_err, 4),
                "relative_error": round(rel_err, 4),
            })

        # 2. Count-Min Sketch evaluation
        dev_data = reference_df["device_type"].values
        exact_counts = pd.Series(dev_data).value_counts().to_dict()

        cms = CountMinSketch(epsilon=0.005, delta=0.01, conservative_update=True)
        for it in dev_data:
            cms.update(it)

        cms_errors = []
        for item, count in exact_counts.items():
            est_cnt = cms.estimate_frequency(item)
            cms_errors.append({
                "item": item,
                "exact_count": count,
                "estimated_count": est_cnt,
                "absolute_error": abs(est_cnt - count),
            })

        # 3. HyperLogLog evaluation
        exact_unique = len(set(dev_data))
        hll = HyperLogLog(precision_bits=10)
        for it in dev_data:
            hll.add(it)
        hll_est = hll.estimate_cardinality()
        hll_rel_err = abs(hll_est - exact_unique) / max(1, exact_unique)

        # 4. Memory footprint comparison
        raw_bytes = tx_data.nbytes
        tdigest_bytes = td.memory_bytes()
        cms_bytes = cms.memory_bytes()
        hll_bytes = hll.memory_bytes()
        total_sketches_bytes = tdigest_bytes + cms_bytes + hll_bytes

        return {
            "sample_size": n,
            "raw_stream_bytes": raw_bytes,
            "tdigest_bytes": tdigest_bytes,
            "cms_bytes": cms_bytes,
            "hll_bytes": hll_bytes,
            "total_sketches_bytes": total_sketches_bytes,
            "memory_saving_ratio": round(raw_bytes / max(1, tdigest_bytes), 2),
            "quantile_benchmarks": quantile_errors,
            "mean_quantile_rel_error": round(float(np.mean([x["relative_error"] for x in quantile_errors])), 4),
            "cms_benchmarks": cms_errors,
            "hll_benchmark": {
                "exact_cardinality": exact_unique,
                "estimated_cardinality": round(hll_est, 2),
                "relative_error": round(hll_rel_err, 4),
            },
        }

    def benchmark_cbpe_accuracy(
        self,
        split: Any,
        model: BaselineModelWrapper,
    ) -> Dict[str, Any]:
        """
        Experiment 4: Validates Confidence-Based Performance Estimation (CBPE)
        against actual ground-truth performance across normal and drifted stream windows.
        """
        config = MonitorConfig(window_size=400, check_interval=100)
        monitor = StreamingDriftMonitor(config=config, model=model)
        monitor.setup_reference(split.reference_df, train_model_if_missing=False)

        # Stream records and deliver delayed labels after 200 records
        records = split.stream_df.to_dict(orient="records")
        for rec in records[:1000]:
            monitor.process_record(rec)

        # Deliver delayed labels for the first 800 records
        labels_map = {
            rec["record_id"]: int(rec["is_fraud"])
            for rec in records[:800]
        }
        verification = monitor.receive_delayed_labels(labels_map)

        return {
            "matched_records": verification["matched_labels"],
            "latest_window_comparison": verification["latest_comparison"],
            "aggregate_estimation_errors": verification["aggregate_errors"],
        }

    def benchmark_multivariate_sensitivity(
        self,
        split: Any,
        model: BaselineModelWrapper,
    ) -> Dict[str, Any]:
        """
        Experiment 5: Demonstrates a scenario where univariate 1D marginal distributions
        experience negligible shift, but joint feature correlation shifts significantly.
        Validates MMD and C2ST detecting what univariate tests miss.
        """
        df_corr = split.stream_df.iloc[:800].copy().reset_index(drop=True)

        # Rotate correlation between tx_amount and sender_risk_score without changing marginal means
        x1 = df_corr["tx_amount"].values
        x2 = df_corr["sender_risk_score"].values
        mean1, std1 = np.mean(x1), np.std(x1)
        mean2, std2 = np.mean(x2), np.std(x2)

        z1 = (x1 - mean1) / max(1e-4, std1)
        z2 = (x2 - mean2) / max(1e-4, std2)

        # Inverse correlation rotation
        theta = np.pi / 3.0  # 60 degrees rotation
        rot_z1 = z1 * np.cos(theta) - z2 * np.sin(theta)
        rot_z2 = z1 * np.sin(theta) + z2 * np.cos(theta)

        df_corr["tx_amount"] = mean1 + rot_z1 * std1
        df_corr["sender_risk_score"] = mean2 + rot_z2 * std2

        config = MonitorConfig(window_size=400, check_interval=100)
        monitor = StreamingDriftMonitor(config=config, model=model)
        monitor.setup_reference(split.reference_df, train_model_if_missing=False)

        for rec in df_corr.to_dict(orient="records"):
            monitor.process_record(rec)

        summary = monitor.get_monitoring_summary()
        return {
            "multivariate_mmd_detected": summary["multivariate_drift"]["mmd"]["drift_detected"]
            if summary["multivariate_drift"]["mmd"] else False,
            "multivariate_c2st_auc": summary["multivariate_drift"]["c2st"]["roc_auc"]
            if summary["multivariate_drift"]["c2st"] else 0.5,
            "univariate_drifted_count": sum(
                1 for f in summary["univariate_drift"].values() if f.get("is_significant", False)
            ),
        }
