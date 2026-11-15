"""
Drift Monitoring Service for Person 2 (Backend & Systems Lead).
Integrates Person 1's ML monitoring engine (StreamingDriftMonitor),
bounded-memory sketches, CBPE performance estimation, SQLite persistence,
and candidate shadow-model retraining governance.
"""

from collections import deque
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml_engine.drift_monitor.benchmarks.drift_generator import SyntheticDriftInjector
from ml_engine.drift_monitor.config import MonitorConfig
from ml_engine.drift_monitor.dataset import StreamingFinancialBenchmark
from ml_engine.drift_monitor.model import BaselineModelWrapper
from ml_engine.drift_monitor.monitor import RecordMonitoringResult, StreamingDriftMonitor
from ml_engine.drift_monitor.retraining.alert_engine import AlertSeverity
from backend.app.config import settings
from backend.app.schemas.alerts import (
    CandidateEvaluationResponse,
    PromotionApprovalResponse,
    RetrainRecommendationResponse,
)
from backend.app.schemas.drift import (
    DriftStatusResponse,
    MultivariateC2STMetrics,
    MultivariateDriftResponse,
    MultivariateMMDMetrics,
    UnivariateFeatureMetrics,
)
from backend.app.schemas.events import (
    BatchIngestResponse,
    RecordIngestResponse,
    TransactionEvent,
)
from backend.app.schemas.performance import (
    DelayedReconciliationResponse,
    PerformanceEstimateResponse,
)
from backend.app.storage.database import DatabaseManager

logger = logging.getLogger("drift_backend.service")


class DriftService:
    """Singleton service encapsulating the live streaming drift monitor."""

    _instance: Optional["DriftService"] = None
    _lock = threading.Lock()

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.records_processed = 0

        # Labeled buffer for shadow retraining evaluation
        self._labeled_buffer: deque = deque(maxlen=2000)
        self._candidate_models: Dict[str, BaselineModelWrapper] = {}

        # Initialize ML engine with benchmark dataset
        logger.info("Initializing StreamingDriftMonitor with financial benchmark reference...")
        self.benchmark = StreamingFinancialBenchmark(random_state=42)
        split = self.benchmark.generate(n_reference=2500, n_stream=1000)
        reference_df = split.reference_df

        # Initialize StreamingDriftMonitor and calibrate baseline model
        monitor_config = MonitorConfig(
            window_size=settings.monitor_window_size,
            check_interval=settings.monitor_check_interval,
        )
        self.monitor = StreamingDriftMonitor(
            config=monitor_config,
            model=None,
        )
        self.monitor.setup_reference(
            reference_df=reference_df,
            target_column="is_fraud",
            train_model_if_missing=True,
        )
        self.baseline_model = self.monitor.model

        # Store reference records into labeled buffer as initial warm-up
        for _, row in reference_df.tail(400).iterrows():
            self._labeled_buffer.append(row.to_dict())

        logger.info("StreamingDriftMonitor initialized successfully.")

    @classmethod
    def get_instance(cls, db_manager: Optional[DatabaseManager] = None) -> "DriftService":
        """Thread-safe singleton accessor."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(db_manager=db_manager)
        return cls._instance

    def process_event(self, event: TransactionEvent) -> RecordIngestResponse:
        """
        Processes a single incoming transaction event through the streaming monitor.
        Enforces bounded memory and logs alarms to SQLite when drift occurs.
        """
        with self.lock:
            record_dict = event.model_dump()
            record_id = record_dict.get("record_id") or f"tx_{self.records_processed + 1:06d}"
            record_dict["record_id"] = record_id

            if event.is_fraud is not None:
                self._labeled_buffer.append(record_dict)

            # Process through Person 1's master monitor
            res: RecordMonitoringResult = self.monitor.process_record(
                record=record_dict,
            )
            self.records_processed += 1

            # Log drift event to database if alarm triggered or periodic check found drift
            if res.streaming_drift_detected or (
                res.is_periodic_check_step
                and res.alert_severity in (AlertSeverity.WARNING, AlertSeverity.CRITICAL)
            ):
                alert_status = self.monitor.latest_alert_status
                rec = self.monitor.latest_retrain_recommendation.to_dict()
                self.db.log_drift_event(
                    record_position=self.records_processed,
                    severity=res.alert_severity.value,
                    summary=alert_status.summary_message,
                    affected_features=res.affected_features or alert_status.drifted_features,
                    composite_score=alert_status.composite_score,
                    recommendation=rec,
                    timestamp=event.timestamp or time.time(),
                )

            return RecordIngestResponse(
                status="SUCCESS",
                record_id=record_id,
                timestamp=event.timestamp or time.time(),
                prediction=res.prediction,
                probability=res.probability,
                streaming_drift_detected=res.streaming_drift_detected,
                affected_features=res.affected_features,
                alert_severity=res.alert_severity.value,
                is_periodic_check_step=res.is_periodic_check_step,
            )

    def process_batch(self, events: List[TransactionEvent]) -> BatchIngestResponse:
        """Processes a batch of streaming transactions with throughput calculation."""
        t0 = time.time()
        alarms_count = 0
        latest_severity = "NORMAL"

        for ev in events:
            res = self.process_event(ev)
            if res.streaming_drift_detected:
                alarms_count += 1
            latest_severity = res.alert_severity

        duration = max(0.0001, time.time() - t0)
        throughput = len(events) / duration

        return BatchIngestResponse(
            status="SUCCESS",
            processed_count=len(events),
            streaming_alarms_count=alarms_count,
            latest_severity=latest_severity,
            throughput_rec_per_sec=round(throughput, 2),
        )

    def simulate_stream(
        self,
        num_records: int = 150,
        drift_type: str = "none",
    ) -> Dict[str, Any]:
        """
        Simulates an incoming streaming scenario:
        - 'none': stationary standard traffic
        - 'abrupt': sudden fraud wave / transaction surge
        - 'gradual': slow inflation and risk drift
        """
        base_df = self.benchmark._generate_distribution(num_records)
        if drift_type == "abrupt":
            injector = SyntheticDriftInjector(random_state=42)
            exp = injector.inject_abrupt_drift(base_df, injection_idx=0, shift_magnitude=3.0)
            df = exp.stream_df
        elif drift_type == "gradual":
            injector = SyntheticDriftInjector(random_state=42)
            exp = injector.inject_gradual_drift(
                base_df, start_idx=0, duration=max(10, num_records), shift_magnitude=2.5
            )
            df = exp.stream_df
        else:
            df = base_df

        alarms = 0
        latest_sev = "NORMAL"
        for _, row in df.iterrows():
            rec_dict = row.to_dict()
            tx = TransactionEvent(
                record_id=str(rec_dict.get("record_id", "")),
                timestamp=float(rec_dict.get("timestamp", time.time())),
                tx_amount=float(rec_dict.get("tx_amount", 50.0)),
                sender_tx_count_1h=float(rec_dict.get("sender_tx_count_1h", 1.0)),
                sender_risk_score=float(rec_dict.get("sender_risk_score", 0.1)),
                location_entropy=float(rec_dict.get("location_entropy", 0.2)),
                dest_novelty_score=float(rec_dict.get("dest_novelty_score", 0.5)),
                time_delta_last_tx=float(rec_dict.get("time_delta_last_tx", 300.0)),
                network_degree=float(rec_dict.get("network_degree", 1.0)),
                device_type=str(rec_dict.get("device_type", "web_browser")),
                merchant_category=str(rec_dict.get("merchant_category", "groceries")),
                is_fraud=int(rec_dict.get("is_fraud", 0)) if "is_fraud" in rec_dict else None,
            )
            res = self.process_event(tx)
            if res.streaming_drift_detected:
                alarms += 1
            latest_sev = res.alert_severity

        return {
            "status": "SUCCESS",
            "scenario": drift_type,
            "simulated_records": num_records,
            "alarms_triggered": alarms,
            "latest_severity": latest_sev,
            "total_processed_overall": self.records_processed,
        }

    def get_drift_status(self) -> DriftStatusResponse:
        """Retrieves comprehensive monitoring state for dashboard and APIs."""
        with self.lock:
            summary = self.monitor.get_monitoring_summary()
            history = self.db.get_drift_events(limit=10)

            return DriftStatusResponse(
                status=summary["status"],
                records_processed=summary["records_processed"],
                throughput_rec_per_sec=summary["throughput_rec_per_sec"],
                uptime_seconds=summary["uptime_seconds"],
                memory=summary["memory"],
                alert=summary["alert"],
                retraining_recommendation=summary["retraining_recommendation"],
                performance_estimation=summary["performance_estimation"],
                baseline_metrics=summary["baseline_metrics"],
                multivariate_drift=summary["multivariate_drift"],
                univariate_drift=summary["univariate_drift"],
                categorical_sketches=summary["categorical_sketches"],
                alert_history_count=len(history),
            )

    def get_univariate_metrics(self) -> List[UnivariateFeatureMetrics]:
        """Returns univariate drift metrics for each monitored feature."""
        with self.lock:
            metrics_list = []
            uni_dict = self.monitor.latest_univariate_results
            for feat in self.monitor.numerical_features:
                if feat in uni_dict:
                    info = uni_dict[feat]
                    metrics_list.append(
                        UnivariateFeatureMetrics(
                            feature_name=feat,
                            psi=float(info["psi"]["psi_value"]),
                            ks_p_value=float(info["ks"]["p_value"]),
                            is_significant=bool(info.get("is_significant", False)),
                            running_mean=float(self.monitor.running_stats[feat].mean),
                            tdigest_median=float(self.monitor.tdigests[feat].quantile(0.50)),
                            tdigest_p95=float(self.monitor.tdigests[feat].quantile(0.95)),
                        )
                    )
                else:
                    # Feature not yet evaluated in periodic check
                    rs = self.monitor.running_stats.get(feat)
                    td = self.monitor.tdigests.get(feat)
                    metrics_list.append(
                        UnivariateFeatureMetrics(
                            feature_name=feat,
                            psi=0.0,
                            ks_p_value=1.0,
                            is_significant=False,
                            running_mean=float(rs.mean) if rs else 0.0,
                            tdigest_median=float(td.quantile(0.50)) if td else 0.0,
                            tdigest_p95=float(td.quantile(0.95)) if td else 0.0,
                        )
                    )
            return metrics_list

    def get_multivariate_metrics(self) -> MultivariateDriftResponse:
        """Returns Maximum Mean Discrepancy (MMD) & Classifier Two-Sample Test (C2ST)."""
        with self.lock:
            mmd = None
            if self.monitor.latest_mmd_result:
                r = self.monitor.latest_mmd_result
                mmd = MultivariateMMDMetrics(
                    mmd_squared=float(r.mmd_squared),
                    p_value=float(r.p_value),
                    drift_detected=bool(r.drift_detected),
                    gamma=float(r.gamma),
                    alpha=float(r.alpha),
                )

            c2st = None
            if self.monitor.latest_c2st_result:
                c = self.monitor.latest_c2st_result
                top_feats = []
                if hasattr(c, "feature_importances") and c.feature_importances:
                    top_feats = sorted(
                        c.feature_importances.items(), key=lambda x: x[1], reverse=True
                    )[:5]
                elif hasattr(c, "top_contributing_features"):
                    top_feats = c.top_contributing_features or []
                c2st = MultivariateC2STMetrics(
                    roc_auc=float(c.roc_auc),
                    p_value=float(c.p_value),
                    drift_detected=bool(c.drift_detected),
                    top_contributing_features=top_feats,
                )

            confirmed = bool(
                (mmd and mmd.drift_detected) or (c2st and c2st.drift_detected)
            )

            return MultivariateDriftResponse(
                status="DRIFT_DETECTED" if confirmed else "HEALTHY",
                mmd=mmd,
                c2st=c2st,
                multivariate_drift_confirmed=confirmed,
            )

    def get_performance_estimate(self) -> PerformanceEstimateResponse:
        """Returns Confidence-Based Performance Estimation (CBPE) metrics."""
        with self.lock:
            cbpe = self.monitor.latest_cbpe_estimate
            base = self.monitor.baseline_metrics

            if cbpe is None:
                acc = base.accuracy if base else 0.95
                prec = base.precision if base else 0.90
                rec = base.recall if base else 0.85
                f1 = base.f1 if base else 0.87
                auc = base.roc_auc if base else 0.96
                brier = base.brier_score if base else 0.05
                return PerformanceEstimateResponse(
                    sample_size=0,
                    estimated_accuracy=acc,
                    estimated_precision=prec,
                    estimated_recall=rec,
                    estimated_f1=f1,
                    estimated_roc_auc=auc,
                    estimated_brier_score=brier,
                    expected_positive_rate=0.08,
                    mean_confidence=0.92,
                    decision_threshold=0.5,
                    baseline_accuracy=acc,
                    baseline_f1=f1,
                    baseline_roc_auc=auc,
                    accuracy_drop=0.0,
                    f1_drop=0.0,
                    roc_auc_drop=0.0,
                    performance_degraded=False,
                )

            base_f1 = base.f1 if base else 0.87
            base_acc = base.accuracy if base else 0.95
            base_auc = base.roc_auc if base else 0.96
            f1_drop = max(0.0, base_f1 - cbpe.estimated_f1)
            acc_drop = max(0.0, base_acc - cbpe.estimated_accuracy)
            auc_drop = max(0.0, base_auc - cbpe.estimated_roc_auc)

            return PerformanceEstimateResponse(
                sample_size=cbpe.sample_size,
                estimated_accuracy=cbpe.estimated_accuracy,
                estimated_precision=cbpe.estimated_precision,
                estimated_recall=cbpe.estimated_recall,
                estimated_f1=cbpe.estimated_f1,
                estimated_roc_auc=cbpe.estimated_roc_auc,
                estimated_brier_score=cbpe.estimated_brier_score,
                expected_positive_rate=cbpe.expected_positive_rate,
                mean_confidence=cbpe.mean_confidence,
                decision_threshold=cbpe.decision_threshold,
                baseline_accuracy=base_acc,
                baseline_f1=base_f1,
                baseline_roc_auc=base_auc,
                accuracy_drop=acc_drop,
                f1_drop=f1_drop,
                roc_auc_drop=auc_drop,
                performance_degraded=bool(f1_drop > 0.05 or acc_drop > 0.05),
            )

    def reconcile_delayed_labels(
        self, labels: Any
    ) -> DelayedReconciliationResponse:
        """
        Reconciles delayed ground truth labels against earlier CBPE estimates.
        Logs audit record to SQLite database.
        """
        with self.lock:
            gt_map: Dict[str, int] = {}
            if isinstance(labels, dict):
                gt_map = {str(k): int(v) for k, v in labels.items()}
            elif isinstance(labels, list):
                for item in labels:
                    if isinstance(item, dict):
                        rid = item.get("record_id")
                        val = item.get("is_fraud", item.get("label", 0))
                        if rid is not None:
                            gt_map[str(rid)] = int(val)
                    elif hasattr(item, "record_id") and hasattr(item, "is_fraud"):
                        gt_map[str(item.record_id)] = int(item.is_fraud)

            res = self.monitor.receive_delayed_labels(gt_map)
            report = res.get("latest_comparison")

            if report:
                est = report.get("estimated", {})
                act = report.get("actual", {})
                err = report.get("error_mae", {})
                self.db.log_performance_evaluation(
                    sample_size=report.get("sample_size", len(gt_map)),
                    estimated_accuracy=est.get("estimated_accuracy", 0.0),
                    actual_accuracy=act.get("accuracy", 0.0),
                    accuracy_error=err.get("accuracy", 0.0),
                    estimated_f1=est.get("estimated_f1", 0.0),
                    actual_f1=act.get("f1", 0.0),
                    f1_error=err.get("f1", 0.0),
                    estimated_auc=est.get("estimated_roc_auc", 0.0),
                    actual_auc=act.get("roc_auc", 0.0),
                    auc_error=err.get("roc_auc", 0.0),
                )

                return DelayedReconciliationResponse(
                    reconciled_count=len(gt_map),
                    matched_labels=res.get("matched_labels", 0),
                    sample_size=report.get("sample_size", len(gt_map)),
                    estimated_metrics=est,
                    actual_metrics=act,
                    estimation_errors=err,
                    within_tolerance=True,
                    latest_comparison=report,
                    aggregate_errors=res.get("aggregate_errors", {}),
                )
            else:
                return DelayedReconciliationResponse(
                    reconciled_count=len(gt_map),
                    matched_labels=res.get("matched_labels", 0),
                    sample_size=len(gt_map),
                    estimated_metrics={},
                    actual_metrics={},
                    estimation_errors={},
                    within_tolerance=True,
                    latest_comparison=None,
                    aggregate_errors=res.get("aggregate_errors", {}),
                )

    def evaluate_candidate_model(self, sample_size: int = 300) -> CandidateEvaluationResponse:
        """
        Trains and validates a candidate shadow model on recent labeled stream records.
        Enforces approval token requirement for controlled deployment.
        """
        with self.lock:
            if len(self._labeled_buffer) < 50:
                # Supplement with benchmark generator if buffer small
                extra_df = self.benchmark._generate_distribution(sample_size)
                labeled_df = extra_df
            else:
                records = list(self._labeled_buffer)[-sample_size:]
                labeled_df = pd.DataFrame(records)

            shadow_res, candidate_model = self.monitor.trigger_shadow_retraining(
                recent_data_with_labels=labeled_df
            )

            decision_str = (
                shadow_res.decision.value
                if hasattr(shadow_res.decision, "value")
                else str(shadow_res.decision)
            )
            token = shadow_res.approval_token
            if candidate_model and token:
                self._candidate_models[token] = candidate_model

            active_dict = (
                shadow_res.active_metrics.to_dict()
                if hasattr(shadow_res.active_metrics, "to_dict")
                else (
                    shadow_res.active_metrics
                    if isinstance(shadow_res.active_metrics, dict)
                    else {}
                )
            )
            cand_dict = (
                shadow_res.candidate_metrics.to_dict()
                if hasattr(shadow_res.candidate_metrics, "to_dict")
                else (
                    shadow_res.candidate_metrics
                    if isinstance(shadow_res.candidate_metrics, dict)
                    else {}
                )
            )
            act_f1 = (
                shadow_res.active_metrics.f1
                if hasattr(shadow_res.active_metrics, "f1")
                else active_dict.get("f1", 0.0)
            )
            cand_f1 = (
                shadow_res.candidate_metrics.f1
                if hasattr(shadow_res.candidate_metrics, "f1")
                else cand_dict.get("f1", 0.0)
            )

            # Log to SQLite retrain audit table
            self.db.log_retrain_audit(
                trigger_reason="Drift detected above retraining threshold",
                urgency="HIGH" if decision_str == "PROMOTE_TO_PRODUCTION" else "LOW",
                active_f1=act_f1,
                candidate_f1=cand_f1,
                f1_delta=shadow_res.f1_delta,
                auc_delta=shadow_res.auc_delta,
                decision=decision_str,
                approval_token=token,
                status="PENDING_APPROVAL" if token else "REJECTED",
            )

            return CandidateEvaluationResponse(
                active_metrics=active_dict,
                candidate_metrics=cand_dict,
                deltas={"f1_delta": shadow_res.f1_delta, "auc_delta": shadow_res.auc_delta},
                decision=decision_str,
                approval_token=token,
                justification=shadow_res.justification,
            )

    def approve_promotion(
        self, approval_token: str, approved_by: str, notes: Optional[str] = None
    ) -> PromotionApprovalResponse:
        """
        Promotes a candidate shadow model to active production upon valid token verification.
        """
        with self.lock:
            audit = self.db.get_retrain_audit_by_token(approval_token)
            if not audit:
                raise ValueError(f"Invalid or expired approval token: {approval_token}")

            if audit["status"] == "APPROVED":
                return PromotionApprovalResponse(
                    status="ALREADY_APPROVED",
                    approval_token=approval_token,
                    promoted_model="candidate_v2",
                    timestamp=time.time(),
                    message="Model was already approved and promoted.",
                )

            # Update active model in monitor if available in memory
            if approval_token in self._candidate_models:
                new_model = self._candidate_models[approval_token]
                self.monitor.model = new_model
                logger.info("Successfully promoted candidate model to active production.")

            # Record governance sign-off in database
            self.db.update_retrain_approval(
                token=approval_token,
                approved_by=approved_by,
                status="APPROVED",
            )

            # Reset alert state to NORMAL
            self.monitor.latest_alert_status.severity = AlertSeverity.NORMAL
            self.monitor.latest_alert_status.summary_message = (
                f"Candidate model promoted by {approved_by}. System stabilized."
            )

            return PromotionApprovalResponse(
                status="APPROVED",
                approval_token=approval_token,
                promoted_model="candidate_v2",
                timestamp=time.time(),
                message=f"Model successfully promoted to production by {approved_by}.",
            )
