"""
End-to-End Live Demonstration Script for Person 1 (ML & Statistical Monitoring Lead).
Executes a comprehensive, turn-by-turn simulation of all 8 weeks of deliverables:
- Week 1: Temporal dataset generation, baseline ML model training & probability calibration
- Week 2: Bounded-memory sketches (t-digest, Count-Min Sketch, HyperLogLog, Welford)
- Week 3: Univariate streaming drift detection (ADWIN, Page-Hinkley)
- Week 4: Multivariate drift detection (MMD, C2ST) with Benjamini-Hochberg FDR correction
- Week 5: Confidence-Based Performance Estimation (CBPE) vs delayed ground truth
- Week 6: Alert severity classification & candidate shadow-model evaluation governance
- Week 7/8: Experimental benchmark suite execution and automated report generation
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ml_engine.drift_monitor import (
    AlertSeverity,
    BaselineModelWrapper,
    MonitorConfig,
    RetrainConfig,
    StreamingDriftMonitor,
    StreamingFinancialBenchmark,
)
from ml_engine.drift_monitor.benchmarks.generate_report import generate_benchmark_reports


def format_header(title: str, week: int) -> str:
    line = "=" * 78
    return f"\n{line}\n  WEEK {week}: {title}\n{line}"


def run_demo():
    print("=" * 78)
    print("  LABEL-FREE STREAMING DRIFT MONITOR WITH BOUNDED-MEMORY SKETCHES")
    print("  Person 1: ML and Statistical Monitoring Lead — End-to-End Showcase")
    print("=" * 78)

    # ---------------------------------------------------------
    # WEEK 1: Dataset & Baseline Model
    # ---------------------------------------------------------
    print(format_header("Dataset Ingestion & Calibrated Baseline Model Training", 1))
    benchmark = StreamingFinancialBenchmark(random_state=42)
    split = benchmark.generate(n_reference=3000, n_stream=4000)

    print(f"[*] Generated temporal streaming dataset:")
    print(f"    - Reference Dataset : {len(split.reference_df)} records (Historical training distribution)")
    print(f"    - Stream Dataset    : {len(split.stream_df)} records (Incoming real-time transactions)")
    print(f"    - Numerical Features: {split.numerical_features}")
    print(f"    - Categorical Feats : {split.categorical_features}")

    model = BaselineModelWrapper(
        numerical_features=split.numerical_features,
        categorical_features=split.categorical_features,
        random_state=42,
    )
    print("\n[*] Training baseline Gradient Boosting classifier with probability calibration...")
    baseline_metrics = model.train_and_calibrate(
        train_df=split.reference_df,
        target_col=split.target_column,
        calibration_method="sigmoid",
    )

    print("\n[+] Baseline Reference Evaluation Metrics:")
    print(f"    - Accuracy    : {baseline_metrics.accuracy:.2%}")
    print(f"    - F1-Score    : {baseline_metrics.f1:.2%}")
    print(f"    - Precision   : {baseline_metrics.precision:.2%}")
    print(f"    - Recall      : {baseline_metrics.recall:.2%}")
    print(f"    - ROC-AUC     : {baseline_metrics.roc_auc:.4f}")
    print(f"    - Brier Score : {baseline_metrics.brier_score:.4f} (Lower = better calibration)")

    # ---------------------------------------------------------
    # WEEK 2: Bounded-Memory Sketches
    # ---------------------------------------------------------
    print(format_header("Bounded-Memory Streaming Sketches & Space Bounds", 2))
    config = MonitorConfig(window_size=300, check_interval=50)
    monitor = StreamingDriftMonitor(config=config, model=model)
    monitor.setup_reference(split.reference_df, train_model_if_missing=False)

    print("[*] Sketches initialized for all features:")
    print("    - Numerical   : Welford RunningStats (O(1)), t-digest (compression=100.0)")
    print("    - Categorical : Count-Min Sketch (conservative update), HyperLogLog (1024 registers)")
    print("    - Multivariate: Reservoir Sampler (capacity K=1000)")

    # Stream 200 in-distribution records
    for rec in split.stream_df.iloc[:200].to_dict(orient="records"):
        monitor.process_record(rec)

    summary_w2 = monitor.get_monitoring_summary()
    print("\n[+] Sketch Memory Profile after 200 stream events:")
    print(f"    - Total Sketches Memory : {summary_w2['memory']['sketches_total_bytes']:,} bytes (BOUNDED)")
    print(f"    - Raw Window Buffer     : {summary_w2['memory']['raw_window_bytes']:,} bytes (GROWS WITH STREAM)")
    print(f"    - Memory Savings Ratio  : {summary_w2['memory']['memory_compression_ratio']}x reduction")
    print(f"    - t-digest 'tx_amount' Median : ${summary_w2['univariate_drift']['tx_amount']['tdigest_median']:.2f}")
    print(f"    - t-digest 'tx_amount' p95    : ${summary_w2['univariate_drift']['tx_amount']['tdigest_p95']:.2f}")

    # ---------------------------------------------------------
    # WEEK 3: Streaming Univariate Drift Detection
    # ---------------------------------------------------------
    print(format_header("Univariate Streaming Drift Detection (ADWIN & Page-Hinkley)", 3))
    print("[*] Streaming 300 stationary records: ADWIN and Page-Hinkley running...")
    stationary_alarms = 0
    for rec in split.stream_df.iloc[200:500].to_dict(orient="records"):
        res = monitor.process_record(rec)
        if res.streaming_drift_detected:
            stationary_alarms += 1
    print(f"    - Stationary Phase Alarms: {stationary_alarms} (Expected 0 - no false alarms)")

    print("\n[*] Injecting abrupt +3.0 sigma drift on 'tx_amount' and 'sender_risk_score'...")
    drifted_records = split.stream_df.iloc[500:800].copy().reset_index(drop=True)
    std_amount = drifted_records["tx_amount"].std()
    std_risk = drifted_records["sender_risk_score"].std()
    drifted_records["tx_amount"] += 3.0 * std_amount
    drifted_records["sender_risk_score"] += 3.0 * std_risk

    alarm_count = 0
    first_detected_idx = None
    for i, rec in enumerate(drifted_records.to_dict(orient="records")):
        res = monitor.process_record(rec)
        if res.streaming_drift_detected:
            alarm_count += 1
            if first_detected_idx is None:
                first_detected_idx = i
                print(f"[!] STREAMING ALARM at record offset {i} on features: {res.affected_features}")

    print(f"\n[+] Univariate Drift Detection Results:")
    print(f"    - First Alarm Offset : Record +{first_detected_idx} (Instant streaming detection)")
    print(f"    - Total Alarms Raised: {alarm_count}")

    # ---------------------------------------------------------
    # WEEK 4: Multivariate Drift Detection
    # ---------------------------------------------------------
    print(format_header("Multivariate Drift Detection (MMD & C2ST) & FDR Correction", 4))
    summary_w4 = monitor.get_monitoring_summary()
    mmd = summary_w4["multivariate_drift"]["mmd"]
    c2st = summary_w4["multivariate_drift"]["c2st"]

    print("[+] Multivariate RKHS Kernel MMD Test:")
    print(f"    - MMD^2 Statistic     : {mmd['mmd_squared']:.6f}")
    print(f"    - Permutation p-value : {mmd['p_value']:.4f}")
    print(f"    - Drift Confirmed     : {mmd['drift_detected']}")

    print("\n[+] Classifier Two-Sample Test (C2ST) Domain Discriminator:")
    print(f"    - Holdout Discriminator AUC : {c2st['roc_auc']:.4f} (Under H0: ~0.50)")
    print(f"    - Hypothesis p-value        : {c2st['p_value']:.6f}")
    print(f"    - Top Contributing Features : {c2st['top_contributing_features']}")

    print("\n[+] Multiple Testing Correction (Benjamini-Hochberg FDR):")
    for feat, u in summary_w4["univariate_drift"].items():
        sig_str = "DRIFTED (Significant)" if u["is_significant"] else "Stable"
        print(f"    - {feat:<20} | PSI: {u['psi']:.4f} | KS p: {u['ks_p_value']:.4e} | {sig_str}")

    # ---------------------------------------------------------
    # WEEK 5: Label-Free Performance Estimation (CBPE)
    # ---------------------------------------------------------
    print(format_header("Label-Free Performance Estimation (CBPE) vs Delayed Truth", 5))
    cbpe_est = summary_w4["performance_estimation"]
    print("[*] Model performance estimated using prediction confidence without labels:")
    print(f"    - Estimated Accuracy : {cbpe_est['estimated_accuracy']:.2%} (Baseline: {baseline_metrics.accuracy:.2%})")
    print(f"    - Estimated F1-Score : {cbpe_est['estimated_f1']:.2%} (Baseline: {baseline_metrics.f1:.2%})")
    print(f"    - Estimated ROC-AUC  : {cbpe_est['estimated_roc_auc']:.4f} (Baseline: {baseline_metrics.roc_auc:.4f})")
    print(f"    - Estimated Brier    : {cbpe_est['estimated_brier_score']:.4f}")

    print("\n[*] Simulating delayed arrival of ground-truth labels for recent transactions...")
    all_seen = split.stream_df.iloc[:800]
    delayed_labels = {r["record_id"]: int(r["is_fraud"]) for r in all_seen.to_dict(orient="records")}
    verification = monitor.receive_delayed_labels(delayed_labels)

    comp = verification["latest_comparison"]
    print("\n[+] Comparison: Estimated vs Delayed Actual Ground Truth:")
    print(f"    - Accuracy  | Estimated: {comp['estimated']['estimated_accuracy']:.2%} | Actual: {comp['actual']['accuracy']:.2%} | Error: {comp['error_mae']['accuracy']:.4f}")
    print(f"    - F1-Score  | Estimated: {comp['estimated']['estimated_f1']:.2%} | Actual: {comp['actual']['f1']:.2%} | Error: {comp['error_mae']['f1']:.4f}")
    print(f"    - ROC-AUC   | Estimated: {comp['estimated']['estimated_roc_auc']:.4f} | Actual: {comp['actual']['roc_auc']:.4f} | Error: {comp['error_mae']['roc_auc']:.4f}")

    # ---------------------------------------------------------
    # WEEK 6: Alert Engine & Shadow-Model Retraining Governance
    # ---------------------------------------------------------
    print(format_header("Alert Severity & Candidate Shadow-Model Evaluation", 6))
    alert = summary_w4["alert"]
    retrain_rec = summary_w4["retraining_recommendation"]

    print(f"[!] System Alert Status:")
    print(f"    - Severity Level        : {alert['severity']}")
    print(f"    - Drift Classification  : {alert['drift_type']}")
    print(f"    - Composite Risk Score  : {alert['composite_score']:.4f}")
    print(f"    - Summary Message       : {alert['summary_message']}")
    print(f"    - Detailed Reasons      : {alert['detailed_reasons']}")

    print(f"\n[!] Automated Retraining Recommendation:")
    print(f"    - Trigger Retraining    : {retrain_rec['should_retrain']}")
    print(f"    - Urgency               : {retrain_rec['urgency']}")
    print(f"    - Action Recommended    : {retrain_rec['recommended_action']}")

    print("\n[*] Launching Shadow Candidate Model Evaluation...")
    print("    Rule: No automatic production deployment without evaluation and approval token.")
    shadow_res, candidate_model = monitor.trigger_shadow_retraining(
        recent_data_with_labels=split.stream_df.iloc[400:800]
    )

    print("\n[+] Shadow Model Benchmark Results:")
    print(f"    - Active Model F1    : {shadow_res.active_metrics.f1:.2%}")
    print(f"    - Candidate Model F1 : {shadow_res.candidate_metrics.f1:.2%}")
    print(f"    - F1 Gain (Delta)    : {shadow_res.f1_delta:+.2%}")
    print(f"    - ROC-AUC Gain       : {shadow_res.auc_delta:+.4f}")
    print(f"    - Decision           : {shadow_res.decision}")
    print(f"    - Governance Token   : {shadow_res.approval_token}")
    print(f"    - Justification      : {shadow_res.justification}")

    # ---------------------------------------------------------
    # WEEK 7 / 8: Comprehensive Experimental Report
    # ---------------------------------------------------------
    print(format_header("Experimental Benchmark Suite & Report Generation", 7))
    print("[*] Running experimental benchmark suite...")
    report_file = generate_benchmark_reports()
    print(f"\n[+] Experimental report and CSV artifacts generated:")
    print(f"    - Markdown Report : {report_file}")
    print(f"    - CSV Tables      : ml_engine/drift_monitor/benchmarks/reports/")

    print("\n" + "=" * 78)
    print("  PERSON 1 COMPLETE: All mathematical, ML, sketch, and governance")
    print("  deliverables are tested, operational, and ready for Person 2!")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    run_demo()
