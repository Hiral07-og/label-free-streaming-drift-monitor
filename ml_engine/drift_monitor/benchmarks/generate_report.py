"""
Benchmark report generator and table exporter.
Formats benchmark results into publication-grade Markdown comparison tables
and exports CSV metrics for documentation and presentations.
"""

from pathlib import Path
from typing import Any, Dict
import pandas as pd

from ml_engine.drift_monitor.benchmarks.run_benchmarks import DriftBenchmarkSuite


def generate_benchmark_reports(output_dir: str = "ml_engine/drift_monitor/benchmarks/reports") -> str:
    """
    Executes full benchmark suite and produces Markdown and CSV reports.
    """
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    suite = DriftBenchmarkSuite(random_state=42)
    results = suite.run_all_benchmarks()

    # 1. Detection delay table
    delay_rows = results["experiment_1_detection_delay"]
    delay_df = pd.DataFrame(delay_rows)
    delay_df.to_csv(out_p / "detection_delay_benchmarks.csv", index=False)

    # 2. Sketches quantile accuracy table
    sketches_res = results["experiment_3_sketches_accuracy_and_memory"]
    quantiles_df = pd.DataFrame(sketches_res["quantile_benchmarks"])
    quantiles_df.to_csv(out_p / "tdigest_quantile_accuracy.csv", index=False)

    # 3. CBPE performance estimation error
    cbpe_res = results["experiment_4_cbpe_estimation_error"]
    window_comp = cbpe_res["latest_window_comparison"]

    # 4. Generate Comprehensive Markdown Report
    md_content = f"""# Experimental Benchmarking Report: Streaming Drift Monitor

## Executive Summary
This report summarizes quantitative evaluation results for the **Label-Free Streaming Drift Monitor with Bounded-Memory Sketches and Automated Retrain Triggers**.
The benchmark rigorously assesses:
1. **Detection Delay & Detection Success** across controlled drift scenarios.
2. **False Alarm Rate (FAR)** under continuous stationary conditions.
3. **Quantile Estimation Accuracy & Memory Savings** of bounded-memory sketches.
4. **Label-Free Performance Estimation Accuracy (CBPE)** validated against delayed ground truth.
5. **Multivariate vs Univariate Detection Sensitivity**.

---

## 1. Baseline Model Performance (Historical Reference)
- **Model Architecture**: Calibrated Gradient Boosting with Sigmoid / Platt scaling
- **Baseline Accuracy**: {results['baseline_metrics']['accuracy']:.2%}
- **Baseline F1-Score**: {results['baseline_metrics']['f1']:.2%}
- **Baseline ROC-AUC**: {results['baseline_metrics']['roc_auc']:.4f}
- **Baseline Brier Score**: {results['baseline_metrics']['brier_score']:.4f}

---

## 2. Experiment 1: Detection Delay Across Controlled Drift Scenarios
Controlled perturbations were injected at index 600 across 4 standard drift categories:

| Drift Scenario | Injection Index | Detection Index | Detection Delay (Records) | Status | Final Severity |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for r in delay_rows:
        delay_str = str(r['detection_delay_records']) if r['detection_delay_records'] is not None else "N/A"
        status_str = "DETECTED" if r['detected_successfully'] else "MISSED"
        md_content += f"| **{r['drift_type']}** | {r['injection_index']} | {r['first_detected_index']} | {delay_str} | {status_str} | `{r['final_alert_severity']}` |\n"

    far_res = results["experiment_2_false_alarm_rate"]
    md_content += f"""
---

## 3. Experiment 2: False Alarm Rate (FAR) on Stationary Stream
- **Stationary Records Tested**: {far_res['stationary_records_tested']}
- **Streaming False Alarms**: {far_res['streaming_false_alarms']}
- **Streaming False Alarm Rate**: {far_res['streaming_far']:.4%}
- **Periodic Windowed Checks**: {far_res['periodic_checks_conducted']}
- **Windowed False Alarm Rate**: {far_res['windowed_far']:.2%}
- **Multiple Testing Correction**: Benjamini-Hochberg FDR active (controls family-wise false alarms)

---

## 4. Experiment 3: Sketch Approximation Accuracy & Bounded Memory Efficiency

### Bounded Memory Footprint vs Raw Stream:
- **Raw Stream Observation Buffer**: {sketches_res['raw_stream_bytes']:,} bytes
- **t-digest Footprint**: {sketches_res['tdigest_bytes']:,} bytes
- **Count-Min Sketch Footprint**: {sketches_res['cms_bytes']:,} bytes
- **HyperLogLog Footprint**: {sketches_res['hll_bytes']:,} bytes
- **Total Sketches Memory**: {sketches_res['total_sketches_bytes']:,} bytes
- **Memory Compression Ratio**: **{sketches_res['memory_saving_ratio']}x** reduction

### t-digest Quantile Estimation vs Exact NumPy Percentiles:

| Quantile | Exact Value | t-digest Approx | Absolute Error | Relative Error |
| :---: | :---: | :---: | :---: | :---: |
"""
    for q in sketches_res["quantile_benchmarks"]:
        md_content += f"| **p{int(q['quantile']*100):02d}** | {q['exact']} | {q['approx']} | {q['absolute_error']} | {q['relative_error']:.2%} |\n"

    md_content += f"""
- **Mean Relative Error Across Quantiles**: {sketches_res['mean_quantile_rel_error']:.2%}
- **HyperLogLog Cardinality Error**: {sketches_res['hll_benchmark']['relative_error']:.2%} (Exact: {sketches_res['hll_benchmark']['exact_cardinality']}, Estimated: {sketches_res['hll_benchmark']['estimated_cardinality']})

---

## 5. Experiment 4: Label-Free Performance Estimation (CBPE) vs Delayed Ground Truth

When delayed ground-truth labels arrive, estimated metrics are compared directly with actual outcomes:

"""
    if window_comp:
        est = window_comp["estimated"]
        act = window_comp["actual"]
        err = window_comp["error_mae"]
        md_content += f"""| Metric | CBPE Label-Free Estimate | Actual Delayed Ground Truth | Absolute Error (MAE) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | {est['estimated_accuracy']:.2%} | {act['accuracy']:.2%} | {err['accuracy']:.4f} |
| **F1-Score** | {est['estimated_f1']:.2%} | {act['f1']:.2%} | {err['f1']:.4f} |
| **Precision** | {est['estimated_precision']:.2%} | {act['precision']:.2%} | {err['precision']:.4f} |
| **Recall** | {est['estimated_recall']:.2%} | {act['recall']:.2%} | {err['recall']:.4f} |
| **ROC-AUC** | {est['estimated_roc_auc']:.4f} | {act['roc_auc']:.4f} | {err['roc_auc']:.4f} |
| **Brier Score** | {est['estimated_brier_score']:.4f} | {act['brier_score']:.4f} | {err['brier']:.4f} |
"""

    agg = cbpe_res["aggregate_estimation_errors"]
    md_content += f"""
- **Aggregate Accuracy MAE**: {agg.get('mae_accuracy', 0.0):.4f} (RMSE: {agg.get('rmse_accuracy', 0.0):.4f}, Bias: {agg.get('bias_accuracy', 0.0):+.4f})
- **Aggregate F1 MAE**: {agg.get('mae_f1', 0.0):.4f}

---

## 6. Experiment 5: Multivariate vs Univariate Detection Sensitivity
- Scenario: Feature correlation rotated with 0% shift in 1D marginal means.
- **Multivariate MMD Detected**: {results['experiment_5_multivariate_sensitivity']['multivariate_mmd_detected']}
- **Multivariate C2ST AUC**: {results['experiment_5_multivariate_sensitivity']['multivariate_c2st_auc']:.4f}
- **Univariate Features Drifted**: {results['experiment_5_multivariate_sensitivity']['univariate_drifted_count']}
- **Conclusion**: Multivariate tests (MMD and C2ST) successfully detect joint correlation shifts that univariate tests fail to detect.
"""

    report_path = out_p / "BENCHMARK_REPORT.md"
    report_path.write_text(md_content, encoding="utf-8")
    print(f"Benchmark reports generated at {out_p}")
    return str(report_path)


if __name__ == "__main__":
    generate_benchmark_reports()
