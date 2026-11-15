"""
Person 2 Standalone Verification and Demonstration Script.
Label-Free Streaming Drift Monitor: Backend & Systems Lead Deliverable.

Demonstrates:
1. FastAPI Application & Health Probe
2. Streaming Transaction Ingestion (Record & Batch)
3. Bounded-Memory Sketch Metrics & Compression Tracking
4. Real-time Stream Simulation & Drift Alert Escalation
5. Univariate (PSI, KS, FDR, t-digest) & Multivariate (MMD, C2ST) Queries
6. Label-Free Performance Estimation (CBPE)
7. Delayed Ground Truth Reconciliation & Error Auditing
8. Shadow Model Retraining Governance & Controlled Approval Token Promotion
9. Prometheus Exporter Endpoint (/metrics)
10. Web Dashboard Static Asset Verification (/dashboard)
"""

import json
import time
from fastapi.testclient import TestClient

from backend.app.main import app


def print_section(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def main():
    print_section("PERSON 2 DELIVERABLE: STREAMING BACKEND & SYSTEMS VERIFICATION")
    client = TestClient(app)

    # 1. Health Probe
    print("\n[1] Verifying System Health Probe (/health)...")
    res = client.get("/health")
    print(f"    Status Code: {res.status_code}")
    print(f"    Payload: {res.json()}")
    assert res.status_code == 200

    # 2. Synchronous Transaction Ingestion
    print("\n[2] Ingesting Single Transaction (/api/v1/stream/record)...")
    tx_payload = {
        "record_id": "tx_demo_001",
        "tx_amount": 85.50,
        "sender_tx_count_1h": 2.0,
        "sender_risk_score": 0.12,
        "location_entropy": 0.22,
        "dest_novelty_score": 0.35,
        "time_delta_last_tx": 400.0,
        "network_degree": 2.0,
        "device_type": "web_browser",
        "merchant_category": "groceries",
    }
    res = client.post("/api/v1/stream/record", json=tx_payload)
    print(f"    Status Code: {res.status_code}")
    record_resp = res.json()
    print(f"    Prediction: {record_resp['prediction']} | Calibrated Probability: {record_resp['probability']:.4f}")
    print(f"    Streaming Drift Detected: {record_resp['streaming_drift_detected']} | Alert Severity: {record_resp['alert_severity']}")
    assert res.status_code == 200

    # 3. Batch Ingestion
    print("\n[3] Ingesting Batch of 25 Streaming Transactions (/api/v1/stream/batch)...")
    batch = [
        {
            "record_id": f"tx_demo_batch_{i:03d}",
            "tx_amount": 60.0 + i * 2.0,
            "sender_tx_count_1h": 1.0,
            "sender_risk_score": 0.08,
            "location_entropy": 0.18,
            "dest_novelty_score": 0.25,
            "time_delta_last_tx": 300.0,
            "network_degree": 1.0,
            "device_type": "mobile_app",
            "merchant_category": "groceries",
        }
        for i in range(25)
    ]
    res = client.post("/api/v1/stream/batch", json={"records": batch})
    batch_resp = res.json()
    print(f"    Batch Processed: {batch_resp['processed_count']} records")
    print(f"    Batch Ingest Throughput: {batch_resp['throughput_rec_per_sec']:.1f} records/sec")
    print(f"    Latest Severity: {batch_resp['latest_severity']}")
    assert res.status_code == 200

    # 4. Stream Simulation: Abrupt Drift Wave
    print("\n[4] Injecting Abrupt Drift Stream Simulation (/api/v1/stream/simulate)...")
    res = client.post(
        "/api/v1/stream/simulate",
        json={"num_records": 100, "drift_type": "abrupt"},
    )
    sim_resp = res.json()
    print(f"    Scenario: {sim_resp['scenario']} | Records: {sim_resp['simulated_records']}")
    print(f"    Alarms Triggered: {sim_resp['alarms_triggered']} | Latest Severity: {sim_resp['latest_severity']}")
    print(f"    Total Processed Overall: {sim_resp['total_processed_overall']}")
    assert res.status_code == 200

    # 5. Full Drift Status & Memory Compression
    print("\n[5] Querying Full Drift Status & Bounded Sketches (/api/v1/drift/status)...")
    res = client.get("/api/v1/drift/status")
    status_resp = res.json()
    mem = status_resp["memory"]
    alert = status_resp["alert"]
    print(f"    Records Processed: {status_resp['records_processed']}")
    print(f"    Throughput: {status_resp['throughput_rec_per_sec']} rec/s | Uptime: {status_resp['uptime_seconds']}s")
    print(f"    Sketches Memory: {mem['sketches_total_bytes']} bytes (~{mem['sketches_total_bytes']/1024:.2f} KB)")
    print(f"    Raw Window Equivalent: {mem['raw_window_bytes']} bytes (~{mem['raw_window_bytes']/1024:.2f} KB)")
    print(f"    Memory Compression Ratio: {mem['memory_compression_ratio']}x")
    print(f"    Active Alert Severity: {alert['severity']} (Composite Score: {alert['composite_score']:.3f})")
    assert res.status_code == 200

    # 6. Univariate Sketch Features
    print("\n[6] Inspecting Univariate Feature Sketches & Tests (/api/v1/drift/univariate)...")
    res = client.get("/api/v1/drift/univariate")
    uni_features = res.json()
    print(f"    {'Feature Name':<22} {'PSI':<10} {'KS p-val':<12} {'Significant?':<14} {'Median':<10} {'p95':<10}")
    print("    " + "-" * 78)
    for feat in uni_features:
        sig_str = "YES (DRIFT)" if feat["is_significant"] else "NO (Stable)"
        print(f"    {feat['feature_name']:<22} {feat['psi']:<10.4f} {feat['ks_p_value']:<12.4f} {sig_str:<14} {feat['tdigest_median']:<10.2f} {feat['tdigest_p95']:<10.2f}")
    assert len(uni_features) > 0

    # 7. Multivariate Drift (MMD & C2ST)
    print("\n[7] Querying Multivariate Drift Tests (/api/v1/drift/multivariate)...")
    res = client.get("/api/v1/drift/multivariate")
    multi_resp = res.json()
    print(f"    Overall Multivariate Status: {multi_resp['status']}")
    if multi_resp.get("mmd"):
        mmd = multi_resp["mmd"]
        print(f"    MMD² Statistic: {mmd['mmd_squared']:.6f} | Permutation p-val: {mmd['p_value']:.4f} | Detected: {mmd['drift_detected']}")
    if multi_resp.get("c2st"):
        c2st = multi_resp["c2st"]
        print(f"    C2ST ROC-AUC: {c2st['roc_auc']:.4f} | p-val: {c2st['p_value']:.4f} | Detected: {c2st['drift_detected']}")
    assert res.status_code == 200

    # 8. Label-Free Performance Estimation (CBPE)
    print("\n[8] Querying CBPE Label-Free Performance Estimates (/api/v1/performance/estimated)...")
    res = client.get("/api/v1/performance/estimated")
    cbpe_resp = res.json()
    print(f"    Sample Size: {cbpe_resp['sample_size']}")
    print(f"    Estimated Accuracy: {cbpe_resp['estimated_accuracy']:.4f} (Baseline: {cbpe_resp['baseline_accuracy']:.4f})")
    print(f"    Estimated F1:       {cbpe_resp['estimated_f1']:.4f} (Baseline: {cbpe_resp['baseline_f1']:.4f})")
    print(f"    Estimated ROC-AUC:  {cbpe_resp['estimated_roc_auc']:.4f} (Baseline: {cbpe_resp['baseline_roc_auc']:.4f})")
    print(f"    Performance Degraded Flag: {cbpe_resp['performance_degraded']}")
    assert res.status_code == 200

    # 9. Delayed Ground Truth Reconciliation
    print("\n[9] Reconciling Delayed Ground Truth Labels (/api/v1/performance/labels)...")
    delayed_labels = [
        {"record_id": f"tx_demo_batch_{i:03d}", "is_fraud": 0}
        for i in range(15)
    ]
    res = client.post("/api/v1/performance/labels", json={"labels": delayed_labels})
    recon_resp = res.json()
    print(f"    Reconciled Count: {recon_resp['reconciled_count']} | Matched Prior Predictions: {recon_resp['matched_labels']}")
    print(f"    Within Tolerance: {recon_resp['within_tolerance']}")
    assert res.status_code == 200

    # 10. Shadow Candidate Retraining & Governance Promotion
    print("\n[10] Evaluating Shadow Candidate Model (/api/v1/retrain/evaluate-candidate)...")
    res = client.post("/api/v1/retrain/evaluate-candidate", json={"sample_size": 150})
    cand_resp = res.json()
    print(f"    Decision: {cand_resp['decision']}")
    print(f"    Justification: {cand_resp['justification']}")
    print(f"    Approval Token Issued: {cand_resp['approval_token'] is not None}")
    assert res.status_code == 200

    print("\n[10b] Testing Human Governance Promotion Sign-Off (/api/v1/retrain/approve)...")
    # Verify rejection on bogus token
    fake_res = client.post(
        "/api/v1/retrain/approve",
        json={"approval_token": "fake_unauthorized_token", "approved_by": "Attacker"},
    )
    print(f"    Bogus Token Rejection Code: {fake_res.status_code} (Expected: 400 Bad Request)")
    assert fake_res.status_code == 400

    # Verify authorized promotion
    from backend.app.services.drift_service import DriftService
    service = DriftService.get_instance()
    auth_token = f"auth_signoff_{int(time.time())}"
    service.db.log_retrain_audit(
        trigger_reason="Significant gradual drift detected in transaction quantiles",
        urgency="HIGH",
        active_f1=0.72,
        candidate_f1=0.84,
        f1_delta=0.12,
        auc_delta=0.07,
        decision="PROMOTE_TO_PRODUCTION",
        approval_token=auth_token,
        status="PENDING_APPROVAL",
    )
    approve_res = client.post(
        "/api/v1/retrain/approve",
        json={
            "approval_token": auth_token,
            "approved_by": "Laxmi Singla (Backend & Systems Lead)",
            "notes": "Verified shadow evaluation metrics outperform production active baseline.",
        },
    )
    print(f"    Authorized Promotion Code: {approve_res.status_code}")
    print(f"    Promotion Result: {approve_res.json()}")
    assert approve_res.status_code == 200
    assert approve_res.json()["status"] == "APPROVED"

    # 11. Prometheus Exporter
    print("\n[11] Scraping Prometheus Metrics Exporter (/metrics)...")
    res = client.get("/metrics")
    assert res.status_code == 200
    metric_sample = [line for line in res.text.splitlines() if line and not line.startswith("#")][:8]
    for line in metric_sample:
        print(f"    [Prometheus] {line}")

    # 12. Dashboard Static Files
    print("\n[12] Verifying Interactive Dashboard Assets (/dashboard)...")
    res = client.get("/dashboard")
    assert res.status_code == 200
    print(f"    Dashboard HTML: {len(res.text)} bytes served successfully")

    print_section("ALL PERSON 2 DELIVERABLES VERIFIED AND READY FOR SUBMISSION!")


if __name__ == "__main__":
    main()
