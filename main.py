"""
Unified Master Entrypoint for the Label-Free Streaming Drift Monitor.

Combines the ML & Statistical Monitoring Engine (`ml_engine/`) and the
Streaming Backend, FastAPI Server, Dashboard & Governance Layer (`backend/`)
into a single cohesive application.

Usage:
    python main.py serve [--host 0.0.0.0] [--port 8000] [--auto-stream] [--reload]
    python main.py demo
    python main.py benchmark
    python main.py test
"""

import argparse
import os
from pathlib import Path
import sys
import time
import unittest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def cmd_serve(args: argparse.Namespace) -> None:
    """Starts the unified FastAPI server, Streaming Consumer, ML Engine, and Web Dashboard."""
    import uvicorn

    if args.auto_stream:
        os.environ["AUTO_STREAM_ON_STARTUP"] = "true"
        os.environ["AUTO_STREAM_RATE"] = str(args.rate)
        os.environ["AUTO_STREAM_MODE"] = args.mode

    print("=" * 78)
    print(" Label-Free Streaming Drift Monitor — Unified Studio Server")
    print("=" * 78)
    print(f"  Dashboard UI     : http://localhost:{args.port}/dashboard")
    print(f"  Swagger API Docs : http://localhost:{args.port}/docs")
    print(f"  Prometheus       : http://localhost:{args.port}/metrics")
    print(f"  Auto-Stream Mode : {'ENABLED (' + args.mode + ' @ ' + str(args.rate) + ' tx/s)' if args.auto_stream else 'STANDBY (toggle in UI or API)'}")
    print("=" * 78)

    uvicorn.run(
        "backend.app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def cmd_benchmark(_args: argparse.Namespace) -> None:
    """Executes the full experimental benchmark suite and generates CSV/Markdown reports."""
    from ml_engine.drift_monitor.benchmarks.generate_report import (
        generate_benchmark_reports,
    )

    reports_dir = REPO_ROOT / "ml_engine" / "drift_monitor" / "benchmarks" / "reports"
    print("=" * 78)
    print(" Running Unified Experimental Benchmark Suite")
    print("=" * 78)
    report_path = generate_benchmark_reports(output_dir=str(reports_dir))
    print(f"\nBenchmarks complete! Reports saved to: {reports_dir}")
    print(f"Markdown report: {report_path}")


def cmd_test(_args: argparse.Namespace) -> None:
    """Discovers and runs all unit, backend, and end-to-end integration tests."""
    print("=" * 78)
    print(" Running Unified Test Suite (ML Engine + Streaming Backend + E2E Integration)")
    print("=" * 78)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_dirs = [
        REPO_ROOT / "ml_engine" / "tests",
        REPO_ROOT / "backend" / "tests",
        REPO_ROOT / "tests",
    ]

    for tdir in test_dirs:
        if tdir.exists():
            discovered = loader.discover(start_dir=str(tdir), pattern="test_*.py")
            suite.addTests(discovered)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)


def cmd_demo(_args: argparse.Namespace) -> None:
    """
    Runs a single unified end-to-end demonstration connecting:
    1. Reference Calibration & Bounded-Memory Sketches (ML Engine)
    2. Live Producer -> Consumer Streaming Ingestion (Backend)
    3. Controlled Drift Injection (Stationary -> Abrupt -> Recurring -> Feature-Specific)
    4. Univariate (PSI, KS, FDR, t-digest) & Multivariate (MMD, C2ST) Detection
    5. Label-Free CBPE & Delayed Ground Truth Reconciliation
    6. Shadow Model Evaluation & Governed Approval Token Promotion
    7. SQLite Audit Trail, Experimental Benchmarks API, Prometheus & Dashboard
    """
    from fastapi.testclient import TestClient
    from backend.app.main import app
    from backend.app.services.drift_service import DriftService

    print("\n" + "=" * 80)
    print(" LABEL-FREE STREAMING DRIFT MONITOR — UNIFIED END-TO-END SYSTEM DEMO")
    print("=" * 80)

    with TestClient(app) as client:
        # 1. Health & Baseline Status
        health = client.get("/health").json()
        print(f"\n[1/8] System Health: {health['status']} ({health['service']} v{health['version']})")

        # 2. Continuous Producer -> Consumer Auto-Stream Test
        print("\n[2/8] Testing Live Producer -> Consumer Streaming Pipeline...")
        client.post("/api/v1/stream/auto-stream/start", json={"drift_mode": "none", "rate_per_sec": 20.0})
        time.sleep(0.6)
        pipe_status = client.post("/api/v1/stream/auto-stream/stop").json()
        time.sleep(0.2)
        pipe_status = client.get("/api/v1/stream/consumer-status").json()
        print(f"      Emitted: {pipe_status['auto_stream_emitted']} | Consumed via Worker: {pipe_status['consumer']['consumed_count']} | Errors: {pipe_status['consumer']['error_count']}")

        # 3. Stationary Baseline Batch + Abrupt & Recurring Drift Injection
        print("\n[3/8] Streaming Controlled Scenarios (Stationary -> Abrupt -> Recurring)...")
        s_none = client.post("/api/v1/stream/simulate", json={"num_records": 50, "drift_type": "none"}).json()
        s_abrupt = client.post("/api/v1/stream/simulate", json={"num_records": 100, "drift_type": "abrupt"}).json()
        s_recur = client.post("/api/v1/stream/simulate", json={"num_records": 60, "drift_type": "recurring"}).json()
        print(f"      Stationary : {s_none['simulated_records']} tx -> Severity: {s_none['latest_severity']}")
        print(f"      Abrupt     : {s_abrupt['simulated_records']} tx -> Alarms: {s_abrupt['alarms_triggered']}, Severity: {s_abrupt['latest_severity']}")
        print(f"      Recurring  : {s_recur['simulated_records']} tx -> Alarms: {s_recur['alarms_triggered']}, Severity: {s_recur['latest_severity']}")

        # 4. Inspect All 5 Bounded-Memory Sketches & Univariate/Multivariate Drift
        print("\n[4/8] Querying Bounded Sketches, Univariate Tests & Multivariate Drift...")
        status = client.get("/api/v1/drift/status").json()
        uni = client.get("/api/v1/drift/univariate").json()
        multi = client.get("/api/v1/drift/multivariate").json()
        drifted_feats = [f["feature_name"] for f in uni if f["is_significant"]]
        print(f"      Total Processed : {status['records_processed']} records @ {status['throughput_rec_per_sec']} rec/s")
        print(f"      Alert Status    : {status['alert']['severity']} (Composite: {status['alert']['composite_score']:.3f})")
        print(f"      Drifted Features: {drifted_feats}")
        print(f"      Categorical HLL : { {k: round(v['estimated_cardinality'], 1) for k, v in status['categorical_sketches'].items()} }")
        if multi.get("mmd") and multi.get("c2st"):
            print(f"      Multivariate    : MMD²={multi['mmd']['mmd_squared']:.5f} (p={multi['mmd']['p_value']:.4f}) | C2ST AUC={multi['c2st']['roc_auc']:.4f}")

        # 5. Label-Free CBPE & Delayed Ground Truth Reconciliation
        print("\n[5/8] Label-Free Performance Estimation (CBPE) & Delayed Reconciliation...")
        cbpe = client.get("/api/v1/performance/estimated").json()
        print(f"      CBPE Estimated  : Acc={cbpe['estimated_accuracy']:.3f} | F1={cbpe['estimated_f1']:.3f} | AUC={cbpe['estimated_roc_auc']:.3f} (Degraded: {cbpe['performance_degraded']})")
        # Send batch & reconcile their delayed labels
        batch = [{"record_id": f"tx_e2e_{i}", "tx_amount": 80.0 + i, "sender_risk_score": 0.15} for i in range(10)]
        client.post("/api/v1/stream/batch", json={"records": batch})
        recon = client.post(
            "/api/v1/performance/labels",
            json={"labels": [{"record_id": f"tx_e2e_{i}", "is_fraud": 0} for i in range(10)]},
        ).json()
        print(f"      Reconciled      : {recon['matched_labels']}/{recon['reconciled_count']} delayed ground-truth labels matched")

        # 6. Shadow Model Evaluation & Governed Token Promotion
        print("\n[6/8] Shadow Retraining Evaluation & Governed Promotion Sign-Off...")
        cand = client.post("/api/v1/retrain/evaluate-candidate", json={"sample_size": 150}).json()
        print(f"      Shadow Decision : {cand['decision']} (Active F1={cand['active_metrics'].get('f1', 0):.3f}, Candidate F1={cand['candidate_metrics'].get('f1', 0):.3f})")
        service = DriftService.get_instance()
        token = f"unified_demo_token_{int(time.time())}"
        service.db.log_retrain_audit(
            trigger_reason="Unified E2E verification promotion",
            urgency="HIGH",
            active_f1=0.74,
            candidate_f1=0.86,
            f1_delta=0.12,
            auc_delta=0.06,
            decision="PROMOTE_TO_PRODUCTION",
            approval_token=token,
            status="PENDING_APPROVAL",
        )
        promo = client.post(
            "/api/v1/retrain/approve",
            json={"approval_token": token, "approved_by": "MLOps Governance Lead"},
        ).json()
        print(f"      Governance Gate : {promo['status']} -> {promo['message']}")

        # 7. Connected ML Engine Benchmarks API
        print("\n[7/8] Querying Connected Experimental Benchmarks API (/api/v1/benchmarks/results)...")
        bench = client.get("/api/v1/benchmarks/results").json()
        print(f"      Benchmark Status: {bench['status']} | Scenarios: {len(bench['detection_delay_benchmarks'])} | Quantiles Checked: {len(bench['tdigest_quantile_accuracy'])}")

        # 8. Prometheus Metrics & Dashboard Verification
        print("\n[8/8] Verifying Prometheus Metrics & Interactive Dashboard...")
        prom = client.get("/metrics")
        dash = client.get("/dashboard")
        assert prom.status_code == 200 and "drift_records_processed_total" in prom.text
        assert dash.status_code == 200 and "Benchmarks & Eval" in dash.text
        print(f"      Prometheus      : HTTP 200 ({len(prom.text.splitlines())} metric lines exported)")
        print(f"      Web Dashboard   : HTTP 200 ({len(dash.text)} bytes served)")

    print("\n" + "=" * 80)
    print(" UNIFIED PROJECT VERIFICATION COMPLETE — ALL COMPONENTS CONNECTED!")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified CLI for the Label-Free Streaming Drift Monitor"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # serve
    p_serve = subparsers.add_parser("serve", help="Start FastAPI server, streaming consumer & web dashboard")
    p_serve.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p_serve.add_argument("--reload", action="store_true", help="Enable hot-reload for development")
    p_serve.add_argument("--auto-stream", action="store_true", help="Automatically start live streaming on startup")
    p_serve.add_argument("--rate", type=float, default=5.0, help="Auto-stream rate in tx/sec (default: 5.0)")
    p_serve.add_argument("--mode", default="none", choices=["none", "abrupt", "gradual"], help="Initial auto-stream drift mode")
    p_serve.set_defaults(func=cmd_serve)

    # demo
    p_demo = subparsers.add_parser("demo", help="Run unified end-to-end demonstration")
    p_demo.set_defaults(func=cmd_demo)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run full controlled drift benchmark suite")
    p_bench.set_defaults(func=cmd_benchmark)

    # test
    p_test = subparsers.add_parser("test", help="Run all ML engine, backend, and E2E integration tests")
    p_test.set_defaults(func=cmd_test)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
