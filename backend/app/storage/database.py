"""
Thread-safe SQLite database manager for storing drift alarms,
alert logs, performance comparisons, and shadow-model governance audit trails.
"""

import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional


class DatabaseManager:
    """Manages SQLite connection and schemas for the drift backend."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            storage_dir = Path(__file__).parent
            storage_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(storage_dir / "drift_backend.db")
        else:
            self.db_path = db_path

        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        """Initializes database tables if they do not already exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # 1. Drift alarm events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drift_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_position INTEGER,
                timestamp REAL,
                severity TEXT,
                summary TEXT,
                affected_features TEXT,
                composite_score REAL,
                recommendation TEXT,
                created_at REAL
            )
        """)

        # 2. Performance comparisons table (CBPE vs Delayed Ground Truth)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                sample_size INTEGER,
                estimated_accuracy REAL,
                actual_accuracy REAL,
                accuracy_error REAL,
                estimated_f1 REAL,
                actual_f1 REAL,
                f1_error REAL,
                estimated_auc REAL,
                actual_auc REAL,
                auc_error REAL,
                created_at REAL
            )
        """)

        # 3. Retraining governance and shadow model audit log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retrain_audits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                trigger_reason TEXT,
                urgency TEXT,
                active_f1 REAL,
                candidate_f1 REAL,
                f1_delta REAL,
                auc_delta REAL,
                decision TEXT,
                approval_token TEXT,
                approved_by TEXT,
                status TEXT,
                created_at REAL
            )
        """)

        conn.commit()

    def log_drift_event(
        self,
        record_position: int,
        severity: str,
        summary: str,
        affected_features: List[str],
        composite_score: float = 0.0,
        recommendation: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> int:
        """Logs a detected drift event."""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = time.time()
        ts = timestamp or now

        cursor.execute(
            """
            INSERT INTO drift_events (
                record_position, timestamp, severity, summary,
                affected_features, composite_score, recommendation, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_position,
                ts,
                severity,
                summary,
                json.dumps(affected_features),
                composite_score,
                json.dumps(recommendation or {}),
                now,
            ),
        )
        conn.commit()
        return cursor.lastrowid

    def get_drift_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves historical drift alarm events."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM drift_events ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = cursor.fetchall()
        events = []
        for r in rows:
            events.append({
                "id": r["id"],
                "record_position": r["record_position"],
                "timestamp": r["timestamp"],
                "severity": r["severity"],
                "summary": r["summary"],
                "affected_features": json.loads(r["affected_features"]) if r["affected_features"] else [],
                "composite_score": r["composite_score"],
                "recommendation": json.loads(r["recommendation"]) if r["recommendation"] else {},
                "created_at": r["created_at"],
            })
        return events

    def log_performance_evaluation(
        self,
        sample_size: int,
        estimated_acc: Optional[float] = None,
        actual_acc: Optional[float] = None,
        estimated_f1: float = 0.0,
        actual_f1: float = 0.0,
        estimated_auc: float = 0.0,
        actual_auc: float = 0.0,
        estimated_accuracy: Optional[float] = None,
        actual_accuracy: Optional[float] = None,
        accuracy_error: Optional[float] = None,
        f1_error: Optional[float] = None,
        auc_error: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> int:
        """Logs a CBPE vs delayed ground truth reconciliation comparison."""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = time.time()
        ts = timestamp or now

        e_acc = estimated_acc if estimated_acc is not None else (estimated_accuracy if estimated_accuracy is not None else 0.0)
        a_acc = actual_acc if actual_acc is not None else (actual_accuracy if actual_accuracy is not None else 0.0)
        acc_err = accuracy_error if accuracy_error is not None else abs(e_acc - a_acc)
        f1_err = f1_error if f1_error is not None else abs(estimated_f1 - actual_f1)
        auc_err = auc_error if auc_error is not None else abs(estimated_auc - actual_auc)

        cursor.execute(
            """
            INSERT INTO performance_evaluations (
                timestamp, sample_size,
                estimated_accuracy, actual_accuracy, accuracy_error,
                estimated_f1, actual_f1, f1_error,
                estimated_auc, actual_auc, auc_error, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                sample_size,
                e_acc,
                a_acc,
                acc_err,
                estimated_f1,
                actual_f1,
                f1_err,
                estimated_auc,
                actual_auc,
                auc_err,
                now,
            ),
        )
        conn.commit()
        return cursor.lastrowid

    def get_performance_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieves recent performance reconciliation records."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM performance_evaluations ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]

    def log_retrain_audit(
        self,
        trigger_reason: str,
        urgency: str,
        active_f1: float,
        candidate_f1: float,
        f1_delta: float,
        auc_delta: float,
        decision: str,
        approval_token: Optional[str] = None,
        status: str = "PENDING_APPROVAL",
    ) -> int:
        """Logs shadow model candidate evaluation."""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = time.time()

        cursor.execute(
            """
            INSERT INTO retrain_audits (
                timestamp, trigger_reason, urgency,
                active_f1, candidate_f1, f1_delta, auc_delta,
                decision, approval_token, approved_by, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                trigger_reason,
                urgency,
                active_f1,
                candidate_f1,
                f1_delta,
                auc_delta,
                decision,
                approval_token,
                None,
                status,
                now,
            ),
        )
        conn.commit()
        return cursor.lastrowid

    def get_performance_evaluations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent performance reconciliation records."""
        return self.get_performance_history(limit=limit)

    def get_retrain_audits(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent retrain governance audits."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM retrain_audits ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]

    def get_retrain_audit_by_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific retrain audit by approval token."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM retrain_audits WHERE approval_token = ?", (token,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def update_retrain_approval(
        self, token: str, approved_by: str = "MLOps Lead", status: str = "APPROVED"
    ) -> bool:
        """Updates approval status and approver for a token."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE retrain_audits
            SET status = ?, approved_by = ?
            WHERE approval_token = ?
            """,
            (status, approved_by, token),
        )
        conn.commit()
        return cursor.rowcount > 0

    def approve_token(self, token: str, approved_by: str = "MLOps Lead") -> bool:
        """Approves a candidate shadow model with its approval token."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE retrain_audits
            SET status = 'APPROVED', approved_by = ?
            WHERE approval_token = ? AND status = 'PENDING_APPROVAL'
            """,
            (approved_by, token),
        )
        conn.commit()
        return cursor.rowcount > 0


db = DatabaseManager()
