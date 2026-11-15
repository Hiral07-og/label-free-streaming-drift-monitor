"""
Unit Tests for Alerts & Shadow Retraining Governance.
Person 2 (Backend & Systems Lead)
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi.testclient import TestClient
from backend.app.main import app


class TestAlertsAndRetraining(unittest.TestCase):
    """Test suite for alerts, shadow evaluation, and promotion governance."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_get_active_alert(self):
        """Verify active alert state retrieval."""
        res = self.client.get("/api/v1/alerts/active")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("severity", data)
        self.assertIn("composite_score", data)
        self.assertIn("summary_message", data)

    def test_02_get_retrain_recommendation(self):
        """Verify retraining recommendation response."""
        res = self.client.get("/api/v1/alerts/recommendation")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("should_retrain", data)
        self.assertIn("urgency", data)
        self.assertIn("recommended_action", data)

    def test_03_evaluate_candidate_shadow_model(self):
        """Verify shadow candidate model training and benchmark."""
        res = self.client.post(
            "/api/v1/retrain/evaluate-candidate",
            json={"sample_size": 150},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("active_metrics", data)
        self.assertIn("candidate_metrics", data)
        self.assertIn("decision", data)

        # Test approving with a unique valid token
        import uuid
        from backend.app.services.drift_service import DriftService
        service = DriftService.get_instance()
        unique_token = f"valid_test_token_{uuid.uuid4().hex[:8]}"
        service.db.log_retrain_audit(
            trigger_reason="Test trigger",
            urgency="HIGH",
            active_f1=0.75,
            candidate_f1=0.85,
            f1_delta=0.10,
            auc_delta=0.08,
            decision="PROMOTE_TO_PRODUCTION",
            approval_token=unique_token,
            status="PENDING_APPROVAL",
        )
        approve_res = self.client.post(
            "/api/v1/retrain/approve",
            json={
                "approval_token": unique_token,
                "approved_by": "Laxmi Singla (Backend Lead)",
                "notes": "Verified shadow test improvement",
            },
        )
        self.assertEqual(approve_res.status_code, 200)
        app_data = approve_res.json()
        self.assertEqual(app_data["status"], "APPROVED")

    def test_04_reject_invalid_token(self):
        """Verify promotion fails with HTTP 400 for bogus token."""
        res = self.client.post(
            "/api/v1/retrain/approve",
            json={
                "approval_token": "fake_invalid_token_12345",
                "approved_by": "Unauthorized User",
            },
        )
        self.assertEqual(res.status_code, 400)


if __name__ == "__main__":
    unittest.main()
