"""
Unit tests for univariate and multivariate drift detectors:
- ADWIN
- Page-Hinkley
- Kolmogorov-Smirnov (KS) & Population Stability Index (PSI)
- Maximum Mean Discrepancy (MMD)
- Classifier Two-Sample Test (C2ST)
- Multiple testing corrections (Bonferroni & Benjamini-Hochberg)
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np

from ml_engine.drift_monitor.detectors.adwin import ADWIN
from ml_engine.drift_monitor.detectors.c2st import ClassifierTwoSampleTest
from ml_engine.drift_monitor.detectors.corrections import (
    benjamini_hochberg_correction,
    bonferroni_correction,
)
from ml_engine.drift_monitor.detectors.distribution_tests import (
    calculate_psi,
    kolmogorov_smirnov_test,
)
from ml_engine.drift_monitor.detectors.mmd import MaximumMeanDiscrepancy
from ml_engine.drift_monitor.detectors.page_hinkley import PageHinkley


class TestDriftDetectors(unittest.TestCase):
    """Test suite for statistical drift detection algorithms."""

    def test_adwin_stationary_vs_drift(self):
        """ADWIN should remain quiet on stationary data and trigger on step shift."""
        rng = np.random.default_rng(42)
        adwin = ADWIN(delta=0.002)

        # 1. Stationary segment (mean 0.0)
        stationary = rng.normal(0.0, 1.0, 500)
        alarms_stationary = 0
        for x in stationary:
            drift, _ = adwin.update(x)
            if drift:
                alarms_stationary += 1

        self.assertEqual(alarms_stationary, 0, "ADWIN generated false alarms on stationary data.")

        # 2. Shifted segment (mean 3.0)
        shifted = rng.normal(3.0, 1.0, 300)
        alarms_shifted = 0
        for x in shifted:
            drift, _ = adwin.update(x)
            if drift:
                alarms_shifted += 1

        self.assertGreater(alarms_shifted, 0, "ADWIN failed to detect abrupt mean shift.")

    def test_page_hinkley(self):
        """Page-Hinkley detects sustained mean shifts."""
        rng = np.random.default_rng(42)
        ph = PageHinkley(threshold=30.0, delta=0.05, alpha=0.999)

        # Stationary: mean 10.0
        for x in rng.normal(10.0, 1.0, 400):
            ph.update(x)

        self.assertFalse(ph.drift_detected)

        # Shift to mean 15.0
        detected = False
        for x in rng.normal(15.0, 1.0, 200):
            drift, _ = ph.update(x)
            if drift:
                detected = True
                break

        self.assertTrue(detected, "Page-Hinkley failed to detect sustained positive shift.")

    def test_ks_test_and_psi(self):
        """Validates Kolmogorov-Smirnov test and Population Stability Index."""
        rng = np.random.default_rng(42)
        ref = rng.normal(50.0, 10.0, 1000)
        cur_same = rng.normal(50.0, 10.0, 1000)
        cur_drift = rng.normal(60.0, 10.0, 1000)

        # In-distribution
        ks_same = kolmogorov_smirnov_test(ref, cur_same, alpha=0.05)
        psi_same = calculate_psi(ref, cur_same)
        self.assertFalse(ks_same.drift_detected)
        self.assertEqual(psi_same.status, "STABLE")

        # Out-of-distribution
        ks_drift = kolmogorov_smirnov_test(ref, cur_drift, alpha=0.05)
        psi_drift = calculate_psi(ref, cur_drift)
        self.assertTrue(ks_drift.drift_detected)
        self.assertIn(psi_drift.status, ["MODERATE", "SIGNIFICANT"])

    def test_mmd_multivariate(self):
        """Validates MMD hypothesis test with RBF kernel and permutation."""
        rng = np.random.default_rng(42)
        X_ref = rng.normal(0.0, 1.0, size=(150, 4))
        Y_same = rng.normal(0.0, 1.0, size=(150, 4))
        Y_drift = rng.normal(1.2, 1.0, size=(150, 4))

        mmd = MaximumMeanDiscrepancy(n_permutations=50, alpha=0.05, random_state=42)

        res_same = mmd.test(X_ref, Y_same)
        self.assertFalse(res_same.drift_detected)

        res_drift = mmd.test(X_ref, Y_drift)
        self.assertTrue(res_drift.drift_detected)
        self.assertLess(res_drift.p_value, 0.05)

    def test_c2st(self):
        """Validates Classifier Two-Sample Test."""
        rng = np.random.default_rng(42)
        X_ref = rng.normal(0.0, 1.0, size=(200, 3))
        Y_same = rng.normal(0.0, 1.0, size=(200, 3))
        Y_drift = rng.normal(1.5, 1.0, size=(200, 3))

        c2st = ClassifierTwoSampleTest(n_splits=3, auc_threshold=0.65, random_state=42)

        res_same = c2st.test(X_ref, Y_same)
        self.assertFalse(res_same.drift_detected)
        self.assertLess(res_same.roc_auc, 0.65)

        res_drift = c2st.test(X_ref, Y_drift)
        self.assertTrue(res_drift.drift_detected)
        self.assertGreaterEqual(res_drift.roc_auc, 0.65)

    def test_multiple_testing_corrections(self):
        """Validates Bonferroni and Benjamini-Hochberg FDR adjustments."""
        p_vals = {
            "f1": 0.001,
            "f2": 0.012,
            "f3": 0.045,
            "f4": 0.120,
            "f5": 0.500,
        }

        # Bonferroni: threshold = 0.05 / 5 = 0.01
        bonf = bonferroni_correction(p_vals, alpha=0.05)
        self.assertTrue(bonf["f1"]["is_significant"])
        self.assertFalse(bonf["f2"]["is_significant"])  # 0.012 > 0.010

        # Benjamini-Hochberg FDR: more powerful than Bonferroni
        bh = benjamini_hochberg_correction(p_vals, alpha=0.05)
        self.assertTrue(bh["f1"]["is_significant"])
        self.assertTrue(bh["f2"]["is_significant"])  # q-val should be < 0.05


if __name__ == "__main__":
    unittest.main()
