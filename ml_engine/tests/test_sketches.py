"""
Unit tests for bounded-memory sketches:
- RunningStats (Welford's algorithm)
- TDigest (Quantile sketch)
- CountMinSketch (Frequency sketch with conservative update)
- HyperLogLog (Distinct cardinality sketch)
- ReservoirSampler (Vitter's algorithm R)
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd

from ml_engine.drift_monitor.sketches.count_min_sketch import CountMinSketch
from ml_engine.drift_monitor.sketches.hyperloglog import HyperLogLog
from ml_engine.drift_monitor.sketches.reservoir import ReservoirSampler
from ml_engine.drift_monitor.sketches.running_stats import RunningStats
from ml_engine.drift_monitor.sketches.tdigest import TDigest


class TestSketches(unittest.TestCase):
    """Test suite for streaming sketch data structures."""

    def test_running_stats_welford(self):
        """Validates that Welford's algorithm matches NumPy exact statistics."""
        data = np.random.normal(loc=50.0, scale=12.5, size=1000)
        stats = RunningStats(name="test_signal")

        for x in data:
            stats.update(x)

        self.assertEqual(stats.count, 1000)
        self.assertAlmostEqual(stats.mean, float(np.mean(data)), places=5)
        self.assertAlmostEqual(stats.variance, float(np.var(data, ddof=1)), places=4)
        self.assertAlmostEqual(stats.std_dev, float(np.std(data, ddof=1)), places=4)
        self.assertAlmostEqual(stats.min_val, float(np.min(data)), places=5)
        self.assertAlmostEqual(stats.max_val, float(np.max(data)), places=5)

    def test_running_stats_merge(self):
        """Validates Chan et al. parallel accumulator merge."""
        d1 = np.random.normal(10.0, 2.0, 500)
        d2 = np.random.normal(20.0, 4.0, 500)
        combined = np.concatenate([d1, d2])

        s1 = RunningStats("s1")
        for x in d1:
            s1.update(x)

        s2 = RunningStats("s2")
        for x in d2:
            s2.update(x)

        s1.merge(s2)
        self.assertEqual(s1.count, 1000)
        self.assertAlmostEqual(s1.mean, float(np.mean(combined)), places=4)
        self.assertAlmostEqual(s1.variance, float(np.var(combined, ddof=1)), places=3)

    def test_tdigest_quantiles(self):
        """Validates that t-digest quantiles closely match exact percentiles."""
        rng = np.random.default_rng(42)
        data = rng.lognormal(mean=3.0, sigma=0.8, size=2000)

        td = TDigest(compression=100.0)
        for x in data:
            td.update(x)
        td.compress()

        for q in [0.05, 0.25, 0.50, 0.75, 0.95]:
            exact = float(np.percentile(data, q * 100))
            est = td.quantile(q)
            rel_err = abs(est - exact) / exact
            self.assertLess(rel_err, 0.08, f"Quantile {q} relative error too large: {rel_err:.4f}")

    def test_tdigest_merge(self):
        """Validates merging two t-digests."""
        rng = np.random.default_rng(42)
        d1 = rng.normal(0, 1, 500)
        d2 = rng.normal(5, 1, 500)
        combined = np.concatenate([d1, d2])

        td1 = TDigest(compression=100.0)
        for x in d1:
            td1.update(x)

        td2 = TDigest(compression=100.0)
        for x in d2:
            td2.update(x)

        td1.merge(td2)
        self.assertEqual(td1.total_weight, 1000)
        median_est = td1.quantile(0.5)
        median_exact = float(np.median(combined))
        self.assertLess(abs(median_est - median_exact), 0.5)

    def test_count_min_sketch(self):
        """Validates Count-Min Sketch frequency lower-bound and error guarantees."""
        cms = CountMinSketch(epsilon=0.005, delta=0.01, conservative_update=True)
        items = ["apple"] * 200 + ["banana"] * 50 + ["cherry"] * 5

        for item in items:
            cms.update(item)

        self.assertGreaterEqual(cms.estimate_frequency("apple"), 200)
        self.assertGreaterEqual(cms.estimate_frequency("banana"), 50)
        self.assertGreaterEqual(cms.estimate_frequency("cherry"), 5)
        self.assertEqual(cms.estimate_frequency("dragonfruit"), 0)

        # Overestimation should be very small under conservative update
        self.assertLessEqual(cms.estimate_frequency("apple"), 205)

    def test_hyperloglog(self):
        """Validates HyperLogLog cardinality estimation accuracy."""
        hll = HyperLogLog(precision_bits=10)
        # 1000 distinct items repeated multiple times
        distinct_elements = [f"entity_{i}" for i in range(1000)]
        stream = distinct_elements * 3

        for it in stream:
            hll.add(it)

        est = hll.estimate_cardinality()
        # Relative error should be within ~3 * standard_error (~10%)
        rel_err = abs(est - 1000.0) / 1000.0
        self.assertLess(rel_err, 0.12, f"HLL relative error {rel_err:.4f} exceeds 12%")

    def test_reservoir_sampler(self):
        """Validates that reservoir maintains bounded capacity K."""
        sampler = ReservoirSampler(capacity=100, random_state=42)
        for i in range(500):
            sampler.add({"val": i})

        self.assertTrue(sampler.is_full())
        self.assertEqual(len(sampler.get_sample()), 100)
        self.assertEqual(sampler.total_seen, 500)


if __name__ == "__main__":
    unittest.main()
