"""
t-digest data structure for accurate streaming quantile estimation.
Maintains bounded memory while providing high-resolution quantile approximations,
particularly at extreme tails (e.g. 1st, 5th, 95th, 99th percentiles).
"""

import math
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class Centroid:
    """A weighted cluster centroid with mean and weight (count)."""
    __slots__ = ("mean", "weight")

    def __init__(self, mean: float, weight: float = 1.0):
        self.mean = float(mean)
        self.weight = float(weight)

    def __repr__(self) -> str:
        return f"Centroid(mean={self.mean:.4f}, weight={self.weight:.2f})"


class TDigest:
    """
    Streaming t-digest implementation with centroid clustering.
    Bounded-memory quantile estimation using scale function:
    k_1(q) = (delta / (2 * pi)) * asin(2 * q - 1)
    """

    def __init__(self, compression: float = 100.0):
        self.compression = max(10.0, float(compression))
        self.centroids: List[Centroid] = []
        self._buffer_values: List[float] = []
        self._buffer_weights: List[float] = []
        self.total_weight: float = 0.0
        self.min_val: float = float("inf")
        self.max_val: float = float("-inf")
        # Buffer flush threshold
        self.buffer_limit = int(5 * self.compression)

    def update(self, val: float, weight: float = 1.0) -> None:
        """Appends value and weight to the buffer. Compresses when buffer fills."""
        if math.isnan(val) or math.isinf(val) or weight <= 0.0:
            return

        self._buffer_values.append(float(val))
        self._buffer_weights.append(float(weight))
        self.total_weight += weight

        if val < self.min_val:
            self.min_val = val
        if val > self.max_val:
            self.max_val = val

        if len(self._buffer_values) >= self.buffer_limit:
            self.compress()

    def compress(self) -> None:
        """Merges buffered data into existing centroids respecting scale bounds."""
        if not self._buffer_values and not self.centroids:
            return

        # Combine existing centroids and buffered items
        all_means = [c.mean for c in self.centroids] + self._buffer_values
        all_weights = [c.weight for c in self.centroids] + self._buffer_weights

        self._buffer_values = []
        self._buffer_weights = []

        # Sort combined points by mean
        order = np.argsort(all_means)
        sorted_means = np.array(all_means, dtype=np.float64)[order]
        sorted_weights = np.array(all_weights, dtype=np.float64)[order]

        n_points = len(sorted_means)
        total_w = np.sum(sorted_weights)
        self.total_weight = float(total_w)

        if n_points == 0 or total_w == 0.0:
            self.centroids = []
            return

        new_centroids: List[Centroid] = []
        current_mean = sorted_means[0]
        current_weight = sorted_weights[0]
        cumulative_w = 0.0

        for i in range(1, n_points):
            w = sorted_weights[i]
            m = sorted_means[i]

            # Current quantile fraction
            q = (cumulative_w + current_weight / 2.0) / total_w
            # Scale function maximum weight limit at quantile q
            # max_weight = 4 * N * q * (1 - q) / delta
            max_w = max(1.0, 4.0 * total_w * q * (1.0 - q) / self.compression)

            if current_weight + w <= max_w:
                # Merge into current centroid
                current_mean = (current_mean * current_weight + m * w) / (current_weight + w)
                current_weight += w
            else:
                # Close current centroid and start new one
                new_centroids.append(Centroid(current_mean, current_weight))
                cumulative_w += current_weight
                current_mean = m
                current_weight = w

        new_centroids.append(Centroid(current_mean, current_weight))
        self.centroids = new_centroids

    def quantile(self, q: float) -> float:
        """
        Estimates the value corresponding to quantile q in [0, 1].
        Uses linear interpolation between adjacent centroids.
        """
        if self._buffer_values:
            self.compress()

        if not self.centroids:
            return 0.0
        if q <= 0.0:
            return self.min_val
        if q >= 1.0:
            return self.max_val
        if len(self.centroids) == 1:
            return self.centroids[0].mean

        target_w = q * self.total_weight
        cumulative_w = 0.0

        for i, c in enumerate(self.centroids):
            # Centroid midpoint weight
            half_w = c.weight / 2.0
            mid_w = cumulative_w + half_w

            if target_w < mid_w:
                if i == 0:
                    # Interpolate between min_val and first centroid mean
                    if mid_w <= 0.0:
                        return self.min_val
                    t = target_w / mid_w
                    return self.min_val + t * (c.mean - self.min_val)
                else:
                    prev = self.centroids[i - 1]
                    prev_mid_w = cumulative_w - prev.weight / 2.0
                    t = (target_w - prev_mid_w) / (mid_w - prev_mid_w)
                    return prev.mean + t * (c.mean - prev.mean)

            cumulative_w += c.weight

        # Between last centroid and max_val
        last = self.centroids[-1]
        last_mid_w = self.total_weight - last.weight / 2.0
        if self.total_weight <= last_mid_w:
            return self.max_val
        t = (target_w - last_mid_w) / (self.total_weight - last_mid_w)
        return last.mean + t * (self.max_val - last.mean)

    def cdf(self, val: float) -> float:
        """
        Estimates the cumulative distribution function P(X <= val) in [0, 1].
        """
        if self._buffer_values:
            self.compress()

        if not self.centroids or self.total_weight == 0.0:
            return 0.0
        if val <= self.min_val:
            return 0.0
        if val >= self.max_val:
            return 1.0
        if len(self.centroids) == 1:
            return 0.5

        cumulative_w = 0.0
        for i, c in enumerate(self.centroids):
            if val < c.mean:
                if i == 0:
                    # Below first centroid
                    t = (val - self.min_val) / max(1e-9, (c.mean - self.min_val))
                    return max(0.0, min(1.0, (t * (c.weight / 2.0)) / self.total_weight))
                else:
                    prev = self.centroids[i - 1]
                    t = (val - prev.mean) / max(1e-9, (c.mean - prev.mean))
                    prev_mid_w = cumulative_w - prev.weight / 2.0
                    cur_mid_w = cumulative_w + c.weight / 2.0
                    interp_w = prev_mid_w + t * (cur_mid_w - prev_mid_w)
                    return max(0.0, min(1.0, interp_w / self.total_weight))

            cumulative_w += c.weight

        return 1.0

    def merge(self, other: "TDigest") -> "TDigest":
        """Merges another t-digest into this one."""
        self.compress()
        other.compress()

        for c in other.centroids:
            self._buffer_values.append(c.mean)
            self._buffer_weights.append(c.weight)
            self.total_weight += c.weight

        self.min_val = min(self.min_val, other.min_val)
        self.max_val = max(self.max_val, other.max_val)
        self.compress()
        return self

    def memory_bytes(self) -> int:
        """Calculates approximate memory footprint in bytes."""
        base_size = sys.getsizeof(self)
        centroids_size = sys.getsizeof(self.centroids) + len(self.centroids) * 48
        buf_size = sys.getsizeof(self._buffer_values) + sys.getsizeof(self._buffer_weights)
        return base_size + centroids_size + buf_size

    def to_dict(self) -> Dict[str, Any]:
        """Summary of quantiles and memory footprint."""
        return {
            "compression": self.compression,
            "n_centroids": len(self.centroids),
            "total_weight": round(self.total_weight, 2),
            "p10": round(self.quantile(0.10), 4),
            "p25": round(self.quantile(0.25), 4),
            "p50": round(self.quantile(0.50), 4),
            "p75": round(self.quantile(0.75), 4),
            "p90": round(self.quantile(0.90), 4),
            "p95": round(self.quantile(0.95), 4),
            "p99": round(self.quantile(0.99), 4),
            "min": round(self.min_val, 4) if self.min_val != float("inf") else 0.0,
            "max": round(self.max_val, 4) if self.max_val != float("-inf") else 0.0,
            "memory_bytes": self.memory_bytes(),
        }
