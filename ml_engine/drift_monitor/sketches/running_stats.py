"""
Welford's Algorithm for streaming online mean, variance, and running statistics.
Provides O(1) time and O(1) memory statistics tracking per numerical stream.
"""

import math
from typing import Any, Dict, Optional


class RunningStats:
    """
    Online accumulator using Welford's algorithm to compute:
    - Count
    - Running Mean
    - Running Variance and Standard Deviation (unbiased sample variance)
    - Minimum and Maximum
    - Running Sum
    """

    def __init__(self, name: str = "signal"):
        self.name = name
        self.count: int = 0
        self.mean: float = 0.0
        self._m2: float = 0.0
        self.min_val: float = float("inf")
        self.max_val: float = float("-inf")
        self.total_sum: float = 0.0

    def update(self, val: float) -> None:
        """Updates statistics with a single incoming value in O(1)."""
        if math.isnan(val) or math.isinf(val):
            return

        self.count += 1
        self.total_sum += val
        delta = val - self.mean
        self.mean += delta / self.count
        delta2 = val - self.mean
        self._m2 += delta * delta2

        if val < self.min_val:
            self.min_val = val
        if val > self.max_val:
            self.max_val = val

    @property
    def variance(self) -> float:
        """Sample variance (n - 1 degrees of freedom)."""
        if self.count < 2:
            return 0.0
        return self._m2 / (self.count - 1)

    @property
    def std_dev(self) -> float:
        """Sample standard deviation."""
        return math.sqrt(max(0.0, self.variance))

    def merge(self, other: "RunningStats") -> "RunningStats":
        """Combines two accumulators using Chan et al.'s parallel algorithm."""
        if other.count == 0:
            return self
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean
            self._m2 = other._m2
            self.min_val = other.min_val
            self.max_val = other.max_val
            self.total_sum = other.total_sum
            return self

        new_count = self.count + other.count
        delta = other.mean - self.mean
        new_mean = self.mean + delta * (other.count / new_count)
        new_m2 = self._m2 + other._m2 + (delta ** 2) * (self.count * other.count / new_count)

        self.count = new_count
        self.mean = new_mean
        self._m2 = new_m2
        self.min_val = min(self.min_val, other.min_val)
        self.max_val = max(self.max_val, other.max_val)
        self.total_sum += other.total_sum
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Returns statistical summary as a dictionary."""
        return {
            "name": self.name,
            "count": self.count,
            "mean": round(self.mean, 6) if self.count > 0 else 0.0,
            "variance": round(self.variance, 6) if self.count > 1 else 0.0,
            "std_dev": round(self.std_dev, 6) if self.count > 1 else 0.0,
            "min": round(self.min_val, 6) if self.count > 0 else 0.0,
            "max": round(self.max_val, 6) if self.count > 0 else 0.0,
            "sum": round(self.total_sum, 6) if self.count > 0 else 0.0,
        }
