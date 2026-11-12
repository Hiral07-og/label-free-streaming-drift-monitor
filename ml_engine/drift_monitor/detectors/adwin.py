"""
Adaptive Windowing (ADWIN) algorithm for streaming concept/data drift detection.
Automatically expands window when data is stationary and shrinks it when drift occurs,
providing rigorous statistical bounds based on Hoeffding's inequality.
Reference: Bifet & Gavalda (SDM 2007)
"""

import math
from typing import Any, Dict, List, Optional, Tuple


class ADWINBucket:
    """Bucket storing count, sum, and squared sum of elements of size 2^k."""
    __slots__ = ("size", "total", "sq_total")

    def __init__(self, size: int, total: float, sq_total: float = 0.0):
        self.size = int(size)
        self.total = float(total)
        self.sq_total = float(sq_total)


class ADWIN:
    """
    Adaptive Windowing drift detector.
    Parameters:
        delta: Confidence value in (0, 1), e.g. 0.002.
        max_buckets_per_size: Max number of buckets allowed per size class (M).
    """

    def __init__(self, delta: float = 0.002, max_buckets_per_size: int = 5, name: str = "adwin"):
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must be between 0 and 1 exclusive.")

        self.name = name
        self.delta = float(delta)
        self.M = max(2, int(max_buckets_per_size))

        # List of lists: bucket_levels[level] holds buckets of size 2^level
        self.bucket_levels: List[List[ADWINBucket]] = [[]]
        self.total_count: int = 0
        self.total_sum: float = 0.0
        self.total_sq_sum: float = 0.0
        self.min_val: float = float("inf")
        self.max_val: float = float("-inf")
        self.drift_detected: bool = False
        self.last_drift_pos: Optional[int] = None
        self.n_elements_seen: int = 0

    def update(self, val: float) -> Tuple[bool, float]:
        """
        Updates ADWIN with a new stream observation.
        Returns:
            (drift_detected: bool, current_window_mean: float)
        """
        if math.isnan(val) or math.isinf(val):
            return False, self.mean

        self.n_elements_seen += 1
        self.drift_detected = False

        if val < self.min_val:
            self.min_val = val
        if val > self.max_val:
            self.max_val = val

        # 1. Insert bucket of size 1 at level 0
        self._insert_element(val)

        # 2. Check for drift (cut in window)
        drift = self._check_cut()
        if drift:
            self.drift_detected = True
            self.last_drift_pos = self.n_elements_seen

        return self.drift_detected, self.mean

    def _insert_element(self, val: float) -> None:
        """Inserts a single element of size 1 and compresses levels."""
        self.bucket_levels[0].append(ADWINBucket(size=1, total=val, sq_total=val * val))
        self.total_count += 1
        self.total_sum += val
        self.total_sq_sum += val * val

        # Compress if buckets in level exceed M
        level = 0
        while level < len(self.bucket_levels):
            if len(self.bucket_levels[level]) > self.M:
                # Merge two oldest buckets into level + 1
                b1 = self.bucket_levels[level].pop(0)
                b2 = self.bucket_levels[level].pop(0)
                merged = ADWINBucket(
                    size=b1.size + b2.size,
                    total=b1.total + b2.total,
                    sq_total=b1.sq_total + b2.sq_total,
                )

                if level + 1 >= len(self.bucket_levels):
                    self.bucket_levels.append([])
                self.bucket_levels[level + 1].append(merged)
            level += 1

    def _check_cut(self) -> bool:
        """
        Iterates over potential split points between older (W0) and newer (W1)
        portions of the window to detect significant differences in mean.
        Uses ADWIN2 variance-scaled Hoeffding/Bernstein bound.
        """
        if self.total_count < 20:
            return False

        has_cut = False

        # Flatten all buckets ordered from oldest to newest
        all_buckets: List[ADWINBucket] = []
        for level_buckets in reversed(self.bucket_levels):
            all_buckets.extend(level_buckets)

        n0 = 0
        s0 = 0.0
        n = self.total_count
        s = self.total_sum

        # Calculate current window variance
        cur_mean = s / n
        variance = max(1e-4, (self.total_sq_sum / n) - (cur_mean ** 2))
        stdev = math.sqrt(variance)
        # Dynamic signal scale
        r = max(1.0, 4.0 * stdev, self.max_val - self.min_val)

        # Check cuts from oldest to newest
        for b in all_buckets[:-1]:
            n0 += b.size
            s0 += b.total
            n1 = n - n0
            s1 = s - s0

            if n0 < 10 or n1 < 10:
                continue

            u0 = s0 / n0
            u1 = s1 / n1
            diff = abs(u0 - u1)

            # Harmonic mean of subwindow lengths
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            # Hoeffding cut threshold scaled by data range R
            delta_prime = self.delta / max(1.0, math.log(n))
            eps_cut = r * math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 / max(1e-12, delta_prime)))

            if diff >= eps_cut:
                has_cut = True
                # Drop oldest buckets from W0
                self._delete_oldest_bucket()
                break

        return has_cut

    def _delete_oldest_bucket(self) -> None:
        """Removes the oldest bucket from the highest non-empty level."""
        for level in range(len(self.bucket_levels) - 1, -1, -1):
            if self.bucket_levels[level]:
                b = self.bucket_levels[level].pop(0)
                self.total_count -= b.size
                self.total_sum -= b.total
                self.total_sq_sum -= b.sq_total
                break

    @property
    def mean(self) -> float:
        """Mean of elements in the current adaptive window."""
        if self.total_count == 0:
            return 0.0
        return self.total_sum / self.total_count

    @property
    def width(self) -> int:
        """Current number of items retained in the window."""
        return self.total_count

    def reset(self) -> None:
        """Resets the detector state."""
        self.bucket_levels = [[]]
        self.total_count = 0
        self.total_sum = 0.0
        self.drift_detected = False
        self.n_elements_seen = 0

    def to_dict(self) -> Dict[str, Any]:
        """Summary of ADWIN state."""
        return {
            "name": self.name,
            "window_width": self.width,
            "window_mean": round(self.mean, 6),
            "delta": self.delta,
            "total_seen": self.n_elements_seen,
            "drift_detected": self.drift_detected,
            "last_drift_pos": self.last_drift_pos,
        }
