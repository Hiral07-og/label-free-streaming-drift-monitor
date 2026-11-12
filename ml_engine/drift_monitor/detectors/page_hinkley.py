"""
Page-Hinkley (PH) Test for streaming sequential change detection.
Monitors cumulative deviations of a continuous numerical signal from its running mean
and raises alarms when deviations exceed a defined cumulative threshold.
"""

import math
from typing import Any, Dict, Optional, Tuple


class PageHinkley:
    """
    Page-Hinkley sequential change detector.
    Parameters:
        threshold (lambda): Cumulative deviation threshold triggering drift alarm.
        delta: Minimum allowable deviation magnitude before accumulating.
        alpha: Forgetting factor for running mean (1.0 = standard sample mean, <1.0 = exponential decay).
        bidirectional: If True, detects both upward and downward mean shifts.
    """

    def __init__(
        self,
        threshold: float = 50.0,
        delta: float = 0.01,
        alpha: float = 0.999,
        bidirectional: bool = True,
        name: str = "page_hinkley",
    ):
        self.name = name
        self.threshold = float(threshold)
        self.delta = float(delta)
        self.alpha = float(alpha)
        self.bidirectional = bidirectional

        self.n: int = 0
        self.mean: float = 0.0
        # Cumulative sum statistics
        self.sum_up: float = 0.0
        self.min_sum_up: float = 0.0

        self.sum_down: float = 0.0
        self.max_sum_down: float = 0.0

        self.drift_detected: bool = False
        self.last_drift_pos: Optional[int] = None

    def update(self, val: float) -> Tuple[bool, float]:
        """
        Processes new stream value.
        Returns:
            (drift_detected: bool, current_statistic: float)
        """
        if math.isnan(val) or math.isinf(val):
            return False, 0.0

        self.n += 1
        self.drift_detected = False

        # Update running mean
        if self.n == 1:
            self.mean = val
        else:
            if self.alpha >= 1.0:
                self.mean += (val - self.mean) / self.n
            else:
                self.mean = self.alpha * self.mean + (1.0 - self.alpha) * val

        # Cumulative upward deviation
        self.sum_up += (val - self.mean - self.delta)
        if self.sum_up < self.min_sum_up:
            self.min_sum_up = self.sum_up

        stat_up = self.sum_up - self.min_sum_up
        stat = stat_up

        if stat_up > self.threshold:
            self.drift_detected = True
            self.last_drift_pos = self.n
            self._reset_statistics()
            return True, stat_up

        # Cumulative downward deviation
        if self.bidirectional:
            self.sum_down += (val - self.mean + self.delta)
            if self.sum_down > self.max_sum_down:
                self.max_sum_down = self.sum_down

            stat_down = self.max_sum_down - self.sum_down
            stat = max(stat_up, stat_down)

            if stat_down > self.threshold:
                self.drift_detected = True
                self.last_drift_pos = self.n
                self._reset_statistics()
                return True, stat_down

        return False, stat

    def _reset_statistics(self) -> None:
        """Resets cumulative deviation counters while preserving running mean."""
        self.sum_up = 0.0
        self.min_sum_up = 0.0
        self.sum_down = 0.0
        self.max_sum_down = 0.0

    def reset(self) -> None:
        """Fully resets detector state."""
        self.n = 0
        self.mean = 0.0
        self._reset_statistics()
        self.drift_detected = False
        self.last_drift_pos = None

    def to_dict(self) -> Dict[str, Any]:
        """Returns detector state summary."""
        return {
            "name": self.name,
            "threshold": self.threshold,
            "delta": self.delta,
            "alpha": self.alpha,
            "running_mean": round(self.mean, 6),
            "total_seen": self.n,
            "drift_detected": self.drift_detected,
            "last_drift_pos": self.last_drift_pos,
        }
