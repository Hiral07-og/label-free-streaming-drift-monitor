"""
Count-Min Sketch with Conservative Update for categorical frequency tracking.
Maintains bounded memory while guaranteeing frequency point estimates:
f_hat(x) >= f(x) and with probability 1 - delta: f_hat(x) <= f(x) + epsilon * N
"""

import hashlib
import math
import sys
from typing import Any, Dict, List, Optional
import numpy as np


class CountMinSketch:
    """
    Count-Min Sketch data structure for frequency tracking in data streams.
    Parameters:
        epsilon: Maximum relative error (default 0.005 -> width = 544)
        delta: Failure probability (default 0.01 -> depth = 5)
        conservative_update: Only increments minimum counters to reduce overestimation.
    """

    def __init__(
        self,
        epsilon: float = 0.005,
        delta: float = 0.01,
        conservative_update: bool = True,
        name: str = "cms",
    ):
        self.name = name
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.conservative_update = conservative_update

        # Width w = ceil(e / epsilon), Depth d = ceil(ln(1 / delta))
        self.width = max(16, int(math.ceil(math.e / self.epsilon)))
        self.depth = max(3, int(math.ceil(math.log(1.0 / self.delta))))

        # 2D table of shape (depth, width)
        self.table = np.zeros((self.depth, self.width), dtype=np.int64)
        self.total_count: int = 0

        # Deterministic salt seeds for d hash functions
        self.salts = [f"{name}_salt_{i}".encode("utf-8") for i in range(self.depth)]

    def _hash(self, item_bytes: bytes, row: int) -> int:
        """Computes hash bucket index in [0, width - 1] for row i."""
        # Fast MD5-based salted hash truncated to integer
        h = hashlib.md5(self.salts[row] + item_bytes).digest()
        # Take first 8 bytes as unsigned 64-bit int
        val = int.from_bytes(h[:8], byteorder="little", signed=False)
        return val % self.width

    def update(self, item: Any, count: int = 1) -> None:
        """Adds observation of item with weight count."""
        if count <= 0:
            return

        item_str = str(item).encode("utf-8")
        self.total_count += count

        indices = [self._hash(item_str, row) for row in range(self.depth)]

        if not self.conservative_update:
            for row in range(self.depth):
                self.table[row, indices[row]] += count
        else:
            # Conservative update: only increment counters that are at the current minimum
            current_vals = [self.table[row, indices[row]] for row in range(self.depth)]
            min_val = min(current_vals)
            target = min_val + count
            for row in range(self.depth):
                if self.table[row, indices[row]] < target:
                    self.table[row, indices[row]] = target

    def estimate_frequency(self, item: Any) -> int:
        """
        Estimates the frequency of an item.
        Returns the minimum count across all d hash buckets.
        """
        item_str = str(item).encode("utf-8")
        min_est = sys.maxsize
        for row in range(self.depth):
            idx = self._hash(item_str, row)
            val = int(self.table[row, idx])
            if val < min_est:
                min_est = val
        return 0 if min_est == sys.maxsize else min_est

    def estimate_relative_frequency(self, item: Any) -> float:
        """Estimates frequency normalized by total stream items observed."""
        if self.total_count == 0:
            return 0.0
        return self.estimate_frequency(item) / self.total_count

    def merge(self, other: "CountMinSketch") -> "CountMinSketch":
        """Combines two sketches with identical dimensions."""
        if self.depth != other.depth or self.width != other.width:
            raise ValueError(
                f"Cannot merge sketches with different shapes: "
                f"({self.depth}, {self.width}) vs ({other.depth}, {other.width})"
            )
        self.table += other.table
        self.total_count += other.total_count
        return self

    def memory_bytes(self) -> int:
        """Calculates total memory footprint of sketch in bytes."""
        return int(self.table.nbytes + sys.getsizeof(self))

    def to_dict(self) -> Dict[str, Any]:
        """Returns metadata and memory profile."""
        return {
            "name": self.name,
            "width": self.width,
            "depth": self.depth,
            "total_count": self.total_count,
            "conservative_update": self.conservative_update,
            "memory_bytes": self.memory_bytes(),
        }
