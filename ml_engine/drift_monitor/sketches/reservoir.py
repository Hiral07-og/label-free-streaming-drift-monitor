"""
Fixed-capacity Reservoir Sampling (Vitter's Algorithm R).
Maintains an unbiased uniform random sample of size K from a stream of unknown length N.
Essential for bounded-memory multivariate drift detection (MMD, C2ST).
"""

import random
import sys
from typing import Any, Dict, List, Optional
import numpy as np


class ReservoirSampler:
    """
    Reservoir sampling maintains a uniform random sample of size K
    from an unbounded stream without knowing stream length in advance.
    """

    def __init__(self, capacity: int = 1000, random_state: Optional[int] = 42, name: str = "reservoir"):
        if capacity <= 0:
            raise ValueError("Capacity must be positive.")
        self.capacity = capacity
        self.name = name
        self.random_state = random_state
        self.rng = random.Random(random_state)
        self.reservoir: List[Any] = []
        self.total_seen: int = 0

    def add(self, item: Any) -> bool:
        """
        Adds an item to the reservoir with probability K / N.
        Returns True if item was placed in the reservoir, False otherwise.
        """
        self.total_seen += 1

        if len(self.reservoir) < self.capacity:
            self.reservoir.append(item)
            return True

        # Generate random index between 0 and total_seen - 1
        j = self.rng.randint(0, self.total_seen - 1)
        if j < self.capacity:
            self.reservoir[j] = item
            return True

        return False

    def get_sample(self) -> List[Any]:
        """Returns shallow copy of current reservoir samples."""
        return list(self.reservoir)

    def get_feature_matrix(self, feature_names: List[str]) -> np.ndarray:
        """
        Extracts a 2D float NumPy matrix of shape (n_samples, n_features)
        from a reservoir storing dictionary records.
        """
        if not self.reservoir:
            return np.empty((0, len(feature_names)), dtype=np.float64)

        rows = []
        for rec in self.reservoir:
            if isinstance(rec, dict):
                row = [float(rec.get(f, 0.0)) for f in feature_names]
            elif isinstance(rec, (list, tuple, np.ndarray)):
                row = [float(x) for x in rec]
            else:
                row = [float(rec)]
            rows.append(row)

        return np.array(rows, dtype=np.float64)

    def is_full(self) -> bool:
        """Whether reservoir has reached target capacity K."""
        return len(self.reservoir) >= self.capacity

    def clear(self) -> None:
        """Resets reservoir."""
        self.reservoir = []
        self.total_seen = 0

    def memory_bytes(self) -> int:
        """Approximate memory in bytes."""
        base = sys.getsizeof(self) + sys.getsizeof(self.reservoir)
        if self.reservoir:
            item_sample_size = sys.getsizeof(self.reservoir[0])
            return base + len(self.reservoir) * item_sample_size
        return base

    def to_dict(self) -> Dict[str, Any]:
        """Returns reservoir state summary."""
        return {
            "name": self.name,
            "capacity": self.capacity,
            "current_size": len(self.reservoir),
            "total_seen": self.total_seen,
            "is_full": self.is_full(),
            "memory_bytes": self.memory_bytes(),
        }
