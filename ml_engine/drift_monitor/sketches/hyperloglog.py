"""
HyperLogLog (HLL) sketch for streaming distinct element cardinality estimation.
Maintains bounded memory (m = 2^b registers) with theoretical relative error
bounded by approximately 1.04 / sqrt(m).
"""

import hashlib
import math
import sys
from typing import Any, Dict
import numpy as np


class HyperLogLog:
    """
    HyperLogLog cardinality estimator using 64-bit hash domain.
    Parameters:
        precision_bits: b in [4, 16], determining m = 2^b registers.
                        Default b=10 gives m=1024 registers (~3.25% standard error).
    """

    def __init__(self, precision_bits: int = 10, name: str = "hll"):
        if not (4 <= precision_bits <= 16):
            raise ValueError("precision_bits must be between 4 and 16 inclusive.")

        self.name = name
        self.p = precision_bits
        self.m = 1 << precision_bits  # 2^b
        self.registers = np.zeros(self.m, dtype=np.uint8)
        self.total_elements: int = 0

        # Alpha bias correction constant
        if self.m == 16:
            self.alpha = 0.673
        elif self.m == 32:
            self.alpha = 0.697
        elif self.m == 64:
            self.alpha = 0.709
        else:
            self.alpha = 0.7213 / (1.0 + 1.079 / self.m)

    def _hash64(self, item_bytes: bytes) -> int:
        """Computes uniform 64-bit integer hash from item bytes."""
        h = hashlib.blake2b(item_bytes, digest_size=8).digest()
        return int.from_bytes(h, byteorder="big", signed=False)

    def add(self, item: Any) -> None:
        """Adds an item to the HyperLogLog sketch."""
        item_bytes = str(item).encode("utf-8")
        x = self._hash64(item_bytes)
        self.total_elements += 1

        # First p bits identify the register index j
        j = x >> (64 - self.p)

        # Remaining 64 - p bits for leading zeros count
        w = x & ((1 << (64 - self.p)) - 1)
        # Leading zeros count + 1
        if w == 0:
            rho = 64 - self.p + 1
        else:
            # Count leading zeros in w
            rho = (64 - self.p) - w.bit_length() + 1

        if rho > self.registers[j]:
            self.registers[j] = rho

    def estimate_cardinality(self) -> float:
        """
        Calculates harmonic mean estimate with Linear Counting small-range correction.
        """
        # Harmonic mean: sum(2^(-M[j]))
        indicator = np.sum(np.power(2.0, -self.registers.astype(np.float64)))
        if indicator == 0.0:
            return 0.0

        raw_estimate = (self.alpha * (self.m ** 2)) / indicator

        # Small range correction: Linear counting
        if raw_estimate <= 2.5 * self.m:
            zeros = int(np.count_nonzero(self.registers == 0))
            if zeros > 0:
                # E* = m * ln(m / V)
                return float(self.m * math.log(self.m / zeros))

        return float(raw_estimate)

    def merge(self, other: "HyperLogLog") -> "HyperLogLog":
        """Merges another HLL with identical register precision."""
        if self.p != other.p:
            raise ValueError(
                f"Cannot merge HyperLogLog with different precision: {self.p} vs {other.p}"
            )
        np.maximum(self.registers, other.registers, out=self.registers)
        self.total_elements += other.total_elements
        return self

    def memory_bytes(self) -> int:
        """Memory footprint of registers and sketch object in bytes."""
        return int(self.registers.nbytes + sys.getsizeof(self))

    def to_dict(self) -> Dict[str, Any]:
        """Returns sketch summary and estimated cardinality."""
        return {
            "name": self.name,
            "precision_bits": self.p,
            "registers_count": self.m,
            "total_observations": self.total_elements,
            "estimated_cardinality": round(self.estimate_cardinality(), 2),
            "memory_bytes": self.memory_bytes(),
        }
