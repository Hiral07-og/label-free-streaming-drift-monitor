"""
Multiple testing corrections for multi-feature statistical drift monitoring.
Controls Family-Wise Error Rate (FWER) via Bonferroni or False Discovery Rate (FDR)
via Benjamini-Hochberg to prevent excessive false alarms across many feature tests.
"""

from typing import Any, Dict, List, Tuple
import numpy as np


def bonferroni_correction(
    p_values: Dict[str, float],
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    """
    Applies Bonferroni multiple testing correction.
    p_adj = min(1.0, p * m)
    Reject if p_adj < alpha (or p < alpha / m).
    """
    m = len(p_values)
    if m == 0:
        return {}

    threshold = alpha / m
    results = {}
    for feature, p in p_values.items():
        p_adj = min(1.0, p * m)
        results[feature] = {
            "p_raw": float(p),
            "p_adjusted": float(p_adj),
            "is_significant": bool(p < threshold),
            "method": "bonferroni",
            "threshold": float(threshold),
        }
    return results


def benjamini_hochberg_correction(
    p_values: Dict[str, float],
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    """
    Applies Benjamini-Hochberg (BH) False Discovery Rate (FDR) procedure.
    Sorts p-values, finds critical cut point, and calculates adjusted q-values.
    """
    m = len(p_values)
    if m == 0:
        return {}

    items = list(p_values.items())
    # Sort by p-value ascending
    items.sort(key=lambda x: x[1])

    features = [it[0] for it in items]
    raw_p = np.array([it[1] for it in items], dtype=np.float64)

    # Calculate q-values step-down
    # q_i = min_{j >= i} (m / j * p_j)
    factors = m / np.arange(1, m + 1)
    q_vals = raw_p * factors
    q_vals = np.minimum.accumulate(q_vals[::-1])[::-1]
    q_vals = np.clip(q_vals, 0.0, 1.0)

    # Significant if p_i <= (i / m) * alpha
    ranks = np.arange(1, m + 1)
    bh_thresholds = (ranks / m) * alpha
    significant_mask = raw_p <= bh_thresholds

    # Find largest rank k where significant
    if np.any(significant_mask):
        max_k = np.max(np.where(significant_mask)[0])
        final_sig = np.zeros(m, dtype=bool)
        final_sig[: max_k + 1] = True
    else:
        final_sig = np.zeros(m, dtype=bool)

    results = {}
    for i, feature in enumerate(features):
        results[feature] = {
            "p_raw": float(raw_p[i]),
            "p_adjusted": float(q_vals[i]),
            "is_significant": bool(final_sig[i]),
            "method": "benjamini_hochberg",
            "threshold": float(bh_thresholds[i]),
        }

    return results
