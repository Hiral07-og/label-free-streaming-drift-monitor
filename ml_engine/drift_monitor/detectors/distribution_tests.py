"""
Distribution comparison tests:
- Two-sample Kolmogorov-Smirnov (KS) test
- Population Stability Index (PSI) with adaptive reference binning
- Categorical Chi-Square Goodness-of-Fit test
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats


@dataclass
class KSTestResult:
    """Result of Kolmogorov-Smirnov two-sample test."""
    feature_name: str
    statistic: float
    p_value: float
    drift_detected: bool
    alpha: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "statistic": round(self.statistic, 5),
            "p_value": round(self.p_value, 6),
            "drift_detected": self.drift_detected,
            "alpha": self.alpha,
        }


@dataclass
class PSIResult:
    """Result of Population Stability Index test."""
    feature_name: str
    psi_value: float
    status: str  # "STABLE", "MODERATE", "SIGNIFICANT"
    drift_detected: bool
    n_bins: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "psi_value": round(self.psi_value, 5),
            "status": self.status,
            "drift_detected": self.drift_detected,
            "n_bins": self.n_bins,
        }


def kolmogorov_smirnov_test(
    reference_sample: np.ndarray,
    current_sample: np.ndarray,
    feature_name: str = "feature",
    alpha: float = 0.05,
) -> KSTestResult:
    """
    Computes two-sided Kolmogorov-Smirnov test between reference and current samples.
    """
    ref = np.asarray(reference_sample, dtype=np.float64).flatten()
    cur = np.asarray(current_sample, dtype=np.float64).flatten()

    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    if len(ref) < 5 or len(cur) < 5:
        return KSTestResult(
            feature_name=feature_name,
            statistic=0.0,
            p_value=1.0,
            drift_detected=False,
            alpha=alpha,
        )

    res = stats.ks_2samp(ref, cur, alternative="two-sided")
    p_val = float(res.pvalue)
    stat = float(res.statistic)

    return KSTestResult(
        feature_name=feature_name,
        statistic=stat,
        p_value=p_val,
        drift_detected=(p_val < alpha),
        alpha=alpha,
    )


def calculate_psi(
    reference_sample: np.ndarray,
    current_sample: np.ndarray,
    feature_name: str = "feature",
    n_bins: int = 10,
    psi_warning: float = 0.10,
    psi_critical: float = 0.25,
    eps: float = 1e-4,
) -> PSIResult:
    """
    Calculates Population Stability Index (PSI) using reference quantiles.
    PSI < 0.10: Stable (No shift)
    0.10 <= PSI < 0.25: Moderate shift
    PSI >= 0.25: Significant shift
    """
    ref = np.asarray(reference_sample, dtype=np.float64).flatten()
    cur = np.asarray(current_sample, dtype=np.float64).flatten()

    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    if len(ref) < 10 or len(cur) < 10:
        return PSIResult(
            feature_name=feature_name,
            psi_value=0.0,
            status="STABLE",
            drift_detected=False,
            n_bins=n_bins,
        )

    # Establish quantile breakpoints from reference distribution
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(ref, quantiles)
    # Deduplicate bin edges in case of constant/discrete values
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return PSIResult(
            feature_name=feature_name,
            psi_value=0.0,
            status="STABLE",
            drift_detected=False,
            n_bins=n_bins,
        )

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Calculate counts in each bin
    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    # Convert to proportions with smoothing epsilon
    ref_props = (ref_counts + eps) / (len(ref) + eps * len(ref_counts))
    cur_props = (cur_counts + eps) / (len(cur) + eps * len(cur_counts))

    # PSI formula: sum((A_i - E_i) * ln(A_i / E_i))
    psi_terms = (cur_props - ref_props) * np.log(cur_props / ref_props)
    psi_value = float(np.sum(psi_terms))

    if psi_value >= psi_critical:
        status = "SIGNIFICANT"
        drift = True
    elif psi_value >= psi_warning:
        status = "MODERATE"
        drift = True
    else:
        status = "STABLE"
        drift = False

    return PSIResult(
        feature_name=feature_name,
        psi_value=psi_value,
        status=status,
        drift_detected=drift,
        n_bins=len(bin_edges) - 1,
    )


def categorical_chi2_test(
    reference_counts: Dict[str, int],
    current_counts: Dict[str, int],
    feature_name: str = "category",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Performs Chi-Square Goodness-of-Fit test comparing observed category
    frequencies against reference category distribution.
    """
    all_keys = set(reference_counts.keys()) | set(current_counts.keys())
    total_ref = max(1, sum(reference_counts.values()))
    total_cur = max(1, sum(current_counts.values()))

    if total_cur < 20 or len(all_keys) < 2:
        return {
            "feature_name": feature_name,
            "chi2_stat": 0.0,
            "p_value": 1.0,
            "drift_detected": False,
        }

    observed = []
    expected = []
    for k in sorted(all_keys):
        o = current_counts.get(k, 0)
        p_ref = max(1e-5, reference_counts.get(k, 0) / total_ref)
        e = p_ref * total_cur
        observed.append(o)
        expected.append(e)

    # Normalize expected to match observed sum exactly
    expected = np.array(expected, dtype=np.float64) * (total_cur / np.sum(expected))
    observed = np.array(observed, dtype=np.float64)

    chi2_stat, p_val = stats.chisquare(f_obs=observed, f_exp=expected)

    return {
        "feature_name": feature_name,
        "chi2_stat": round(float(chi2_stat), 4),
        "p_value": round(float(p_val), 6),
        "drift_detected": bool(p_val < alpha),
    }
