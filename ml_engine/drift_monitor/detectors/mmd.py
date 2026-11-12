"""
Maximum Mean Discrepancy (MMD) multivariate two-sample drift detector.
Computes kernel-based distance between multidimensional reference distributions
and incoming streaming reservoir samples using Gaussian RBF kernels with permutation testing.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy.spatial.distance import cdist


@dataclass
class MMDResult:
    """Result of Maximum Mean Discrepancy test."""
    mmd_squared: float
    p_value: float
    drift_detected: bool
    gamma: float
    alpha: float
    sample_size_ref: int
    sample_size_cur: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mmd_squared": round(self.mmd_squared, 6),
            "p_value": round(self.p_value, 6),
            "drift_detected": self.drift_detected,
            "gamma": round(self.gamma, 6),
            "alpha": self.alpha,
            "sample_size_ref": self.sample_size_ref,
            "sample_size_cur": self.sample_size_cur,
        }


class MaximumMeanDiscrepancy:
    """
    Multivariate drift detector using unbiased MMD^2 with Gaussian RBF kernel.
    """

    def __init__(
        self,
        gamma: Optional[float] = None,
        n_permutations: int = 100,
        alpha: float = 0.05,
        random_state: int = 42,
    ):
        self.gamma = gamma
        self.n_permutations = n_permutations
        self.alpha = alpha
        self.rng = np.random.default_rng(random_state)

    def _compute_gamma(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Computes Gaussian kernel bandwidth gamma using median pairwise heuristic."""
        if self.gamma is not None:
            return self.gamma

        # Subsample if dataset is large to maintain efficiency
        max_sub = 300
        x_sub = X if len(X) <= max_sub else X[self.rng.choice(len(X), max_sub, replace=False)]
        y_sub = Y if len(Y) <= max_sub else Y[self.rng.choice(len(Y), max_sub, replace=False)]
        combined = np.vstack([x_sub, y_sub])

        dists = cdist(combined, combined, metric="euclidean")
        # Extract upper triangle off-diagonal distances
        triu_dists = dists[np.triu_indices_from(dists, k=1)]
        triu_dists = triu_dists[triu_dists > 0]

        if len(triu_dists) == 0:
            return 1.0

        median_dist = float(np.median(triu_dists))
        if median_dist <= 0:
            return 1.0

        sigma = median_dist
        return 1.0 / (2.0 * (sigma ** 2))

    def _unbiased_mmd2(self, K_xx: np.ndarray, K_yy: np.ndarray, K_xy: np.ndarray) -> float:
        """Computes unbiased estimator of MMD squared from kernel matrices."""
        m = K_xx.shape[0]
        n = K_yy.shape[0]

        # Zero out diagonal self-similarities
        np.fill_diagonal(K_xx, 0.0)
        np.fill_diagonal(K_yy, 0.0)

        term_xx = np.sum(K_xx) / (m * (m - 1)) if m > 1 else 0.0
        term_yy = np.sum(K_yy) / (n * (n - 1)) if n > 1 else 0.0
        term_xy = 2.0 * np.sum(K_xy) / (m * n) if (m > 0 and n > 0) else 0.0

        return float(term_xx + term_yy - term_xy)

    def test(
        self,
        X_ref: np.ndarray,
        Y_cur: np.ndarray,
        max_samples: int = 500,
    ) -> MMDResult:
        """
        Runs MMD hypothesis test comparing X_ref and Y_cur.
        Returns MMDResult with test statistic and permutation p-value.
        """
        X = np.asarray(X_ref, dtype=np.float64)
        Y = np.asarray(Y_cur, dtype=np.float64)

        if len(X) < 10 or len(Y) < 10:
            return MMDResult(
                mmd_squared=0.0,
                p_value=1.0,
                drift_detected=False,
                gamma=1.0,
                alpha=self.alpha,
                sample_size_ref=len(X),
                sample_size_cur=len(Y),
            )

        # Subsample for bounded runtime if samples exceed max_samples
        if len(X) > max_samples:
            idx_x = self.rng.choice(len(X), max_samples, replace=False)
            X = X[idx_x]
        if len(Y) > max_samples:
            idx_y = self.rng.choice(len(Y), max_samples, replace=False)
            Y = Y[idx_y]

        m = len(X)
        n = len(Y)
        gamma = self._compute_gamma(X, Y)

        # Precompute kernel matrix for combined pool
        Z = np.vstack([X, Y])
        total_n = m + n
        D2 = cdist(Z, Z, metric="sqeuclidean")
        K = np.exp(-gamma * D2)

        K_xx = K[:m, :m].copy()
        K_yy = K[m:, m:].copy()
        K_xy = K[:m, m:].copy()

        obs_stat = self._unbiased_mmd2(K_xx, K_yy, K_xy)

        # Permutation test
        perm_stats = []
        for _ in range(self.n_permutations):
            perm = self.rng.permutation(total_n)
            idx_x_perm = perm[:m]
            idx_y_perm = perm[m:]

            K_xx_p = K[np.ix_(idx_x_perm, idx_x_perm)].copy()
            K_yy_p = K[np.ix_(idx_y_perm, idx_y_perm)].copy()
            K_xy_p = K[np.ix_(idx_x_perm, idx_y_perm)].copy()

            s = self._unbiased_mmd2(K_xx_p, K_yy_p, K_xy_p)
            perm_stats.append(s)

        perm_stats = np.array(perm_stats)
        p_val = float((np.sum(perm_stats >= obs_stat) + 1.0) / (self.n_permutations + 1.0))
        drift = bool(p_val < self.alpha and obs_stat > 0.0)

        return MMDResult(
            mmd_squared=obs_stat,
            p_value=p_val,
            drift_detected=drift,
            gamma=gamma,
            alpha=self.alpha,
            sample_size_ref=m,
            sample_size_cur=n,
        )
