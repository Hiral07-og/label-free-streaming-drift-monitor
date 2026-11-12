"""
Classifier Two-Sample Test (C2ST) for multivariate drift detection.
Trains a domain classifier to distinguish reference data (label 0) from current
streaming data (label 1). High holdout discriminability (AUC >> 0.5) rejects H0: P = Q.
Also identifies top contributing features responsible for the distribution change.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


@dataclass
class C2STResult:
    """Result of Classifier Two-Sample Test."""
    roc_auc: float
    p_value: float
    drift_detected: bool
    feature_importances: Dict[str, float]
    sample_size_ref: int
    sample_size_cur: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "roc_auc": round(self.roc_auc, 4),
            "p_value": round(self.p_value, 6),
            "drift_detected": self.drift_detected,
            "top_contributing_features": sorted(
                self.feature_importances.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "sample_size_ref": self.sample_size_ref,
            "sample_size_cur": self.sample_size_cur,
        }


class ClassifierTwoSampleTest:
    """
    Evaluates multivariate drift by testing whether a classifier can distinguish
    reference observations from current streaming observations.
    """

    def __init__(
        self,
        n_splits: int = 3,
        auc_threshold: float = 0.65,
        alpha: float = 0.05,
        random_state: int = 42,
    ):
        self.n_splits = max(2, n_splits)
        self.auc_threshold = auc_threshold
        self.alpha = alpha
        self.random_state = random_state

    def test(
        self,
        X_ref: np.ndarray,
        Y_cur: np.ndarray,
        feature_names: Optional[List[str]] = None,
        max_samples: int = 600,
    ) -> C2STResult:
        """
        Runs C2ST on reference vs current feature matrices.
        """
        X = np.asarray(X_ref, dtype=np.float64)
        Y = np.asarray(Y_cur, dtype=np.float64)

        if len(X) < 15 or len(Y) < 15:
            return C2STResult(
                roc_auc=0.5,
                p_value=1.0,
                drift_detected=False,
                feature_importances={},
                sample_size_ref=len(X),
                sample_size_cur=len(Y),
            )

        rng = np.random.default_rng(self.random_state)
        if len(X) > max_samples:
            X = X[rng.choice(len(X), max_samples, replace=False)]
        if len(Y) > max_samples:
            Y = Y[rng.choice(len(Y), max_samples, replace=False)]

        m, n = len(X), len(Y)
        features = np.vstack([X, Y])
        labels = np.concatenate([np.zeros(m, dtype=int), np.ones(n, dtype=int)])

        scaler = StandardScaler()
        features_norm = scaler.fit_transform(features)

        # Cross-validated domain discriminator
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        oof_probs = np.zeros(len(labels), dtype=np.float64)
        coefs = np.zeros(features.shape[1], dtype=np.float64)

        for train_idx, val_idx in skf.split(features_norm, labels):
            clf = LogisticRegression(C=1.0, max_iter=200, random_state=self.random_state)
            clf.fit(features_norm[train_idx], labels[train_idx])
            oof_probs[val_idx] = clf.predict_proba(features_norm[val_idx])[:, 1]
            coefs += np.abs(clf.coef_[0])

        coefs /= self.n_splits
        auc = float(roc_auc_score(labels, oof_probs))

        # Statistical hypothesis test: H0 is AUC = 0.5 (Hanley-McNeil variance approximation)
        # SE = sqrt((AUC*(1-AUC) + (n_pos-1)*(Q1 - AUC^2) + (n_neg-1)*(Q2 - AUC^2)) / (n_pos * n_neg))
        q1 = auc / (2.0 - auc) if auc < 1.0 else 1.0
        q2 = 2.0 * (auc ** 2) / (1.0 + auc) if auc > 0.0 else 0.0
        se_auc = np.sqrt(
            max(
                1e-6,
                (auc * (1.0 - auc) + (n - 1) * (q1 - auc ** 2) + (m - 1) * (q2 - auc ** 2))
                / (m * n),
            )
        )
        z = (auc - 0.5) / max(1e-6, se_auc)
        # One-sided p-value for AUC > 0.5
        p_val = float(1.0 - stats.norm.cdf(z))

        drift = bool(auc >= self.auc_threshold and p_val < self.alpha)

        # Map feature importances
        if feature_names is None or len(feature_names) != features.shape[1]:
            f_names = [f"feat_{i}" for i in range(features.shape[1])]
        else:
            f_names = feature_names

        f_imp = {f_names[i]: float(round(coefs[i], 4)) for i in range(len(f_names))}

        return C2STResult(
            roc_auc=auc,
            p_value=p_val,
            drift_detected=drift,
            feature_importances=f_imp,
            sample_size_ref=m,
            sample_size_cur=n,
        )
