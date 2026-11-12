"""
Baseline ML Model training, probability calibration, evaluation, and inference.
Calibrates probabilities via Isotonic or Sigmoid scaling to ensure confidence
estimates are mathematically valid for Confidence-Based Performance Estimation (CBPE).
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class BaselineEvaluationMetrics:
    """Evaluation metrics on baseline holdout or reference dataset."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    brier_score: float
    sample_size: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaselineModelWrapper:
    """
    Encapsulates feature preprocessing, base classifier, probability calibrator,
    and inference pipeline for streaming monitoring.
    """

    def __init__(
        self,
        numerical_features: List[str],
        categorical_features: List[str],
        random_state: int = 42,
    ):
        self.numerical_features = list(numerical_features)
        self.categorical_features = list(categorical_features)
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None
        self.baseline_metrics: Optional[BaselineEvaluationMetrics] = None
        self.is_fitted: bool = False

    def train_and_calibrate(
        self,
        train_df: pd.DataFrame,
        target_col: str = "is_fraud",
        val_df: Optional[pd.DataFrame] = None,
        calibration_method: str = "sigmoid",
    ) -> BaselineEvaluationMetrics:
        """
        Fits a calibrated ML pipeline:
        1. Numerical scaling & categorical one-hot encoding.
        2. GradientBoostingClassifier (or RandomForest).
        3. Cross-validated probability calibration (Platt scaling or Isotonic).
        4. Calculates baseline metrics on train/validation set.
        """
        X_train = train_df[self.numerical_features + self.categorical_features]
        y_train = train_df[target_col].values

        # Build column transformer
        transformers = []
        if self.numerical_features:
            transformers.append(
                ("num", StandardScaler(), self.numerical_features)
            )
        if self.categorical_features:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.categorical_features,
                )
            )

        preprocessor = ColumnTransformer(transformers=transformers)

        base_estimator = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=4,
            random_state=self.random_state,
        )

        calibrated_clf = CalibratedClassifierCV(
            estimator=base_estimator,
            method=calibration_method,
            cv=3,
        )

        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", calibrated_clf),
            ]
        )

        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True

        # Evaluate baseline performance
        eval_df = val_df if val_df is not None else train_df
        X_eval = eval_df[self.numerical_features + self.categorical_features]
        y_eval = eval_df[target_col].values

        probs = self.pipeline.predict_proba(X_eval)[:, 1]
        preds = (probs >= 0.5).astype(int)

        self.baseline_metrics = BaselineEvaluationMetrics(
            accuracy=float(accuracy_score(y_eval, preds)),
            precision=float(precision_score(y_eval, preds, zero_division=0)),
            recall=float(recall_score(y_eval, preds, zero_division=0)),
            f1=float(f1_score(y_eval, preds, zero_division=0)),
            roc_auc=float(roc_auc_score(y_eval, probs)),
            brier_score=float(brier_score_loss(y_eval, probs)),
            sample_size=int(len(y_eval)),
        )

        return self.baseline_metrics

    def predict_record(self, record_dict: Dict[str, Any]) -> Tuple[int, float]:
        """
        Infers binary class prediction and calibrated positive probability
        for an individual streaming record dictionary.
        Returns: (predicted_class: 0 or 1, calibrated_probability: [0.0, 1.0])
        """
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Model must be trained before predicting.")

        row = {k: [record_dict.get(k, 0.0)] for k in self.numerical_features + self.categorical_features}
        df_single = pd.DataFrame(row)
        prob = float(self.pipeline.predict_proba(df_single)[0, 1])
        pred = int(prob >= 0.5)
        return pred, prob

    def predict_batch(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Infers binary predictions and probabilities for a batch DataFrame.
        Returns: (preds_array, probs_array)
        """
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Model must be trained before predicting.")

        X = df[self.numerical_features + self.categorical_features]
        probs = self.pipeline.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return preds, probs

    def save(self, filepath: Union[str, Path]) -> None:
        """Serializes trained wrapper, preprocessor, and baseline metrics."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "pipeline": self.pipeline,
            "baseline_metrics": self.baseline_metrics,
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "random_state": self.random_state,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BaselineModelWrapper":
        """Deserializes trained wrapper from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file {filepath} not found.")

        state = joblib.load(path)
        wrapper = cls(
            numerical_features=state["numerical_features"],
            categorical_features=state["categorical_features"],
            random_state=state["random_state"],
        )
        wrapper.pipeline = state["pipeline"]
        wrapper.baseline_metrics = state["baseline_metrics"]
        wrapper.is_fitted = state["is_fitted"]
        return wrapper
