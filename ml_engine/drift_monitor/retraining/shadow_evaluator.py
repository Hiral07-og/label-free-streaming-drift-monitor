"""
Shadow-Model and Canary Candidate Evaluation Module.
Evaluates newly trained candidate models in shadow mode against the active production model
on a shared validation split, preventing unverified automated replacements.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_engine.drift_monitor.config import RetrainConfig
from ml_engine.drift_monitor.model import BaselineEvaluationMetrics, BaselineModelWrapper


@dataclass
class ShadowEvaluationResult:
    """Detailed comparison between active production model and candidate shadow model."""
    active_metrics: BaselineEvaluationMetrics
    candidate_metrics: BaselineEvaluationMetrics
    f1_delta: float
    auc_delta: float
    accuracy_delta: float
    brier_delta: float
    decision: str  # "PROMOTE_TO_PRODUCTION", "KEEP_ACTIVE", "NEEDS_HUMAN_REVIEW"
    approval_token: Optional[str]
    justification: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_metrics": self.active_metrics.to_dict(),
            "candidate_metrics": self.candidate_metrics.to_dict(),
            "deltas": {
                "f1_delta": round(self.f1_delta, 4),
                "auc_delta": round(self.auc_delta, 4),
                "accuracy_delta": round(self.accuracy_delta, 4),
                "brier_delta": round(self.brier_delta, 4),
            },
            "decision": self.decision,
            "approval_token": self.approval_token,
            "justification": self.justification,
        }


class ShadowModelEvaluator:
    """
    Trains and benchmarks a candidate model against the active model on fresh data.
    """

    def __init__(
        self,
        config: Optional[RetrainConfig] = None,
        random_state: int = 42,
    ):
        self.config = config or RetrainConfig()
        self.random_state = random_state

    def evaluate_candidate(
        self,
        active_model: BaselineModelWrapper,
        recent_data_with_labels: pd.DataFrame,
        target_column: str = "is_fraud",
    ) -> Tuple[ShadowEvaluationResult, Optional[BaselineModelWrapper]]:
        """
        Trains candidate model on train split of recent_data_with_labels,
        evaluates active vs candidate on validation split,
        and determines promotion recommendation.
        """
        if len(recent_data_with_labels) < 40:
            raise ValueError("Insufficient data to train and evaluate shadow model (minimum 40 rows required).")

        train_df, val_df = train_test_split(
            recent_data_with_labels,
            test_size=self.config.shadow_val_split,
            random_state=self.random_state,
            stratify=recent_data_with_labels[target_column]
            if recent_data_with_labels[target_column].nunique() > 1
            else None,
        )

        # 1. Train candidate model on recent train split
        candidate_model = BaselineModelWrapper(
            numerical_features=active_model.numerical_features,
            categorical_features=active_model.categorical_features,
            random_state=self.random_state,
        )
        cand_metrics = candidate_model.train_and_calibrate(
            train_df=train_df,
            val_df=val_df,
            target_col=target_column,
        )

        # 2. Evaluate active model on the exact same validation split
        active_preds, active_probs = active_model.predict_batch(val_df)
        y_val = val_df[target_column].values

        from sklearn.metrics import (
            accuracy_score,
            brier_score_loss,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        active_auc = (
            float(roc_auc_score(y_val, active_probs))
            if len(np.unique(y_val)) > 1
            else 0.5
        )

        act_metrics = BaselineEvaluationMetrics(
            accuracy=float(accuracy_score(y_val, active_preds)),
            precision=float(precision_score(y_val, active_preds, zero_division=0)),
            recall=float(recall_score(y_val, active_preds, zero_division=0)),
            f1=float(f1_score(y_val, active_preds, zero_division=0)),
            roc_auc=active_auc,
            brier_score=float(brier_score_loss(y_val, active_probs)),
            sample_size=int(len(y_val)),
        )

        # 3. Calculate deltas (candidate - active)
        f1_delta = cand_metrics.f1 - act_metrics.f1
        auc_delta = cand_metrics.roc_auc - act_metrics.roc_auc
        acc_delta = cand_metrics.accuracy - act_metrics.accuracy
        brier_delta = cand_metrics.brier_score - act_metrics.brier_score  # lower is better

        # 4. Decide promotion
        if f1_delta >= self.config.candidate_min_f1_gain and auc_delta >= -0.01:
            decision = "PROMOTE_TO_PRODUCTION"
            import uuid
            token = f"approval-{uuid.uuid4().hex[:12]}"
            justification = (
                f"Candidate model demonstrates superior performance on fresh data: "
                f"F1 gain +{f1_delta:.2%}, AUC gain {auc_delta:+.2%}. "
                f"Requires human sign-off with token {token} before activation."
            )
        elif f1_delta < 0.0 or auc_delta < -0.03:
            decision = "KEEP_ACTIVE"
            token = None
            justification = (
                f"Candidate model did not outperform active model (F1 delta: {f1_delta:+.2%}, "
                f"AUC delta: {auc_delta:+.2%}). Retaining active model in production."
            )
        else:
            decision = "NEEDS_HUMAN_REVIEW"
            token = None
            justification = (
                f"Candidate model performance is comparable to active model within noise margins "
                f"(F1 delta: {f1_delta:+.2%}). Manual engineer review recommended."
            )

        result = ShadowEvaluationResult(
            active_metrics=act_metrics,
            candidate_metrics=cand_metrics,
            f1_delta=f1_delta,
            auc_delta=auc_delta,
            accuracy_delta=acc_delta,
            brier_delta=brier_delta,
            decision=decision,
            approval_token=token,
            justification=justification,
        )

        return result, candidate_model
