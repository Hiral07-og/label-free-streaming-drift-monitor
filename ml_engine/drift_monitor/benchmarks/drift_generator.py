"""
Synthetic Drift Generator for controlled experimental evaluation.
Injects:
- Abrupt drift (step distribution shift at known index)
- Gradual drift (sigmoid transition over specified window)
- Recurring drift (oscillating cyclic shifts)
- Feature-specific drift (perturbation applied only to selected features)
- Concept drift (perturbing the decision boundary P(Y|X))
Tracks exact injection points to enable precise measurement of detection delay and false alarms.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class DriftExperimentDataset:
    """Container for synthetic drift stream with ground-truth change metadata."""
    stream_df: pd.DataFrame
    drift_type: str
    injection_start_idx: int
    injection_end_idx: Optional[int]
    affected_features: List[str]
    shift_magnitude: float
    description: str


class SyntheticDriftInjector:
    """
    Transforms stationary base streaming datasets by injecting mathematical perturbations
    at specified time positions with defined transition profiles.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def inject_abrupt_drift(
        self,
        base_df: pd.DataFrame,
        injection_idx: int = 1000,
        affected_features: Optional[List[str]] = None,
        shift_magnitude: float = 2.5,
    ) -> DriftExperimentDataset:
        """
        Abrupt drift: sudden step change in distribution beginning at injection_idx.
        """
        df = base_df.copy().reset_index(drop=True)
        n = len(df)
        if affected_features is None:
            affected_features = ["tx_amount", "sender_risk_score"]

        mask = np.arange(n) >= injection_idx

        for feat in affected_features:
            if feat in df.columns:
                std_val = max(1e-4, float(df[feat].std()))
                # Add shift_magnitude standard deviations
                df.loc[mask, feat] = df.loc[mask, feat] + shift_magnitude * std_val

        return DriftExperimentDataset(
            stream_df=df,
            drift_type="ABRUPT",
            injection_start_idx=injection_idx,
            injection_end_idx=n - 1,
            affected_features=affected_features,
            shift_magnitude=shift_magnitude,
            description=f"Abrupt drift of {shift_magnitude} sigma injected at index {injection_idx}",
        )

    def inject_gradual_drift(
        self,
        base_df: pd.DataFrame,
        start_idx: int = 1000,
        duration: int = 1500,
        affected_features: Optional[List[str]] = None,
        shift_magnitude: float = 2.5,
    ) -> DriftExperimentDataset:
        """
        Gradual drift: smooth sigmoid transition between baseline and shifted distribution
        over a window of duration records.
        """
        df = base_df.copy().reset_index(drop=True)
        n = len(df)
        if affected_features is None:
            affected_features = ["tx_amount", "dest_novelty_score"]

        # Sigmoid transition weights in [0, 1]
        x = np.arange(n)
        z = (x - (start_idx + duration / 2.0)) / (duration / 6.0)
        weights = 1.0 / (1.0 + np.exp(-np.clip(z, -10.0, 10.0)))
        weights[:start_idx] = 0.0

        for feat in affected_features:
            if feat in df.columns:
                std_val = max(1e-4, float(df[feat].std()))
                shift = weights * (shift_magnitude * std_val)
                df[feat] = df[feat] + shift

        return DriftExperimentDataset(
            stream_df=df,
            drift_type="GRADUAL",
            injection_start_idx=start_idx,
            injection_end_idx=min(n - 1, start_idx + duration),
            affected_features=affected_features,
            shift_magnitude=shift_magnitude,
            description=f"Gradual drift starting at {start_idx} over {duration} records",
        )

    def inject_recurring_drift(
        self,
        base_df: pd.DataFrame,
        cycle_length: int = 1000,
        affected_features: Optional[List[str]] = None,
        shift_magnitude: float = 2.0,
    ) -> DriftExperimentDataset:
        """
        Recurring drift: periodic sinusoidal fluctuation simulating seasonal/cyclical patterns.
        """
        df = base_df.copy().reset_index(drop=True)
        n = len(df)
        if affected_features is None:
            affected_features = ["tx_amount", "sender_tx_count_1h"]

        # Sinusoidal multiplier in [0, 1]
        t = np.arange(n)
        cycles = np.sin(2.0 * np.pi * t / cycle_length)
        weights = np.maximum(0.0, cycles)

        for feat in affected_features:
            if feat in df.columns:
                std_val = max(1e-4, float(df[feat].std()))
                df[feat] = df[feat] + weights * (shift_magnitude * std_val)

        return DriftExperimentDataset(
            stream_df=df,
            drift_type="RECURRING",
            injection_start_idx=int(cycle_length / 4),
            injection_end_idx=n - 1,
            affected_features=affected_features,
            shift_magnitude=shift_magnitude,
            description=f"Recurring cyclical drift with period {cycle_length}",
        )

    def inject_feature_specific_drift(
        self,
        base_df: pd.DataFrame,
        injection_idx: int = 1000,
        shifts: Optional[Dict[str, float]] = None,
    ) -> DriftExperimentDataset:
        """
        Feature-specific drift: isolated shifts applied to specific subset of features
        while leaving all other features completely stationary.
        """
        df = base_df.copy().reset_index(drop=True)
        n = len(df)
        if shifts is None:
            shifts = {
                "dest_novelty_score": 3.0,
                "location_entropy": 2.0,
            }

        mask = np.arange(n) >= injection_idx
        for feat, shift_sigmas in shifts.items():
            if feat in df.columns:
                std_val = max(1e-4, float(df[feat].std()))
                df.loc[mask, feat] = df.loc[mask, feat] + shift_sigmas * std_val

        return DriftExperimentDataset(
            stream_df=df,
            drift_type="FEATURE_SPECIFIC",
            injection_start_idx=injection_idx,
            injection_end_idx=n - 1,
            affected_features=list(shifts.keys()),
            shift_magnitude=float(np.mean(list(shifts.values()))),
            description=f"Feature-specific drift affecting: {list(shifts.keys())}",
        )
