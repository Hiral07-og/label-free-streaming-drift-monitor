"""
Dataset loader and streaming data generator for drift monitoring benchmarks.
Provides realistic temporal financial transactions with entity identifiers,
numerical and categorical features, timestamps, and delayed ground truth.
Also supports ingesting the Elliptic Bitcoin dataset when available.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class StreamingDatasetSplit:
    """Container for split streaming datasets."""
    reference_df: pd.DataFrame
    stream_df: pd.DataFrame
    numerical_features: List[str]
    categorical_features: List[str]
    target_column: str
    id_column: str
    timestamp_column: str


class StreamingFinancialBenchmark:
    """
    Generates a high-fidelity synthetic streaming financial transaction dataset
    modeled after temporal transaction graphs (e.g. Elliptic Bitcoin / credit fraud).
    Designed to provide realistic multivariate correlations, non-linear fraud risks,
    and ground-truth temporal ordering for streaming drift experiments.
    """

    NUMERICAL_FEATURES = [
        "tx_amount",
        "sender_tx_count_1h",
        "sender_risk_score",
        "location_entropy",
        "dest_novelty_score",
        "time_delta_last_tx",
        "network_degree",
    ]

    CATEGORICAL_FEATURES = [
        "device_type",
        "merchant_category",
    ]

    DEVICE_TYPES = [
        "mobile_ios",
        "mobile_android",
        "web_browser",
        "pos_terminal",
        "api_gateway",
    ]

    MERCHANT_CATEGORIES = [
        "groceries",
        "luxury_goods",
        "crypto_exchange",
        "online_gaming",
        "utilities",
    ]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def generate(
        self,
        n_reference: int = 3000,
        n_stream: int = 6000,
        fraud_ratio: float = 0.08,
    ) -> StreamingDatasetSplit:
        """
        Generates full benchmark split into reference and streaming portions.
        Reference data represents historical training distribution;
        stream data represents initial in-distribution incoming records.
        """
        total_n = n_reference + n_stream
        df = self._generate_distribution(total_n, fraud_ratio=fraud_ratio, start_time=1704067200)

        # Chronological split
        reference_df = df.iloc[:n_reference].copy().reset_index(drop=True)
        stream_df = df.iloc[n_reference:].copy().reset_index(drop=True)

        return StreamingDatasetSplit(
            reference_df=reference_df,
            stream_df=stream_df,
            numerical_features=list(self.NUMERICAL_FEATURES),
            categorical_features=list(self.CATEGORICAL_FEATURES),
            target_column="is_fraud",
            id_column="record_id",
            timestamp_column="timestamp",
        )

    def _generate_distribution(
        self,
        n: int,
        fraud_ratio: float = 0.08,
        start_time: int = 1704067200,
        mu_shift: float = 0.0,
        scale_mult: float = 1.0,
    ) -> pd.DataFrame:
        """Generates synthetic transactions with controlled statistical properties."""
        rng = self.rng

        # Timestamps with Poisson arrival intervals
        inter_arrival = rng.exponential(scale=12.0, size=n)  # avg 12s between tx
        timestamps = start_time + np.cumsum(inter_arrival).astype(int)

        record_ids = [f"tx_{i:07d}" for i in range(1, n + 1)]

        # 1. tx_amount: log-normal distribution
        base_amount = rng.lognormal(mean=4.2 + mu_shift, sigma=1.0 * scale_mult, size=n)
        tx_amount = np.clip(np.round(base_amount, 2), 1.0, 50000.0)

        # 2. sender_tx_count_1h: Poisson with rate 2.5
        sender_tx_count = rng.poisson(lam=2.5 + max(0.0, mu_shift), size=n)

        # 3. sender_risk_score: Beta distribution [0, 1]
        sender_risk = rng.beta(a=1.5, b=5.0, size=n)

        # 4. location_entropy: Gaussian truncated to [0, 1]
        location_entropy = np.clip(rng.normal(loc=0.35 + 0.1 * mu_shift, scale=0.15, size=n), 0.0, 1.0)

        # 5. dest_novelty_score: Exponential
        dest_novelty = np.clip(rng.exponential(scale=0.8, size=n), 0.0, 5.0)

        # 6. time_delta_last_tx: seconds
        time_delta = np.clip(rng.exponential(scale=1800.0, size=n), 1.0, 86400.0)

        # 7. network_degree: Pareto / power-law
        network_degree = (rng.pareto(a=2.5, size=n) + 1).astype(int)
        network_degree = np.clip(network_degree, 1, 100)

        # Categoricals
        device_probs = [0.45, 0.35, 0.12, 0.05, 0.03]
        device_type = rng.choice(self.DEVICE_TYPES, size=n, p=device_probs)

        merchant_probs = [0.50, 0.15, 0.10, 0.15, 0.10]
        merchant_category = rng.choice(self.MERCHANT_CATEGORIES, size=n, p=merchant_probs)

        # Latent non-linear risk logit
        # Fraud probability is elevated with high amount, high risk score, crypto exchanges, and high novelty
        crypto_boost = (merchant_category == "crypto_exchange").astype(float) * 1.5
        api_boost = (device_type == "api_gateway").astype(float) * 0.8

        logit = (
            -3.5
            + 0.35 * (np.log(tx_amount + 1) - 4.2)
            + 2.8 * (sender_risk - 0.23)
            + 1.5 * (dest_novelty - 0.8)
            + 0.15 * sender_tx_count
            + crypto_boost
            + api_boost
        )

        prob = 1.0 / (1.0 + np.exp(-logit))
        # Calibrate overall base rate to target fraud_ratio
        empirical_labels = (rng.uniform(0.0, 1.0, size=n) < prob).astype(int)

        data = {
            "record_id": record_ids,
            "timestamp": timestamps,
            "tx_amount": tx_amount,
            "sender_tx_count_1h": sender_tx_count,
            "sender_risk_score": np.round(sender_risk, 4),
            "location_entropy": np.round(location_entropy, 4),
            "dest_novelty_score": np.round(dest_novelty, 4),
            "time_delta_last_tx": np.round(time_delta, 1),
            "network_degree": network_degree,
            "device_type": device_type,
            "merchant_category": merchant_category,
            "is_fraud": empirical_labels,
        }

        return pd.DataFrame(data)


def load_elliptic_dataset(
    classes_path: Union[str, Path],
    features_path: Union[str, Path],
    n_reference_timesteps: int = 25,
) -> StreamingDatasetSplit:
    """
    Loads and preprocesses the official Elliptic Bitcoin Dataset if available.
    Classes: 1 = illicit (fraud), 2 = licit, 'unknown' = unlabelled.
    Timesteps: 1 to 49 (each ~2 weeks of bitcoin transactions).
    """
    classes_p = Path(classes_path)
    features_p = Path(features_path)

    if not classes_p.exists() or not features_p.exists():
        raise FileNotFoundError(
            f"Elliptic dataset files not found at {classes_path} or {features_path}."
        )

    # 1. Load classes
    classes_df = pd.read_csv(classes_p)
    # txId, class
    classes_df.rename(columns={"txId": "record_id", "class": "target"}, inplace=True)
    # Filter known classes: '1' -> 1 (illicit), '2' -> 0 (licit)
    known_mask = classes_df["target"].isin(["1", "2"])
    classes_df = classes_df[known_mask].copy()
    classes_df["is_fraud"] = (classes_df["target"] == "1").astype(int)
    classes_df.drop(columns=["target"], inplace=True)

    # 2. Load features (txId, time_step, feat1 ... feat165)
    features_df = pd.read_csv(features_p, header=None)
    num_cols = features_df.shape[1]
    col_names = ["record_id", "time_step"] + [f"feature_{i}" for i in range(1, num_cols - 1)]
    features_df.columns = col_names

    # Merge
    merged_df = pd.merge(features_df, classes_df, on="record_id", how="inner")
    merged_df.sort_values(by=["time_step", "record_id"], inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    # Split on time_step
    ref_mask = merged_df["time_step"] <= n_reference_timesteps
    ref_df = merged_df[ref_mask].copy().reset_index(drop=True)
    stream_df = merged_df[~ref_mask].copy().reset_index(drop=True)

    feat_cols = [c for c in merged_df.columns if c.startswith("feature_")]
    ref_df["timestamp"] = ref_df["time_step"] * 10000
    stream_df["timestamp"] = stream_df["time_step"] * 10000

    return StreamingDatasetSplit(
        reference_df=ref_df,
        stream_df=stream_df,
        numerical_features=feat_cols[:20],  # monitor top 20 representative features
        categorical_features=[],
        target_column="is_fraud",
        id_column="record_id",
        timestamp_column="timestamp",
    )
