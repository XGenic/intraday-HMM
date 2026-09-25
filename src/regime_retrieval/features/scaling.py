"""Frozen training-only feature standardization and preprocessing."""

import numpy as np
import pandas as pd

from regime_retrieval.config import FeatureConfig
from regime_retrieval.features.seasonality import VolumeSeasonality


class FeatureScaler:
    """Per-feature training mean and population standard deviation (ddof=0).

    Nonfinite values are missing observations. Numerically constant training
    features use a scale of one; entirely unobserved features cannot be fitted.
    """

    def __init__(self) -> None:
        self.mean_: pd.Series | None = None
        self.scale_: pd.Series | None = None

    def fit(self, training: pd.DataFrame) -> "FeatureScaler":
        finite = training.where(np.isfinite(training))
        mean = finite.mean()
        missing = mean.index[mean.isna()].tolist()
        if missing:
            raise ValueError(f"No finite training observations for features: {missing}")
        scale = finite.std(ddof=0)
        self.mean_ = mean
        # Do not magnify floating-point roundoff in near-constant price ratios.
        resolution = np.finfo(float).eps * np.maximum(mean.abs(), 1.0)
        self.scale_ = scale.where(scale > resolution, 1.0)
        return self

    def transform(self, values: pd.DataFrame) -> pd.DataFrame:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("FeatureScaler must be fitted before transform")
        selected = values.loc[:, self.mean_.index]
        return (selected.where(np.isfinite(selected)) - self.mean_) / self.scale_


class FeatureTransformer:
    """Fit seasonality and scaling only on the explicitly supplied training rows.

    ``transform`` retains the input index, metadata, OHLCV and auxiliary columns.
    Selected core feature names hold standardized values; ``raw_<feature>`` keeps
    each corresponding unstandardized value (for volume this is the time-of-day
    adjusted surprise, not raw volume). Only ``feature_columns`` enter retrieval.
    Warmup NaNs and unseen seasonal slots remain NaN. A fit lacking any finite
    training observations for a selected feature fails instead of guessing a
    distribution. ``fit_end`` is the latest timestamp supplied to a successful
    fit, including training warmups. Later transforms never alter fit state.
    """

    def __init__(self, feature_config: FeatureConfig) -> None:
        self.feature_config = feature_config.model_copy(deep=True)
        self.feature_columns: list[str] = []
        if feature_config.log_return:
            self.feature_columns.append("log_return")
        self.feature_columns.append("realized_volatility")
        if feature_config.range_fraction:
            self.feature_columns.append("range_fraction")
        if feature_config.volume_tod_adjusted:
            self.feature_columns.append("volume_surprise")
        if feature_config.vwap_distance:
            self.feature_columns.append("vwap_distance")
        self.seasonality: VolumeSeasonality | None = None
        self.scaler: FeatureScaler | None = None
        self.fit_end: pd.Timestamp | None = None

    def fit(self, causal_training: pd.DataFrame) -> "FeatureTransformer":
        if causal_training.empty:
            raise ValueError("Cannot fit feature preprocessing on empty training rows")
        timestamps = pd.to_datetime(causal_training["timestamp"], utc=True)
        if timestamps.isna().any():
            raise ValueError("Training timestamps must be available for every row")
        raw_columns = [name for name in self.feature_columns if name != "volume_surprise"]
        values = causal_training.loc[:, raw_columns].copy()
        seasonality = None
        if self.feature_config.volume_tod_adjusted:
            seasonality = VolumeSeasonality().fit(causal_training)
            values["volume_surprise"] = seasonality.transform(causal_training)
        scaler = FeatureScaler().fit(values.loc[:, self.feature_columns])
        self.seasonality = seasonality
        self.scaler = scaler
        self.fit_end = timestamps.max()
        return self

    def transform(self, causal: pd.DataFrame) -> pd.DataFrame:
        if self.scaler is None:
            raise RuntimeError("FeatureTransformer must be fitted before transform")
        result = causal.copy()
        if self.seasonality is not None:
            result["volume_surprise"] = self.seasonality.transform(causal)
        for name in self.feature_columns:
            result[f"raw_{name}"] = result[name]
        result[self.feature_columns] = self.scaler.transform(result[self.feature_columns])
        return result
