"""Bar-end features that use only observations available at that bar."""

import numpy as np
import pandas as pd

from regime_retrieval.config import FeatureConfig

METADATA_COLUMNS = [
    "timestamp",
    "session",
    "session_index",
    "minute_of_session",
    "segment_id",
]


def compute_causal_features(
    bars: pd.DataFrame, feature_config: FeatureConfig, bar_minutes: int
) -> pd.DataFrame:
    """Return aligned features without dropping warmups or filling missing bars.

    Returns and their rolling sample standard deviation reset at every contiguous
    segment. VWAP instead accumulates all *observed* bars in the current session;
    a missing bar is never synthesized. When supplied, vendor ``vwap`` is treated
    as a per-bar traded price, not a cumulative session VWAP, and is cumulatively
    volume-weighted exactly like the fallback typical price. Zero cumulative
    volume has undefined VWAP. An unknown positive-volume price keeps cumulative
    VWAP unknown thereafter. Input OHLCV, vendor fields and metadata are retained.

    Volume surprise requires training-only statistics and is added later by
    :class:`FeatureTransformer`. Day of week uses the exchange session date.
    """
    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive")
    required = METADATA_COLUMNS + ["open", "high", "low", "close", "volume"]
    missing = [name for name in required if name not in bars]
    if missing:
        raise ValueError(f"Prepared bars are missing columns: {missing}")

    result = bars.copy()
    groups = [result["session"], result["segment_id"]]
    previous_close = result["close"].groupby(groups, sort=False).shift()
    log_return = np.log(result["close"] / previous_close)
    if feature_config.log_return:
        result["log_return"] = log_return
    window = feature_config.realized_vol_window
    result["realized_volatility"] = log_return.groupby(groups, sort=False).transform(
        lambda values: values.rolling(window, min_periods=window).std(ddof=1)
    )
    if feature_config.range_fraction:
        result["range_fraction"] = (result["high"] - result["low"]) / result["close"]
    if feature_config.vwap_distance:
        price = (
            result["vwap"]
            if "vwap" in result
            else (result["high"] + result["low"] + result["close"]) / 3.0
        )
        weighted_price = (price * result["volume"]).where(result["volume"] != 0, 0.0)
        cumulative_value = weighted_price.groupby(result["session"], sort=False).cumsum(
            skipna=False
        )
        cumulative_volume = result["volume"].groupby(result["session"], sort=False).cumsum()
        result["session_vwap"] = cumulative_value / cumulative_volume.where(cumulative_volume > 0)
        result["vwap_distance"] = (result["close"] - result["session_vwap"]) / result["close"]
    result["day_of_week"] = pd.to_datetime(result["session"]).dt.dayofweek
    return result
