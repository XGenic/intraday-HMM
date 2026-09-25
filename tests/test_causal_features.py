import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from regime_retrieval.config import FeatureConfig
from regime_retrieval.features import compute_causal_features


def prepared_bars(close, volume=None, sessions=None, segments=None):
    count = len(close)
    sessions = sessions if sessions is not None else [0] * count
    segments = segments if segments is not None else sessions
    dates = pd.Timestamp("2024-01-02") + pd.to_timedelta(sessions, unit="D")
    slots = pd.Series(sessions).groupby(sessions).cumcount() * 5
    prices = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": dates.tz_localize("UTC")
            + pd.Timedelta(hours=14, minutes=35)
            + pd.to_timedelta(slots.to_numpy(), unit="m"),
            "session": dates,
            "session_index": sessions,
            "minute_of_session": slots,
            "segment_id": segments,
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": volume if volume is not None else np.ones(count),
        }
    )


def test_hand_calculated_returns_volatility_range_and_vwap():
    bars = prepared_bars([10, 20, 80], [1, 2, 1])
    result = compute_causal_features(bars, FeatureConfig(realized_vol_window=2), 5)
    np.testing.assert_allclose(result.log_return, [np.nan, np.log(2), np.log(4)], equal_nan=True)
    np.testing.assert_allclose(
        result.realized_volatility, [np.nan, np.nan, np.log(2) / np.sqrt(2)], equal_nan=True
    )
    np.testing.assert_allclose(result.range_fraction, [0.2, 0.1, 0.025])
    np.testing.assert_allclose(result.session_vwap, [10, 50 / 3, 32.5])
    np.testing.assert_allclose(result.vwap_distance, [0, 1 / 6, 47.5 / 80])
    assert_frame_equal(result[bars.columns], bars)
    assert result.day_of_week.tolist() == [1, 1, 1]


def test_zero_volume_is_undefined_until_volume_arrives_and_resets_each_session():
    bars = prepared_bars([10, 20, 30, 40, 50], [0, 0, 2, 0, 1], [0, 0, 0, 1, 1])
    result = compute_causal_features(bars, FeatureConfig(realized_vol_window=2), 5)
    np.testing.assert_allclose(
        result.session_vwap, [np.nan, np.nan, 30, np.nan, 50], equal_nan=True
    )
    np.testing.assert_allclose(result.vwap_distance, [np.nan, np.nan, 0, np.nan, 0], equal_nan=True)


def test_vendor_vwap_is_per_bar_price_for_cumulative_session_vwap():
    bars = prepared_bars([10, 20, 30], [1, 3, 2])
    bars["vwap"] = [9, 18, 27]
    result = compute_causal_features(bars, FeatureConfig(realized_vol_window=2), 5)
    np.testing.assert_allclose(result.session_vwap, [9, 63 / 4, 117 / 6])
    np.testing.assert_allclose(result.vwap_distance, [0.1, (20 - 63 / 4) / 20, (30 - 117 / 6) / 30])


def test_segment_and_session_boundaries_reset_returns_and_rolling_history():
    bars = prepared_bars(
        [10, 20, 80, 160, 320, 1280, 20, 40, 160],
        sessions=[0, 0, 0, 0, 0, 0, 2, 2, 2],
        segments=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )
    bars.loc[3:5, "minute_of_session"] += 5
    bars.loc[3:5, "timestamp"] += pd.Timedelta(minutes=5)
    result = compute_causal_features(bars, FeatureConfig(realized_vol_window=2), 5)
    np.testing.assert_allclose(
        result.log_return,
        [np.nan, np.log(2), np.log(4)] * 3,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        result.realized_volatility,
        [np.nan, np.nan, np.log(2) / np.sqrt(2)] * 3,
        equal_nan=True,
    )
    # Missing bars are not invented: session VWAP retains only observed volume.
    assert result.loc[3, "session_vwap"] == pytest.approx((10 + 20 + 80 + 160) / 4)
    assert result.loc[6, "session_vwap"] == 20


def test_feature_switches_do_not_disable_volatility_or_discard_raw_data():
    config = FeatureConfig(
        log_return=False,
        realized_vol_window=2,
        range_fraction=False,
        volume_tod_adjusted=False,
        vwap_distance=False,
    )
    bars = prepared_bars([10, 20, 80])
    result = compute_causal_features(bars, config, 5)
    assert result.loc[2, "realized_volatility"] == pytest.approx(np.log(2) / np.sqrt(2))
    assert "log_return" not in result
    assert "range_fraction" not in result
    assert "vwap_distance" not in result
    assert_frame_equal(result[bars.columns], bars)


def test_future_perturbation_and_prefix_calculation_leave_past_unchanged():
    bars = prepared_bars([10, 12, 11, 15, 14, 20, 22, 19], [1, 4, 2, 6, 3, 8, 5, 7])
    config = FeatureConfig(realized_vol_window=2)
    full = compute_causal_features(bars, config, 5)
    perturbed = bars.copy()
    perturbed.loc[5:, ["open", "high", "low", "close"]] *= 100
    perturbed.loc[5:, "volume"] *= 10000
    changed = compute_causal_features(perturbed, config, 5)
    prefix = compute_causal_features(bars.iloc[:5], config, 5)
    assert_frame_equal(full.iloc[:5], changed.iloc[:5])
    assert_frame_equal(full.iloc[:5], prefix)
