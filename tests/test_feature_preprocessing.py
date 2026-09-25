import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from regime_retrieval.config import FeatureConfig
from regime_retrieval.features import FeatureScaler, FeatureTransformer, VolumeSeasonality


def causal_training():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02T15:00Z", "2024-01-03T15:00Z"]),
            "session": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "session_index": [0, 1],
            "minute_of_session": [25, 25],
            "segment_id": [0, 1],
            "volume": [10.0, 14.0],
            "log_return": [-0.1, 0.1],
            "realized_volatility": [0.2, 0.4],
            "range_fraction": [0.02, 0.02],
            "vwap_distance": [0.0, 0.04],
        },
        index=[5, 9],
    )


def test_time_of_day_median_and_scaled_mad_are_hand_calculated_per_slot():
    train = pd.DataFrame({"minute_of_session": [0, 5, 0, 5], "volume": [10, 100, 14, 120]})
    seasonal = VolumeSeasonality().fit(train)
    query = pd.DataFrame({"minute_of_session": [0, 5, 10], "volume": [16, 130, 200]})
    np.testing.assert_allclose(
        seasonal.transform(query), [4 / (1.4826 * 2), 20 / (1.4826 * 10), np.nan], equal_nan=True
    )
    # Slot scales differ, so equal relative surprises agree despite raw volume levels.
    assert seasonal.transform(query).iloc[0] == pytest.approx(seasonal.transform(query).iloc[1])


def test_standardization_does_not_amplify_roundoff_in_constant_price_ratios():
    ratio = 0.002
    values = pd.DataFrame({"range_fraction": [ratio, np.nextafter(ratio, np.inf), ratio]})
    transformed = FeatureScaler().fit(values).transform(values)
    assert transformed.range_fraction.abs().max() < 1e-16


def test_zero_mad_and_all_zero_volume_have_deterministic_nonzero_scales():
    train = pd.DataFrame({"minute_of_session": [0, 0, 5, 5], "volume": [0, 0, 20, 20]})
    seasonal = VolumeSeasonality().fit(train)
    query = pd.DataFrame({"minute_of_session": [0, 5], "volume": [5, 40]})
    np.testing.assert_allclose(seasonal.transform(query), [5, 1])
    np.testing.assert_allclose(seasonal.transform(train), [0, 0, 0, 0])


def test_scaler_uses_training_population_variance_and_retains_missing_positions():
    training = pd.DataFrame({"varying": [1.0, 3.0, np.nan], "constant": [5.0, 5.0, 5.0]})
    scaler = FeatureScaler().fit(training)
    future = pd.DataFrame({"varying": [5.0, np.nan], "constant": [8.0, 5.0]}, index=[4, 7])
    expected = pd.DataFrame({"varying": [3.0, np.nan], "constant": [3.0, 0.0]}, index=[4, 7])
    assert_frame_equal(scaler.transform(future), expected)
    with pytest.raises(ValueError, match="No finite training"):
        FeatureScaler().fit(pd.DataFrame({"unobserved": [np.nan]}))


def test_transformer_standardizes_selected_features_and_preserves_raw_and_metadata():
    training = causal_training()
    transformer = FeatureTransformer(FeatureConfig(realized_vol_window=2)).fit(training)
    result = transformer.transform(training)
    expected_names = [
        "log_return",
        "realized_volatility",
        "range_fraction",
        "volume_surprise",
        "vwap_distance",
    ]
    assert transformer.feature_columns == expected_names
    np.testing.assert_allclose(result[expected_names], [[-1, -1, 0, -1, -1], [1, 1, 0, 1, 1]])
    assert transformer.fit_end == training.timestamp.max()
    metadata = [
        "timestamp",
        "session",
        "session_index",
        "minute_of_session",
        "segment_id",
        "volume",
    ]
    assert_frame_equal(result[metadata], training[metadata])
    for name in ["log_return", "realized_volatility", "range_fraction", "vwap_distance"]:
        assert_series_equal(result[f"raw_{name}"], training[name], check_names=False)
    np.testing.assert_allclose(result.raw_volume_surprise, [-1 / 1.4826, 1 / 1.4826])


def test_future_transforms_cannot_refit_seasonality_or_scaling():
    training = causal_training()
    transformer = FeatureTransformer(FeatureConfig(realized_vol_window=2)).fit(training)
    training_before = transformer.transform(training)
    seasonal_before = transformer.seasonality.statistics_.copy(deep=True)
    means_before = transformer.scaler.mean_.copy(deep=True)
    scales_before = transformer.scaler.scale_.copy(deep=True)
    fit_end_before = transformer.fit_end
    future = training.iloc[[0]].copy()
    future["timestamp"] = pd.Timestamp("2024-02-01T15:00Z")
    future["volume"] = 16.0
    future["log_return"] = 0.3
    future["realized_volatility"] = 0.5
    future["range_fraction"] = 0.03
    future["vwap_distance"] = 0.06
    np.testing.assert_allclose(
        transformer.transform(future)[transformer.feature_columns], [[3, 2, 0.01, 2, 2]], atol=1e-14
    )
    future["volume"] = 1e12
    future["log_return"] = 100.0
    transformer.transform(future)
    assert_frame_equal(transformer.transform(training), training_before)
    assert_frame_equal(transformer.seasonality.statistics_, seasonal_before)
    assert_series_equal(transformer.scaler.mean_, means_before)
    assert_series_equal(transformer.scaler.scale_, scales_before)
    assert transformer.fit_end == fit_end_before


def test_warmups_and_unseen_slots_remain_missing_without_row_removal():
    training = causal_training()
    transformer = FeatureTransformer(FeatureConfig(realized_vol_window=2)).fit(training)
    query = training.copy()
    query.loc[5, "log_return"] = np.nan
    query.loc[5, "realized_volatility"] = np.nan
    query.loc[9, "minute_of_session"] = 30
    result = transformer.transform(query)
    assert result.index.equals(query.index)
    assert pd.isna(result.loc[5, "log_return"])
    assert pd.isna(result.loc[5, "realized_volatility"])
    assert pd.isna(result.loc[9, "volume_surprise"])
    assert result.loc[9, "log_return"] == pytest.approx(1)


def test_prefit_transform_fails_and_disabled_features_need_no_fit_values():
    training = causal_training()
    with pytest.raises(RuntimeError, match="fitted"):
        FeatureTransformer(FeatureConfig()).transform(training)
    with pytest.raises(RuntimeError, match="fitted"):
        VolumeSeasonality().transform(training)
    with pytest.raises(RuntimeError, match="fitted"):
        FeatureScaler().transform(training[["log_return"]])
    config = FeatureConfig(
        log_return=False,
        realized_vol_window=2,
        range_fraction=False,
        volume_tod_adjusted=False,
        vwap_distance=False,
    )
    minimal = training.drop(
        columns=["volume", "minute_of_session", "log_return", "range_fraction", "vwap_distance"]
    )
    transformer = FeatureTransformer(config).fit(minimal)
    np.testing.assert_allclose(
        transformer.transform(minimal)[transformer.feature_columns], [[-1], [1]]
    )
    assert transformer.feature_columns == ["realized_volatility"]
