import numpy as np
import pandas as pd
import pytest

from regime_retrieval.evaluation.metrics import (
    LOG_LOSS_EPSILON,
    metric_tables,
    paired_error_differences,
)


def forecasts():
    timestamps = pd.to_datetime(
        [
            "2025-01-02T15:00:00Z",
            "2025-01-02T15:05:00Z",
            "2025-01-03T15:00:00Z",
            "2025-01-03T15:05:00Z",
        ]
    )
    return pd.DataFrame(
        {
            "query_timestamp": timestamps,
            "query_session": timestamps.tz_localize(None).normalize(),
            "method": "raw_knn",
            "horizon_minutes": 5,
            "fold": [7, 7, 2, 2],
            "actual_return": [-0.02, 0.01, 0.03, 0.0],
            "median": [-0.01, 0.02, 0.01, 0.0],
            "probability_up": [0.0, 1.0, 0.5, 0.1],
            "q10": [-0.03, 0.0, 0.0, -0.01],
            "q90": [0.0, 0.03, 0.02, 0.01],
            "return_std": [0.01, 0.02, 0.03, 0.04],
            "effective_neighbor_count": [2, 4, 6, 8],
            "distance_mean": [0.1, 0.2, 0.3, 0.4],
            "distance_min": [0.0, 0.1, 0.2, 0.3],
            "distance_max": [0.2, 0.3, 0.4, 0.5],
        }
    )


def test_hand_calculable_point_probability_interval_and_neighbor_metrics():
    aggregate, _, _ = metric_tables(forecasts())
    metrics = aggregate.iloc[0]
    assert metrics["mae"] == pytest.approx(0.01)
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.00015))
    assert metrics["mean_signed_error"] == pytest.approx(0)
    assert metrics["spearman"] == pytest.approx(0.8)
    assert metrics["brier"] == pytest.approx(0.065)
    epsilon = LOG_LOSS_EPSILON
    expected_log_loss = -(2 * np.log1p(-epsilon) + np.log(0.5) + np.log(0.9)) / 4
    assert metrics["log_loss"] == pytest.approx(expected_log_loss)
    assert metrics["n_forecasts"] == 4
    assert metrics["n_sessions"] == 2
    assert metrics["n_intervals"] == 4
    assert metrics["interval_10_90_width"] == pytest.approx(0.025)
    assert metrics["interval_10_90_coverage"] == pytest.approx(0.75)
    assert metrics["neighbor_return_std_mean"] == pytest.approx(0.025)
    assert metrics["effective_neighbor_count_mean"] == 5
    assert metrics["distance_mean"] == pytest.approx(0.25)
    assert metrics["distance_min_mean"] == pytest.approx(0.15)
    assert metrics["distance_max_mean"] == pytest.approx(0.35)


def test_calibration_includes_all_exact_bin_boundaries_and_probability_one():
    frame = pd.concat([forecasts().iloc[[0]]] * 11, ignore_index=True)
    frame["query_timestamp"] = pd.date_range("2025-01-02T15:00Z", periods=11, freq="5min")
    frame["probability_up"] = np.arange(11) / 10
    frame["actual_return"] = [0.0] * 10 + [0.01]
    aggregate, _, calibration = metric_tables(frame)
    assert calibration["n_forecasts"].tolist() == [1] * 9 + [2]
    assert calibration.iloc[0]["mean_probability_up"] == 0
    assert calibration.iloc[0]["observed_up_frequency"] == 0
    assert calibration.iloc[-1]["mean_probability_up"] == pytest.approx(0.95)
    assert calibration.iloc[-1]["observed_up_frequency"] == 0.5
    assert aggregate.iloc[0]["n_forecasts"] == calibration["n_forecasts"].sum()


def test_certain_wrong_forecasts_have_finite_clipped_log_loss_and_exact_brier():
    frame = forecasts().iloc[:2].copy()
    frame["actual_return"] = [0.01, 0.0]
    frame["probability_up"] = [0.0, 1.0]
    aggregate, _, calibration = metric_tables(frame)
    expected = -(np.log(LOG_LOSS_EPSILON) + np.log1p(-(1 - LOG_LOSS_EPSILON))) / 2
    assert aggregate.iloc[0]["log_loss"] == pytest.approx(expected)
    assert aggregate.iloc[0]["brier"] == 1
    assert calibration.loc[calibration["bin"].isin([0, 9]), "n_forecasts"].tolist() == [1, 1]
    empty = calibration.loc[calibration["n_forecasts"].eq(0)]
    assert empty["mean_probability_up"].isna().all()
    assert empty["observed_up_frequency"].isna().all()


@pytest.mark.parametrize("constant_column", ["median", "actual_return"])
def test_spearman_is_undefined_when_either_series_is_constant(constant_column):
    frame = forecasts()
    frame[constant_column] = 0.01
    aggregate, _, _ = metric_tables(frame)
    assert np.isnan(aggregate.iloc[0]["spearman"])


def test_point_only_momentum_has_no_fabricated_intervals_or_dispersion():
    frame = forecasts()
    frame["method"] = "momentum"
    frame[["q10", "q90", "return_std", "distance_mean", "distance_min", "distance_max"]] = np.nan
    frame["effective_neighbor_count"] = 0
    aggregate, _, _ = metric_tables(frame)
    assert aggregate.iloc[0]["n_intervals"] == 0
    assert np.isnan(aggregate.iloc[0]["interval_10_90_width"])
    assert np.isnan(aggregate.iloc[0]["interval_10_90_coverage"])
    assert np.isnan(aggregate.iloc[0]["neighbor_return_std_mean"])
    assert aggregate.iloc[0]["effective_neighbor_count_mean"] == 0


def test_interval_coverage_includes_both_endpoints():
    frame = forecasts().iloc[:2].copy()
    frame["actual_return"] = [-0.03, 0.03]
    aggregate, _, _ = metric_tables(frame)
    assert aggregate.iloc[0]["interval_10_90_coverage"] == 1


def test_chronological_folds_and_timestamp_paired_losses_ignore_input_order():
    raw = forecasts()
    control = raw.assign(method="unconditional", median=0.0)
    frame = pd.concat([control, raw], ignore_index=True).sample(frac=1, random_state=42)
    aggregate, folds, _ = metric_tables(frame)
    assert folds["fold"].drop_duplicates().tolist() == [7, 2]
    raw_folds = folds.loc[folds["method"].eq("raw_knn")]
    assert raw_folds["n_forecasts"].tolist() == [2, 2]
    assert raw_folds["n_sessions"].tolist() == [1, 1]
    assert raw_folds["mean_signed_error"].tolist() == pytest.approx([0.01, -0.01])
    assert aggregate["n_forecasts"].tolist() == [4, 4]
    paired = paired_error_differences(frame).iloc[0]
    assert paired["control"] == "unconditional"
    assert paired["n_forecasts"] == 4
    assert paired["n_sessions"] == 2
    assert paired["mean_absolute_error_difference"] == pytest.approx(-0.005)
    assert paired["mean_squared_error_difference"] == pytest.approx(-0.0002)


def test_comparisons_reject_unpaired_forecasts_instead_of_changing_the_cohort():
    raw = forecasts()
    frame = pd.concat([raw, raw.iloc[:-1].assign(method="unconditional")], ignore_index=True)
    with pytest.raises(ValueError, match="identical timestamp cohorts"):
        metric_tables(frame)
    with pytest.raises(ValueError, match="identical timestamp cohorts"):
        paired_error_differences(frame)
