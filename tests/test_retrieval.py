import numpy as np
import pandas as pd
import pytest

from regime_retrieval.forecasting.analog_forecast import summarize_returns
from regime_retrieval.forecasting.baselines import (
    momentum_forecast,
    same_time_of_day,
)
from regime_retrieval.retrieval.neighbors import exact_knn
from regime_retrieval.retrieval.representations import build_contexts


def prepared_features(values, sessions, segments=None, minutes=None):
    sessions = np.asarray(sessions)
    if minutes is None:
        minutes = pd.Series(sessions).groupby(sessions).cumcount().to_numpy() * 5
    dates = pd.Timestamp("2024-01-02") + pd.to_timedelta(sessions, unit="D")
    frame = pd.DataFrame(values, columns=["second", "first"])
    frame["session"] = dates
    frame["session_index"] = sessions
    frame["timestamp"] = (
        dates.tz_localize("UTC")
        + pd.Timedelta(hours=14, minutes=35)
        + pd.to_timedelta(minutes, unit="m")
    )
    frame["minute_of_session"] = minutes
    frame["segment_id"] = sessions if segments is None else segments
    return frame


def test_context_chronology_and_feature_order_determine_distance():
    # Source columns intentionally differ from the requested feature order.
    bars = prepared_features(
        [[10, 1], [20, 2], [30, 3], [30, 3], [20, 2], [10, 1], [10, 1], [20, 2], [31, 3]],
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
    )
    contexts = build_contexts(bars, ["first", "second"], context_bars=3)
    np.testing.assert_array_equal(contexts.end_positions, [2, 5, 8])
    # An independently encoded chronological query distinguishes time-major
    # flattening from reversed time, feature-major order, or source-column order.
    selected, distances = exact_knn(
        np.array([1, 10, 2, 20, 3, 30]),
        contexts.vectors,
        bars.session_index.to_numpy()[contexts.end_positions],
        k=3,
        max_per_session=1,
    )
    np.testing.assert_array_equal(contexts.end_positions[selected], [2, 8, 5])
    np.testing.assert_allclose(distances, [0, 1, np.sqrt(808)])


def test_contexts_never_bridge_sessions_gaps_or_nonfinite_warmup():
    values = np.arange(28, dtype=float).reshape(14, 2)
    values[[0, 3, 7, 10], 0] = np.nan
    values[11, 1] = np.inf
    bars = prepared_features(
        values,
        sessions=[0] * 7 + [1] * 7,
        segments=[0] * 3 + [1] * 4 + [2] * 7,
        minutes=[0, 5, 10, 20, 25, 30, 35, 0, 5, 10, 15, 20, 25, 30],
    )
    contexts = build_contexts(bars, ["first", "second"], context_bars=2)
    np.testing.assert_array_equal(contexts.end_positions, [2, 5, 6, 9, 13])
    # Finite rows on opposite sides of a boundary must not form a context either.
    bars[["first", "second"]] = np.arange(28).reshape(14, 2)
    complete = build_contexts(bars, ["first", "second"], context_bars=2)
    np.testing.assert_array_equal(complete.end_positions, [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13])


def test_context_longer_than_available_history_has_no_neighbors():
    bars = prepared_features([[1, 2], [3, 4]], sessions=[0, 0])
    contexts = build_contexts(bars, ["first", "second"], context_bars=3)
    selected, distances = exact_knn(
        np.zeros(6), contexts.vectors, np.array([], dtype=int), k=10, max_per_session=1
    )
    assert selected.size == distances.size == 0


@pytest.mark.parametrize(
    ("metric", "expected_rows", "expected_distances"),
    [
        ("euclidean", [1, 3, 2, 0], [1, 2, np.sqrt(5), 9]),
        ("cosine", [0, 1, 2, 3], [0, 1 - 1 / np.sqrt(2), 1, 2]),
    ],
)
def test_exact_distance_metric_changes_neighbor_selection(
    metric, expected_rows, expected_distances
):
    selected, distances = exact_knn(
        np.array([1.0, 0.0]),
        np.array([[10.0, 0.0], [1.0, 1.0], [0.0, 2.0], [-1.0, 0.0]]),
        np.arange(4),
        k=4,
        max_per_session=1,
        metric=metric,
    )
    np.testing.assert_array_equal(selected, expected_rows)
    np.testing.assert_allclose(distances, expected_distances)


def test_stable_ties_and_session_cap_skip_nearest_duplicates_before_filling_k():
    candidates = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [2, 0]], dtype=float)
    sessions = np.array([9, 9, 4, 7, 8])
    selected, distances = exact_knn(np.zeros(2), candidates, sessions, k=3, max_per_session=1)
    np.testing.assert_array_equal(selected, [0, 2, 3])
    np.testing.assert_allclose(distances, [1, 1, 1])
    # Asking for more cannot violate the cap, repeat a neighbor, or pad the result.
    selected, distances = exact_knn(np.zeros(2), candidates, sessions, k=9, max_per_session=1)
    np.testing.assert_array_equal(selected, [0, 2, 3, 4])
    np.testing.assert_allclose(distances, [1, 1, 1, 2])
    selected, _ = exact_knn(np.zeros(2), candidates, sessions, k=3, max_per_session=2)
    np.testing.assert_array_equal(selected, [0, 1, 2])


def test_cosine_zero_vectors_match_each_other_and_are_orthogonal_to_nonzero():
    candidates = np.array([[1, 0], [0, 0], [-1, 0], [0, 0]], dtype=float)
    selected, distances = exact_knn(
        np.zeros(2), candidates, np.arange(4), k=4, max_per_session=1, metric="cosine"
    )
    np.testing.assert_array_equal(selected, [1, 3, 0, 2])
    np.testing.assert_allclose(distances, [0, 0, 1, 1])
    selected, distances = exact_knn(
        np.array([1.0, 0]), candidates, np.arange(4), k=4, max_per_session=1, metric="cosine"
    )
    np.testing.assert_array_equal(selected, [0, 1, 3, 2])
    np.testing.assert_allclose(distances, [0, 1, 1, 2])


def test_empirical_forecast_uses_equal_weights_interpolated_quantiles_and_strict_up():
    summary = summarize_returns(
        np.array([-0.2, -0.1, 0, 0.1, 0.6]),
        distances=np.array([0, 1, 2, 3, 100], dtype=float),
    )
    expected = {
        "mean": 0.08,
        "median": 0,
        "probability_up": 0.4,
        "q10": -0.16,
        "q25": -0.1,
        "q75": 0.1,
        "q90": 0.4,
        "return_std": np.sqrt(0.0776),
        "effective_neighbor_count": 5,
        "distance_mean": 21.2,
        "distance_min": 0,
        "distance_max": 100,
    }
    for key, value in expected.items():
        assert summary[key] == pytest.approx(value)


@pytest.mark.parametrize("returns", [np.array([]), np.array([0.1, np.nan]), np.array([np.inf])])
def test_empirical_forecast_rejects_unknown_or_empty_outcomes(returns):
    with pytest.raises(ValueError):
        summarize_returns(returns)


def test_time_of_day_selection_cannot_reintroduce_ineligible_contexts():
    eligible = np.array([1, 4, 6])
    minutes = np.array([5, 5, 5, 10, 10, 5, 5])
    np.testing.assert_array_equal(same_time_of_day(eligible, minutes, query_minute=5), [1, 6])
    assert same_time_of_day(eligible, minutes, query_minute=15).size == 0


@pytest.mark.parametrize(
    ("recent", "horizon_bars", "expected", "probability"),
    [(0.21, 2, 0.1, 1), (0.21, 8, 0.4641, 1), (-0.36, 2, -0.2, 0), (0, 8, 0, 0.5)],
)
def test_momentum_log_extrapolation_scales_horizon_without_inventing_distribution(
    recent, horizon_bars, expected, probability
):
    forecast = momentum_forecast(recent, horizon_bars=horizon_bars, context_bars=4)
    assert forecast["median"] == pytest.approx(expected)
    assert forecast["mean"] == pytest.approx(expected)
    assert forecast["probability_up"] == probability
    for key in ("q10", "q25", "q75", "q90", "return_std"):
        assert np.isnan(forecast[key])
