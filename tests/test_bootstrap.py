import numpy as np
import pandas as pd
import pytest

from regime_retrieval.config import BootstrapConfig
from regime_retrieval.evaluation.bootstrap import paired_session_bootstrap

METHODS = (
    "unconditional",
    "same_time_of_day",
    "momentum",
    "raw_knn",
    "hmm_current",
    "hmm_trajectory",
)


def forecasts(lengths=(10, 10), embargoes=(0,), horizons=(5,)):
    rows = []
    for embargo in embargoes:
        for horizon in horizons:
            for session_index, length in enumerate(lengths):
                start = pd.Timestamp("2025-01-02T15:00Z") + pd.Timedelta(days=session_index)
                for bar in range(length):
                    for method in METHODS:
                        hmm = method.startswith("hmm_")
                        rows.append(
                            {
                                "query_timestamp": start + pd.Timedelta(minutes=5 * bar),
                                "query_session": start.date(),
                                "horizon_minutes": horizon,
                                "embargo_sessions": embargo,
                                "fold": session_index + 1,
                                "method": method,
                                "actual_return": 0.0,
                                "median": (1.0 + 2 * session_index) if hmm else 0.0,
                                "probability_up": (0.25 + 0.5 * session_index) if hmm else 0.0,
                            }
                        )
    return pd.DataFrame(rows)


def test_whole_sessions_not_bars_are_resampled_for_all_losses():
    # Two sessions with repeated bars: four equally likely ordered session draws
    # are (A,A), (A,B), (B,A), (B,B). The central 80% interval spans both extremes.
    result = paired_session_bootstrap(
        forecasts(), BootstrapConfig(n_resamples=1000, confidence=0.8), 19
    )
    aggregate = result.loc[result["fold"].eq(0)]
    expected = {
        "absolute_error": (1.0, 2.0, 3.0),
        "squared_error": (1.0, 5.0, 9.0),
        "brier": (0.0625, 0.3125, 0.5625),
    }
    for row in aggregate.itertuples():
        lower, mean, upper = expected[row.loss]
        assert row.mean_difference == pytest.approx(mean)
        assert row.ci_lower == pytest.approx(lower)
        assert row.ci_upper == pytest.approx(upper)
        assert row.n_sessions == 2
        assert row.n_forecasts == 20
        assert row.status == "ok"


def test_unequal_session_lengths_weight_forecasts_in_each_resample():
    # A has one forecast; B has three. Ordered draw means for absolute loss are
    # [1, 2.5, 2.5, 3], not [1, 2, 2, 3]. Its middle 40% is exactly 2.5.
    result = paired_session_bootstrap(
        forecasts(lengths=(1, 3)), BootstrapConfig(n_resamples=1000, confidence=0.4), 19
    )
    aggregate = result.loc[result["fold"].eq(0)]
    for row in aggregate.itertuples():
        expected = {"absolute_error": 2.5, "squared_error": 7.0, "brier": 0.4375}[row.loss]
        assert row.mean_difference == pytest.approx(expected)
        assert row.ci_lower == pytest.approx(expected)
        assert row.ci_upper == pytest.approx(expected)
        assert row.n_forecasts == 4


def test_single_session_never_claims_a_confidence_interval():
    result = paired_session_bootstrap(forecasts(lengths=(8,)), BootstrapConfig(n_resamples=20), 42)
    assert set(result["fold"]) == {0, 1}
    assert result["ci_lower"].isna().all()
    assert result["ci_upper"].isna().all()
    assert result["status"].eq("insufficient_sessions").all()
    assert result["n_sessions"].eq(1).all()
    assert result["mean_difference"].notna().all()


def test_bootstrap_is_exactly_input_order_invariant_and_preserves_all_groups():
    frame = forecasts(lengths=(1, 3), embargoes=(5, 0, 1), horizons=(10, 5))
    config = BootstrapConfig(n_resamples=20)
    ordered = paired_session_bootstrap(frame, config, 31)
    shuffled = paired_session_bootstrap(frame.sample(frac=1, random_state=7), config, 31)
    pd.testing.assert_frame_equal(ordered, shuffled)
    assert set(ordered["embargo_sessions"]) == {0, 1, 5}
    assert set(ordered["horizon_minutes"]) == {5, 10}
    assert set(ordered["fold"]) == {0, 1, 2}
    assert len(ordered) == 3 * 2 * 3 * 2 * 3
    assert ordered.loc[ordered["fold"].ne(0), "status"].eq("insufficient_sessions").all()


def test_negative_differences_favor_hmm_and_zero_returns_are_not_up():
    frame = forecasts(lengths=(1, 1))
    frame.loc[frame["method"].eq("raw_knn"), ["median", "probability_up"]] = [0.2, 0.9]
    frame.loc[frame["method"].str.startswith("hmm_"), ["median", "probability_up"]] = [0.1, 0.1]
    result = paired_session_bootstrap(frame, BootstrapConfig(n_resamples=20), 3)
    expected = {"absolute_error": -0.1, "squared_error": -0.03, "brier": -0.8}
    for row in result.loc[result["fold"].eq(0)].itertuples():
        assert row.mean_difference == pytest.approx(expected[row.loss])
        assert row.ci_lower == pytest.approx(expected[row.loss])
        assert row.ci_upper == pytest.approx(expected[row.loss])


@pytest.mark.parametrize(
    "column,value", [("actual_return", 0.3), ("query_session", "other"), ("fold", 2)]
)
def test_rejects_inconsistent_paired_observations(column, value):
    frame = forecasts(lengths=(2, 2))
    if column == "query_session":
        frame[column] = frame[column].astype(object)
    frame.loc[frame["method"].eq("hmm_trajectory").idxmax(), column] = value
    with pytest.raises(ValueError, match=column):
        paired_session_bootstrap(frame, BootstrapConfig(n_resamples=20), 0)


@pytest.mark.parametrize(
    "problem", ["duplicate", "missing_query", "missing_model", "different_query"]
)
def test_rejects_unpaired_forecasts_without_silent_intersection(problem):
    frame = forecasts(lengths=(2, 2))
    index = frame["method"].eq("hmm_trajectory").idxmax()
    if problem == "duplicate":
        frame = pd.concat([frame, frame.loc[[index]]], ignore_index=True)
    elif problem == "missing_query":
        frame = frame.drop(index)
    elif problem == "missing_model":
        frame = frame.loc[frame["method"].ne("hmm_trajectory")]
    else:
        frame.loc[index, "query_timestamp"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError):
        paired_session_bootstrap(frame, BootstrapConfig(n_resamples=20), 0)


def test_rejects_session_split_across_folds():
    frame = forecasts(lengths=(2, 2))
    timestamp = frame["query_timestamp"].min() + pd.Timedelta(minutes=5)
    frame.loc[frame["query_timestamp"].eq(timestamp), "fold"] = 2
    with pytest.raises(ValueError, match="exactly one fold"):
        paired_session_bootstrap(frame, BootstrapConfig(n_resamples=20), 0)


@pytest.mark.parametrize(
    "column,value", [("median", np.inf), ("probability_up", np.nan), ("probability_up", 1.1)]
)
def test_rejects_undefined_or_invalid_paired_losses(column, value):
    frame = forecasts(lengths=(1, 1))
    frame.loc[frame["method"].eq("hmm_current"), column] = value
    with pytest.raises(ValueError):
        paired_session_bootstrap(frame, BootstrapConfig(n_resamples=20), 0)
