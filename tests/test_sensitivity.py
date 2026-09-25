import pandas as pd
import pytest

from regime_retrieval.evaluation.sensitivity import sensitivity_metrics


def forecasts():
    rows = []
    methods = (
        "unconditional",
        "same_time_of_day",
        "momentum",
        "raw_knn",
        "hmm_current",
        "hmm_trajectory",
    )
    for embargo in (0, 1, 5):
        for horizon in (5, 10):
            for day in range(2):
                start = pd.Timestamp("2025-01-02T15:00Z") + pd.Timedelta(days=day)
                for bar in range(2):
                    timestamp = start + pd.Timedelta(minutes=5 * bar)
                    for method in methods:
                        rows.append(
                            {
                                "embargo_sessions": embargo,
                                "query_timestamp": timestamp,
                                "query_session": start.date(),
                                "horizon_minutes": horizon,
                                "actual_outcome_end": timestamp + pd.Timedelta(minutes=horizon),
                                "method": method,
                                "fold": day + 1,
                                "actual_return": 0.0,
                                "median": (embargo + 1) * 0.01,
                                "probability_up": 0.5,
                                "q10": -0.1,
                                "q90": 0.1,
                                "return_std": 0.02,
                                "effective_neighbor_count": 10,
                                "distance_min": 0.1,
                                "distance_mean": 0.2,
                                "distance_max": 0.3,
                            }
                        )
    return pd.DataFrame(rows)


def test_sensitivity_reports_each_embargo_and_actual_fold_on_common_cohorts():
    result = sensitivity_metrics(forecasts())
    assert len(result) == 3 * 2 * 6 * 3
    assert set(result["embargo_sessions"]) == {0, 1, 5}
    assert set(result["fold"]) == {0, 1, 2}
    for row in result.itertuples():
        assert row.mae == pytest.approx((row.embargo_sessions + 1) * 0.01)
        assert row.rmse == pytest.approx((row.embargo_sessions + 1) * 0.01)
        assert row.brier == 0.25
        assert row.n_forecasts == (4 if row.fold == 0 else 2)
        assert row.n_sessions == (2 if row.fold == 0 else 1)


def test_sensitivity_is_input_order_invariant():
    frame = forecasts()
    pd.testing.assert_frame_equal(
        sensitivity_metrics(frame), sensitivity_metrics(frame.sample(frac=1, random_state=18))
    )


@pytest.mark.parametrize("missing", ["query", "horizon", "method", "one_prediction"])
def test_sensitivity_rejects_changed_cohorts_even_when_variant_is_internally_paired(missing):
    frame = forecasts()
    selected = frame["embargo_sessions"].eq(5)
    if missing == "query":
        selected &= frame["query_timestamp"].eq(frame["query_timestamp"].min())
    elif missing == "horizon":
        selected &= frame["horizon_minutes"].eq(10)
    elif missing == "method":
        selected &= frame["method"].eq("hmm_current")
    else:
        selected &= frame.index == frame.loc[selected].index[0]
    with pytest.raises(ValueError, match="cohorts"):
        sensitivity_metrics(frame.loc[~selected])


@pytest.mark.parametrize("column", ["actual_return", "query_session", "fold", "actual_outcome_end"])
def test_sensitivity_rejects_outcome_or_metadata_changes_between_embargoes(column):
    frame = forecasts()
    selected = frame["embargo_sessions"].eq(1)
    if column == "actual_return":
        frame.loc[selected, column] = 0.01
    elif column == "query_session":
        frame.loc[selected, column] = frame.loc[selected, column].map(
            lambda value: value + pd.Timedelta(days=1)
        )
    elif column == "fold":
        frame.loc[selected, column] += 2
    else:
        frame.loc[selected, column] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match=column):
        sensitivity_metrics(frame)


def test_sensitivity_rejects_duplicate_forecasts():
    frame = forecasts()
    with pytest.raises(ValueError, match="duplicate"):
        sensitivity_metrics(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))


@pytest.mark.parametrize("column,value", [("fold", 0), ("fold", 1.5), ("embargo_sessions", -1)])
def test_evaluation_identities_cannot_alias_aggregate_or_invalid_groups(column, value):
    frame = forecasts()
    frame[column] = value
    with pytest.raises(ValueError, match=column):
        sensitivity_metrics(frame)
