import numpy as np
import pandas as pd

from regime_retrieval.forecasting.labels import forward_returns
from regime_retrieval.retrieval.eligibility import EligibleCandidateIndex
from regime_retrieval.retrieval.representations import ContextIndex, build_contexts


def prepared_bars(session_dates, session_ordinals, bars_per_session=16):
    frames = []
    for segment, (date, ordinal) in enumerate(zip(session_dates, session_ordinals, strict=True)):
        minutes = np.arange(bars_per_session) * 5
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": pd.Timestamp(date, tz="UTC")
                    + pd.Timedelta(hours=14, minutes=35)
                    + pd.to_timedelta(minutes, unit="m"),
                    "session": pd.Timestamp(date),
                    "session_index": ordinal,
                    "minute_of_session": minutes,
                    "segment_id": segment,
                    "close": 100 + np.arange(bars_per_session, dtype=float),
                    "feature": np.arange(bars_per_session, dtype=float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_maximum_outcome_must_be_known_and_equality_at_query_is_available():
    bars = prepared_bars(["2024-01-02"], [10], bars_per_session=18)
    labels = forward_returns(bars, [5, 15, 30, 60], bar_minutes=5)
    contexts = build_contexts(bars, ["feature"], context_bars=2)
    candidates = EligibleCandidateIndex(bars, labels, contexts, max_horizon=60)
    query = bars.timestamp.iloc[14]
    # The contexts ending at positions 1 and 2 complete their 60-minute
    # outcomes at positions 13 and 14. Later contexts have shorter labels
    # available, but their maximum-horizon outcomes are still in the future.
    np.testing.assert_array_equal(candidates.eligible(query, query_session=10, embargo=0), [0, 1])
    np.testing.assert_array_equal(
        candidates.eligible(query - pd.Timedelta(nanoseconds=1), query_session=10, embargo=0),
        [0],
    )
    assert candidates.eligible(bars.timestamp.iloc[12], query_session=10, embargo=0).size == 0
    # Even long after the data ends, the final contexts cannot acquire labels
    # that never existed; only context ends 1 through 5 have full outcomes.
    np.testing.assert_array_equal(
        candidates.eligible(query + pd.Timedelta(days=10), query_session=20, embargo=0),
        [0, 1, 2, 3, 4],
    )


def test_candidate_end_must_be_strictly_before_query_independently_of_outcome_cutoff():
    bars = prepared_bars(["2024-01-02"], [10], bars_per_session=2)
    query = bars.timestamp.iloc[1]
    contexts = ContextIndex(np.array([0, 1]), np.array([[0.0], [1.0]]))
    # Deliberately degenerate outcome metadata isolates the strict candidate-end
    # guard: outcome availability alone must not authorize the query itself.
    labels = pd.DataFrame({"outcome_end_60m": [query, query]})
    candidates = EligibleCandidateIndex(bars, labels, contexts, max_horizon=60)
    np.testing.assert_array_equal(candidates.eligible(query, query_session=10, embargo=0), [0])


def test_missing_close_outcomes_preserve_original_context_rows_across_sessions():
    bars = prepared_bars(["2024-01-02", "2024-01-03"], [10, 11])
    labels = forward_returns(bars, [5, 15, 30, 60], bar_minutes=5)
    contexts = build_contexts(bars, ["feature"], context_bars=2)
    candidates = EligibleCandidateIndex(bars, labels, contexts, max_horizon=60)
    eligible = candidates.eligible(bars.timestamp.iloc[30], query_session=11, embargo=0)
    # Contexts ending near the first session's close lack a 60-minute label.
    # Removing those holes must not renumber the surviving context-row IDs.
    np.testing.assert_array_equal(eligible, [0, 1, 2, 15, 16])
    np.testing.assert_array_equal(contexts.end_positions[eligible], [1, 2, 3, 17, 18])


def test_positive_embargo_excludes_current_and_n_preceding_calendar_sessions():
    # Exchange sessions 12 and 13 (January 4 and 5) are wholly absent from data.
    bars = prepared_bars(
        ["2024-01-02", "2024-01-03", "2024-01-08", "2024-01-09", "2024-01-10"],
        [10, 11, 14, 15, 16],
        bars_per_session=14,
    )
    labels = forward_returns(bars, [5, 15, 30, 60], bar_minutes=5)
    contexts = build_contexts(bars, ["feature"], context_bars=1)
    candidates = EligibleCandidateIndex(bars, labels, contexts, max_horizon=60)
    query = bars.timestamp.iloc[-1]
    np.testing.assert_array_equal(
        candidates.eligible(query, query_session=16, embargo=0),
        [0, 1, 14, 15, 28, 29, 42, 43, 56, 57],
    )
    np.testing.assert_array_equal(
        candidates.eligible(query, query_session=16, embargo=1), [0, 1, 14, 15, 28, 29]
    )
    np.testing.assert_array_equal(
        candidates.eligible(query, query_session=16, embargo=2), [0, 1, 14, 15]
    )
    # For query session 14, the previous exchange session is missing session 13,
    # not observed session 11. An observed-session-count embargo over-purges 11.
    np.testing.assert_array_equal(
        candidates.eligible(bars.timestamp.iloc[41], query_session=14, embargo=1),
        [0, 1, 14, 15],
    )
