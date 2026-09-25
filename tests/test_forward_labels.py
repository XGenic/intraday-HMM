import numpy as np
import pandas as pd
import pytest

from regime_retrieval.forecasting.labels import forward_returns, recent_return

HORIZONS = [5, 15, 30, 60]


def prepared_bars(close, sessions=None, minutes=None, segments=None):
    count = len(close)
    sessions = np.zeros(count, dtype=int) if sessions is None else np.asarray(sessions)
    if minutes is None:
        minutes = pd.Series(sessions).groupby(sessions).cumcount().to_numpy() * 5
    dates = pd.Timestamp("2024-01-02") + pd.to_timedelta(sessions, unit="D")
    return pd.DataFrame(
        {
            "timestamp": dates.tz_localize("UTC")
            + pd.Timedelta(hours=14, minutes=35)
            + pd.to_timedelta(minutes, unit="m"),
            "session": dates,
            "session_index": sessions,
            "minute_of_session": minutes,
            "segment_id": sessions if segments is None else segments,
            "close": np.asarray(close, dtype=float),
        }
    )


def test_hand_calculated_five_fifteen_thirty_and_sixty_minute_simple_returns():
    bars = prepared_bars([100, 105, 104, 90, 95, 98, 120, 119, 121, 122, 126, 125, 80])
    labels = forward_returns(bars, HORIZONS, bar_minutes=5)
    for horizon, expected in zip(HORIZONS, [0.05, -0.1, 0.2, -0.2], strict=True):
        assert labels.loc[0, f"return_{horizon}m"] == pytest.approx(expected)
        assert labels.loc[0, f"outcome_end_{horizon}m"] == (
            bars.timestamp.iloc[0] + pd.Timedelta(minutes=horizon)
        )
    # A second starting price rules out returning price differences or dividing
    # by a fixed session-open denominator instead of the candidate's close.
    assert labels.loc[1, "return_5m"] == pytest.approx(-1 / 105)
    assert labels.loc[1, "return_15m"] == pytest.approx(-10 / 105)
    assert labels.loc[1, "return_30m"] == pytest.approx(14 / 105)
    assert pd.isna(labels.loc[1, "return_60m"])
    assert pd.isna(labels.loc[1, "outcome_end_60m"])


def test_outcome_at_session_close_is_valid_but_next_session_is_never_a_continuation():
    # The first session's final observed bar ends at 21:00 UTC (16:00 New York).
    bars = prepared_bars(
        np.arange(100, 126),
        sessions=[0] * 13 + [1] * 13,
        minutes=list(range(325, 390, 5)) + list(range(0, 65, 5)),
    )
    labels = forward_returns(bars, HORIZONS, bar_minutes=5)
    for horizon in HORIZONS:
        last_start = 12 - horizon // 5
        assert labels.loc[last_start, f"outcome_end_{horizon}m"] == pd.Timestamp(
            "2024-01-02 21:00:00", tz="UTC"
        )
        assert labels.loc[last_start, f"return_{horizon}m"] == pytest.approx(
            112 / (100 + last_start) - 1
        )
        assert labels.loc[last_start + 1 : 12, f"return_{horizon}m"].isna().all()
        assert labels.loc[last_start + 1 : 12, f"outcome_end_{horizon}m"].isna().all()
        assert pd.isna(labels.loc[25, f"return_{horizon}m"])


def test_missing_bar_does_not_compress_time_and_contiguous_history_recovers():
    # The 15-minute slot is absent; every row after it belongs to a fresh segment.
    minutes = [0, 5, 10] + list(range(20, 90, 5))
    bars = prepared_bars(
        [100 + minute for minute in minutes],
        minutes=minutes,
        segments=[0] * 3 + [1] * 14,
    )
    labels = forward_returns(bars, HORIZONS, bar_minutes=5)
    assert labels.loc[1, "return_5m"] == pytest.approx(110 / 105 - 1)
    assert labels.loc[1, "outcome_end_5m"] == bars.timestamp.iloc[2]
    assert pd.isna(labels.loc[2, "return_5m"])
    assert pd.isna(labels.loc[2, "outcome_end_5m"])
    for horizon in [15, 30, 60]:
        assert labels.loc[:2, f"return_{horizon}m"].isna().all()
        assert labels.loc[:2, f"outcome_end_{horizon}m"].isna().all()
    for horizon in HORIZONS:
        assert labels.loc[3, f"return_{horizon}m"] == pytest.approx(horizon / 120)
        assert labels.loc[3, f"outcome_end_{horizon}m"] == (
            bars.timestamp.iloc[3] + pd.Timedelta(minutes=horizon)
        )


def test_recent_return_uses_full_context_lag_without_crossing_gap_or_session():
    bars = prepared_bars(
        [100, 110, 121, 133.1, 200, 210, 220, 240, 300, 320, 340, 360],
        sessions=[0] * 8 + [1] * 4,
        minutes=[0, 5, 10, 15, 25, 30, 35, 40, 0, 5, 10, 15],
        segments=[0] * 4 + [1] * 4 + [2] * 4,
    )
    result = recent_return(bars, context_bars=3)
    np.testing.assert_allclose(
        result,
        [np.nan, np.nan, np.nan, 0.331, np.nan, np.nan, np.nan, 0.2, np.nan, np.nan, np.nan, 0.2],
        equal_nan=True,
    )
