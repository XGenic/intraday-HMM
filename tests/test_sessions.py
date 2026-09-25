import pandas as pd
import pytest

from regime_retrieval.config import ResearchConfig
from regime_retrieval.data import (
    DataValidationError,
    expected_bar_grid,
    filter_regular_hours,
    validate_bars,
)


def test_holiday_is_absent_and_black_friday_closes_early():
    grid = expected_bar_grid(
        pd.Timestamp("2024-11-27 00:00", tz="America/New_York"),
        pd.Timestamp("2024-11-29 23:00", tz="America/New_York"),
    )
    counts = grid.groupby("session").size().to_dict()
    assert counts == {pd.Timestamp("2024-11-27"): 78, pd.Timestamp("2024-11-29"): 42}
    shortened = grid.loc[grid.session.eq(pd.Timestamp("2024-11-29"))]
    assert shortened.timestamp.iloc[0] == pd.Timestamp("2024-11-29 14:35Z")
    assert shortened.timestamp.iloc[-1] == pd.Timestamp("2024-11-29 18:00Z")
    assert shortened.session_index.unique().tolist() == [1]
    assert shortened.minute_of_session.iloc[-1] == 205
    bars = shortened[["timestamp"]].assign(
        open=100.0, high=101.0, low=99.0, close=100.0, volume=1.0
    )
    _, report = validate_bars(bars, ResearchConfig())
    assert report.missing_bars_by_session == {"2024-11-29": 0}
    assert report.session_coverage == {"2024-11-29": 1.0}


def test_dst_moves_utc_open_without_changing_local_slots():
    grid = expected_bar_grid(pd.Timestamp("2024-03-08 14:35Z"), pd.Timestamp("2024-03-11 20:00Z"))
    first_bars = grid.groupby("session").first()
    last_bars = grid.groupby("session").last()
    assert first_bars.timestamp.tolist() == [
        pd.Timestamp("2024-03-08 14:35Z"),
        pd.Timestamp("2024-03-11 13:35Z"),
    ]
    assert last_bars.timestamp.tolist() == [
        pd.Timestamp("2024-03-08 21:00Z"),
        pd.Timestamp("2024-03-11 20:00Z"),
    ]
    assert grid.groupby("session").size().tolist() == [78, 78]
    assert first_bars.minute_of_session.tolist() == [0, 0]
    assert last_bars.minute_of_session.tolist() == [385, 385]


def test_full_edge_grid_is_not_truncated_to_input_times():
    grid = expected_bar_grid(pd.Timestamp("2024-01-03 16:00Z"), pd.Timestamp("2024-01-03 17:00Z"))
    assert len(grid) == 78
    assert grid.timestamp.iloc[0] == pd.Timestamp("2024-01-03 14:35Z")
    assert grid.timestamp.iloc[-1] == pd.Timestamp("2024-01-03 21:00Z")


def test_explicit_filter_keeps_aligned_rth_rows_without_mutating_start_labels():
    raw = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-11-28 14:30Z",  # Thanksgiving.
                    "2024-11-29 14:25Z",  # Before open.
                    "2024-11-29 14:30Z",  # First start-labeled bar.
                    "2024-11-29 14:31Z",  # Misaligned.
                    "2024-11-29 17:55Z",  # Final bar before early close.
                    "2024-11-29 18:00Z",  # Starts at the close, so unavailable during RTH.
                ]
            ),
            "vendor_id": list(range(6)),
        }
    )
    original = raw.copy(deep=True)
    filtered = filter_regular_hours(raw, timestamp_label="start")
    assert filtered.vendor_id.tolist() == [2, 4]
    assert filtered.timestamp.tolist() == raw.timestamp.iloc[[2, 4]].tolist()
    pd.testing.assert_frame_equal(raw, original)


def test_holiday_only_bars_fail_rather_than_inventing_a_session():
    bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-11-28 15:00Z"]),
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1.0],
        }
    )
    with pytest.raises(DataValidationError, match="outside regular trading hours") as caught:
        validate_bars(bars, ResearchConfig())
    assert caught.value.report.scheduled_session_count == 0
    assert caught.value.report.missing_bars_by_session == {}
