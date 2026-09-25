"""Exchange-calendar bar grids; all canonical timestamps label bar availability."""

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from .validation import DataValidationError, validate_timestamps


def _aware_timestamp(value: pd.Timestamp, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise DataValidationError(f"{name} must be a nonmissing timezone-aware timestamp")
    return timestamp.tz_convert("UTC")


def expected_bar_grid(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    calendar: str = "XNYS",
    bar_minutes: int = 5,
) -> pd.DataFrame:
    """Return full sessions between exchange-local input dates, never truncated edges.

    Timestamps are bar ends. Only complete intervals entirely inside a trading
    segment are included, so no bar straddles a lunch break or session close.
    Session ordinals include scheduled sessions with no observed market data.
    """
    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive")
    start = _aware_timestamp(start, "start")
    end = _aware_timestamp(end, "end")
    if start > end:
        raise DataValidationError("start must not follow end")
    # Padding covers neighboring local dates without relying on a calendar's
    # default (rolling) date range. This also allows historical Parquet inputs.
    exchange = xcals.get_calendar(
        calendar,
        start=(start - pd.Timedelta(days=7)).date().isoformat(),
        end=(end + pd.Timedelta(days=7)).date().isoformat(),
    )
    first = start.tz_convert(exchange.tz).tz_localize(None).normalize()
    last = end.tz_convert(exchange.tz).tz_localize(None).normalize()
    scheduled = exchange.sessions_in_range(first, last)
    interval = pd.Timedelta(minutes=bar_minutes)
    frames = []
    for ordinal, session in enumerate(scheduled):
        opening = exchange.session_open(session)
        closing = exchange.session_close(session)
        break_start = exchange.session_break_start(session)
        break_end = exchange.session_break_end(session)
        segments = (
            [(opening, closing)]
            if pd.isna(break_start)
            else [(opening, break_start), (break_end, closing)]
        )
        date = pd.Timestamp(session).tz_localize(None).normalize()
        for segment_open, segment_close in segments:
            stamps = pd.date_range(segment_open + interval, segment_close, freq=interval)
            frames.append(
                pd.DataFrame(
                    {
                        "timestamp": stamps,
                        "session": date,
                        "session_index": ordinal,
                        "minute_of_session": (
                            (stamps - interval - opening) / pd.Timedelta(minutes=1)
                        ).astype("int64"),
                    }
                )
            )
    if not frames:
        return pd.DataFrame(
            {
                "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
                "session": pd.Series(dtype="datetime64[ns]"),
                "session_index": pd.Series(dtype="int64"),
                "minute_of_session": pd.Series(dtype="int64"),
            }
        )
    grid = pd.concat(frames, ignore_index=True)
    grid["timestamp"] = grid["timestamp"].astype("datetime64[ns, UTC]")
    return grid


def attach_session_metadata(
    bars: pd.DataFrame, grid: pd.DataFrame, bar_minutes: int
) -> pd.DataFrame:
    """Attach calendar identities and break contiguous segments at every gap."""
    positions = pd.DatetimeIndex(grid["timestamp"]).get_indexer(bars["timestamp"])
    if (positions < 0).any():
        raise DataValidationError(
            "bars are outside regular trading hours or not exactly interval-aligned"
        )
    result = bars.copy(deep=True).reset_index(drop=True)
    for column in ("session", "session_index", "minute_of_session"):
        result[column] = grid[column].iloc[positions].reset_index(drop=True)
    boundaries = result["session"].ne(result["session"].shift()) | result["timestamp"].diff().ne(
        pd.Timedelta(minutes=bar_minutes)
    )
    result["segment_id"] = boundaries.cumsum().astype("int64") - 1
    return result


def filter_regular_hours(
    bars: pd.DataFrame,
    *,
    calendar: str = "XNYS",
    bar_minutes: int = 5,
    timestamp_label: str = "end",
) -> pd.DataFrame:
    """Explicit opt-in filtering; preserve vendor labels and never mutate input.

    The strict ingestion pipeline does not call this helper: it rejects every
    off-grid row instead of concealing it. This helper also excludes misaligned
    intervals, not just timestamps falling somewhere during regular hours.
    """
    validate_timestamps(bars)
    if timestamp_label not in ("start", "end"):
        raise ValueError("timestamp_label must be 'start' or 'end'")
    stamps = bars["timestamp"].dt.tz_convert("UTC")
    if timestamp_label == "start":
        stamps = stamps + pd.Timedelta(minutes=bar_minutes)
    grid = expected_bar_grid(
        stamps.iloc[0], stamps.iloc[-1], calendar=calendar, bar_minutes=bar_minutes
    )
    keep = np.asarray(stamps.isin(grid["timestamp"]))
    return bars.loc[keep].copy(deep=True).reset_index(drop=True)
