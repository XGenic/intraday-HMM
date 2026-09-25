import json

import numpy as np
import pandas as pd
import pytest

from regime_retrieval.config import ResearchConfig
from regime_retrieval.data import (
    DataValidationError,
    ParquetBarDataProvider,
    load_and_validate,
    validate_bars,
)


def bars_for_day(day="2024-01-03", *, start_label=False):
    stamps = pd.date_range(f"{day} 09:35", f"{day} 16:00", freq="5min", tz="America/New_York")
    if start_label:
        stamps -= pd.Timedelta(minutes=5)
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.0,
            "volume": 10.0,
            "trade_count": np.arange(len(stamps)),
        }
    )


def test_parquet_inclusive_bounds_and_read_only_optional_columns(tmp_path):
    original = bars_for_day()
    path = tmp_path / "SPY.parquet"
    original.to_parquet(path, index=False)
    raw_bytes = path.read_bytes()
    provider = ParquetBarDataProvider(path, symbol="SPY")
    selected = provider.load("SPY", original.timestamp.iloc[2], original.timestamp.iloc[4])
    assert selected.trade_count.tolist() == [2, 3, 4]
    assert selected.timestamp.tolist() == original.timestamp.iloc[2:5].tolist()
    selected.loc[0, "open"] = 1
    assert path.read_bytes() == raw_bytes
    pd.testing.assert_frame_equal(pd.read_parquet(path), original)
    with pytest.raises(DataValidationError, match="symbol"):
        provider.load("QQQ")
    with pytest.raises(DataValidationError, match="timezone-aware"):
        provider.load("SPY", pd.Timestamp("2024-01-03"))


def test_parquet_rejects_wrong_embedded_symbol_and_hidden_disorder(tmp_path):
    bars = bars_for_day()
    path = tmp_path / "bars.parquet"
    bars.assign(symbol="QQQ").to_parquet(path, index=False)
    provider = ParquetBarDataProvider(path, "SPY")
    with pytest.raises(DataValidationError, match="symbol"):
        provider.load("SPY")
    bars.iloc[[1, 0, *range(2, len(bars))]].to_parquet(path, index=False)
    with pytest.raises(DataValidationError, match="ordered"):
        provider.load("SPY", start=bars.timestamp.iloc[3])


def test_load_converts_start_labels_without_mutating_vendor_data(tmp_path):
    original = bars_for_day(start_label=True)
    path = tmp_path / "bars.parquet"
    original.to_parquet(path, index=False)
    config = ResearchConfig(data={"path": path, "timestamp_label": "start"})
    prepared, report = load_and_validate(config)
    assert prepared.timestamp.iloc[0] == pd.Timestamp("2024-01-03 14:35Z")
    assert prepared.timestamp.iloc[-1] == pd.Timestamp("2024-01-03 21:00Z")
    assert prepared.minute_of_session.tolist() == list(range(0, 390, 5))
    assert prepared.session.tolist() == [pd.Timestamp("2024-01-03")] * 78
    assert isinstance(prepared.index, pd.RangeIndex)
    assert report.session_coverage == {"2024-01-03": 1.0}
    assert report.missing_bars_by_session == {"2024-01-03": 0}
    json.dumps(report.to_dict(), allow_nan=False)
    pd.testing.assert_frame_equal(pd.read_parquet(path), original)


def test_missing_bars_and_entire_sessions_remain_visible_without_fill():
    first = bars_for_day().drop(index=20)
    last = bars_for_day("2024-01-05")
    raw = pd.concat([first, last], ignore_index=True)
    snapshot = raw.copy(deep=True)
    prepared, report = validate_bars(raw, ResearchConfig())
    assert len(prepared) == 155
    assert report.missing_bars_by_session == {
        "2024-01-03": 1,
        "2024-01-04": 78,
        "2024-01-05": 0,
    }
    assert report.number_of_sessions == 2
    assert report.scheduled_session_count == 3
    assert prepared.session_index.unique().tolist() == [0, 2]
    assert prepared.segment_id.iloc[19:21].tolist() == [0, 1]
    assert prepared.segment_id.iloc[77] == 2
    assert report.largest_timestamp_gaps[0]["minutes"] > 24 * 60
    pd.testing.assert_frame_equal(raw, snapshot)


def test_truncated_edge_sessions_use_full_grid_and_fail_coverage():
    raw = bars_for_day().iloc[20:]
    with pytest.raises(DataValidationError, match="coverage") as caught:
        validate_bars(raw, ResearchConfig())
    assert caught.value.report.missing_bars_by_session == {"2024-01-03": 20}
    assert caught.value.report.session_coverage["2024-01-03"] == pytest.approx(58 / 78)
    single = bars_for_day().iloc[:1]
    with pytest.raises(DataValidationError, match="coverage") as caught:
        validate_bars(single, ResearchConfig())
    assert caught.value.report.largest_timestamp_gaps == []


def test_timestamps_reject_missing_naive_nat_unsorted_and_duplicates():
    bars = bars_for_day()
    invalid_cases = [
        (bars.drop(columns="timestamp"), "missing required"),
        (bars.assign(timestamp=bars.timestamp.dt.tz_localize(None)), "timezone-aware"),
        (bars.iloc[::-1], "ordered"),
        (pd.concat([bars.iloc[:1], bars], ignore_index=True), "duplicate"),
    ]
    nat = bars.copy()
    nat.loc[2, "timestamp"] = pd.NaT
    invalid_cases.append((nat, "NaT"))
    for invalid, message in invalid_cases:
        with pytest.raises(DataValidationError, match=message) as caught:
            validate_bars(invalid, ResearchConfig())
        assert caught.value.report is not None
    with pytest.raises(DataValidationError) as caught:
        validate_bars(invalid_cases[3][0], ResearchConfig())
    assert caught.value.report.duplicate_count == 1


@pytest.mark.parametrize(
    "column,value,message",
    [
        ("open", 0.0, "finite positive"),
        ("close", np.inf, "finite positive"),
        ("low", np.nan, "finite positive"),
        ("volume", -1.0, "finite nonnegative"),
        ("volume", np.inf, "finite nonnegative"),
        ("high", 99.5, "OHLC consistency"),
        ("vwap", 0.0, "vendor vwap"),
        ("vwap", np.nan, "vendor vwap"),
    ],
)
def test_invalid_prices_volume_and_vendor_vwap_fail(column, value, message):
    bars = bars_for_day()
    if column == "vwap":
        bars["vwap"] = 100.0
    bars.loc[0, column] = value
    with pytest.raises(DataValidationError, match=message) as caught:
        validate_bars(bars, ResearchConfig())
    json.dumps(caught.value.report.to_dict(), allow_nan=False)


def test_zero_volume_and_session_vwap_outside_current_bar_are_usable():
    bars = bars_for_day().assign(volume=0.0, vwap=98.0)
    prepared, report = validate_bars(bars, ResearchConfig())
    assert prepared.volume.eq(0).all()
    assert prepared.vwap.eq(98).all()
    assert report.ohlc_consistency_violations == 0


def test_off_grid_rows_are_not_silently_filtered():
    bars = bars_for_day()
    extra = bars.iloc[:1].copy()
    extra["timestamp"] -= pd.Timedelta(minutes=5)
    bars = pd.concat([extra, bars], ignore_index=True)
    with pytest.raises(DataValidationError, match="outside regular trading hours"):
        validate_bars(bars, ResearchConfig())
    misaligned = bars_for_day()
    misaligned.loc[1, "timestamp"] += pd.Timedelta(seconds=1)
    with pytest.raises(DataValidationError, match="interval-aligned"):
        validate_bars(misaligned, ResearchConfig())
