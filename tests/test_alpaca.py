import json

import pandas as pd
import pytest

from regime_retrieval.data import alpaca
from regime_retrieval.data.sessions import expected_bar_grid


def vendor_bars(start, end):
    grid = expected_bar_grid(pd.Timestamp(start), pd.Timestamp(end))
    return [
        {
            "t": (stamp - pd.Timedelta(minutes=5)).isoformat(),
            "o": 100.0,
            "h": 102.0,
            "l": 99.0,
            "c": 101.0,
            "v": 10,
            "vw": 100.3,
            "n": 2,
        }
        for stamp in grid.timestamp
    ]


def test_paginated_download_rth_vwap_provenance_and_cache(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("APCA_API_KEY_ID=test-id\nAPCA_API_SECRET_KEY=test-secret\n")
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    bars = vendor_bars("2024-11-29 14:30Z", "2024-11-29 18:00Z")
    bars.append({**bars[-1], "t": "2024-11-29T18:00:00Z"})
    requests = []

    def fetch(url, headers):
        requests.append(url)
        assert headers["APCA-API-SECRET-KEY"] == "test-secret"
        second = "page_token=" in url
        return json.dumps(
            {
                "symbol": "SPY",
                "bars": bars[20:] if second else bars[:20],
                "next_page_token": None if second else "page+two",
            }
        ).encode()

    monkeypatch.setattr(alpaca, "request_page", fetch)
    args = dict(
        symbol="SPY",
        start="2024-11-29",
        end="2024-11-29",
        archive=tmp_path / "archive",
        env_file=env,
    )
    output = tmp_path / "bars.parquet"
    result = alpaca.download(output=output, **args)
    saved = pd.read_parquet(output)
    assert len(saved) == 42
    assert saved.timestamp.iloc[0] == pd.Timestamp("2024-11-29 14:30Z")
    assert saved.timestamp.iloc[-1] == pd.Timestamp("2024-11-29 17:55Z")
    assert saved.vwap.eq(100.3).all()
    assert result["excluded_non_rth_or_off_grid"] == 1
    assert result["missing_bars_by_requested_session"] == {}
    assert len(requests) == 2
    assert "feed=sip" in requests[0] and "adjustment=raw" in requests[0]
    assert "page_token=page%2Btwo" in requests[1]
    assert "test-secret" not in output.with_suffix(".manifest.json").read_text()
    alpaca.download(output=tmp_path / "cached.parquet", **args)
    assert len(requests) == 2
    with pytest.raises(FileExistsError):
        alpaca.download(output=output, **args)
    raw = next((tmp_path / "archive").rglob("page-0001.json"))
    raw.write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        alpaca.download(output=tmp_path / "corrupt.parquet", **args)


def test_duplicate_pages_fail_and_missing_edge_session_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "test-id")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-secret")
    bars = vendor_bars("2024-03-11 13:30Z", "2024-03-11 20:00Z")
    monkeypatch.setattr(
        alpaca,
        "request_page",
        lambda *_: json.dumps({"symbol": "SPY", "bars": bars, "next_page_token": None}).encode(),
    )
    result = alpaca.download(
        symbol="SPY",
        start="2024-03-08",
        end="2024-03-11",
        output=tmp_path / "edge.parquet",
        archive=tmp_path / "edge",
        env_file=tmp_path / "absent",
    )
    assert result["missing_bars_by_requested_session"] == {"2024-03-08": 78}
    bars.append(bars[-1])
    with pytest.raises(ValueError, match="duplicate timestamps"):
        alpaca.download(
            symbol="SPY",
            start="2024-03-11",
            end="2024-03-11",
            output=tmp_path / "duplicate.parquet",
            archive=tmp_path / "duplicate",
            env_file=tmp_path / "absent",
        )


def test_repeated_token_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "test-id")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-secret")
    monkeypatch.setattr(
        alpaca,
        "request_page",
        lambda *_: json.dumps(
            {"symbol": "SPY", "bars": [], "next_page_token": "repeated"}
        ).encode(),
    )
    with pytest.raises(ValueError, match="Repeated Alpaca pagination token"):
        alpaca.download(
            symbol="SPY",
            start="2024-03-11",
            end="2024-03-11",
            output=tmp_path / "bars.parquet",
            archive=tmp_path / "raw",
            env_file=tmp_path / "absent",
        )


def test_requested_edge_audit_does_not_extend_past_spring_dst_sunday(tmp_path, monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "test-id")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-secret")
    bars = vendor_bars("2024-03-08 14:30Z", "2024-03-08 21:00Z")
    monkeypatch.setattr(
        alpaca,
        "request_page",
        lambda *_: json.dumps({"symbol": "SPY", "bars": bars, "next_page_token": None}).encode(),
    )
    result = alpaca.download(
        symbol="SPY",
        start="2024-03-08",
        end="2024-03-10",
        output=tmp_path / "bars.parquet",
        archive=tmp_path / "archive",
        env_file=tmp_path / "absent",
    )
    assert result["requested_session_count"] == 1
    assert result["missing_bars_by_requested_session"] == {}
