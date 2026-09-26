"""Repeatable, resumable historical SIP downloads; never save authentication headers.

Run with ``python -m regime_retrieval.data.alpaca --help``. Date bounds are
inclusive exchange-local dates. Raw responses and their request/checksum sidecars
are immutable; reruns verify and reuse them rather than silently refreshing data.
"""

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from regime_retrieval.config import ResearchConfig
from regime_retrieval.data.sessions import expected_bar_grid, filter_regular_hours
from regime_retrieval.data.validation import validate_bars, validate_timestamps
from regime_retrieval.reporting.artifacts import write_json

ENDPOINT = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"


def credentials(env_file: Path) -> dict[str, str]:
    """Read only the two required keys, preferring the process environment."""
    names = ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY")
    values = {name: os.environ.get(name, "") for name in names}
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            key, separator, value = line.strip().removeprefix("export ").partition("=")
            key = key.strip()
            if separator and key in values and not values[key]:
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                values[key] = value
    if not all(values.values()):
        raise ValueError("Supply APCA_API_KEY_ID and APCA_API_SECRET_KEY via environment or .env")
    return {
        "APCA-API-KEY-ID": values[names[0]],
        "APCA-API-SECRET-KEY": values[names[1]],
    }


def request_page(url: str, headers: dict[str, str]) -> bytes:
    for attempt in range(6):
        try:
            with urlopen(Request(url, headers=headers), timeout=60) as response:
                return response.read()
        except HTTPError as exc:
            # Do not echo response bodies or request headers into logs.
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise OSError(f"Alpaca historical bars returned HTTP {exc.code}") from None
            retry = exc.headers.get("Retry-After", "")
            delay = min(float(retry), 60) if retry.isdigit() else min(2**attempt, 30)
        except (URLError, TimeoutError):
            if attempt == 5:
                raise OSError(
                    "Alpaca historical bars network request failed after retries"
                ) from None
            delay = min(2**attempt, 30)
        time.sleep(delay)
    raise RuntimeError("unreachable")


def cached_page(
    directory: Path, page: int, endpoint: str, params: dict, headers: dict
) -> tuple[dict, dict]:
    raw_path = directory / f"page-{page:04d}.json"
    meta_path = directory / f"page-{page:04d}.request.json"
    request = {"endpoint": endpoint, "parameters": params}
    if raw_path.exists() or meta_path.exists():
        if not (raw_path.exists() and meta_path.exists()):
            raise ValueError(f"Incomplete cached page: {raw_path}; use a new archive directory")
        raw = raw_path.read_bytes()
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata["request"] != request or metadata["sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValueError(f"Cached request/checksum mismatch: {raw_path}")
    else:
        raw = request_page(endpoint + "?" + urlencode(params), headers)
        payload = json.loads(raw)
        if not isinstance(payload.get("bars"), (list, type(None))) or "bars" not in payload:
            raise ValueError("Alpaca response lacks a bars array")
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "request": request,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "response_file": str(raw_path.resolve()),
        }
        with raw_path.open("xb") as handle:
            handle.write(raw)
        with meta_path.open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
    return json.loads(raw), metadata


def download(
    *, symbol: str, start: str, end: str, output: Path, archive: Path, env_file: Path
) -> dict:
    if not re.fullmatch(r"[A-Z][A-Z0-9.]{0,14}", symbol):
        raise ValueError("Use an uppercase stock symbol")
    first = pd.Timestamp(start, tz="America/New_York")
    last = pd.Timestamp(end, tz="America/New_York")
    if first != first.normalize() or last != last.normalize() or first > last:
        raise ValueError("start/end must be ordered calendar dates")
    stop = last + pd.DateOffset(days=1)
    if stop.tz_convert("UTC") > pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=15):
        raise ValueError("Historical SIP end must be at least 15 minutes old")
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite dataset/provenance: {output}")
    headers = credentials(env_file)
    endpoint = ENDPOINT.format(symbol=symbol)
    records, pages = [], []
    cursor = first
    while cursor < stop:
        next_month = cursor.normalize().replace(day=1) + pd.offsets.MonthBegin(1)
        boundary = min(next_month, stop)
        directory = archive / symbol / f"{cursor.date()}_{boundary.date()}"
        params = {
            "feed": "sip",
            "timeframe": "5Min",
            "adjustment": "raw",
            "sort": "asc",
            "limit": 10000,
            "asof": "-",
            "start": cursor.tz_convert("UTC").isoformat(),
            "end": (boundary.tz_convert("UTC") - pd.Timedelta(seconds=1)).isoformat(),
        }
        seen_tokens = set()
        page = 1
        before = len(records)
        while True:
            payload, metadata = cached_page(directory, page, endpoint, dict(params), headers)
            if payload.get("symbol") != symbol:
                raise ValueError("Alpaca response symbol mismatch")
            records.extend(payload.get("bars") or [])
            pages.append(metadata)
            token = payload.get("next_page_token")
            if not token:
                break
            if token in seen_tokens:
                raise ValueError("Repeated Alpaca pagination token")
            seen_tokens.add(token)
            params["page_token"] = token
            page += 1
        print(
            f"{cursor.date()} through {(boundary - pd.Timedelta(days=1)).date()}: "
            f"{len(records) - before:,} vendor bars, {page} page(s)",
            flush=True,
        )
        cursor = boundary
    frame = pd.DataFrame(records).rename(
        columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
            "n": "trade_count",
        }
    )
    required = ["timestamp", "open", "high", "low", "close", "volume", "vwap", "trade_count"]
    if not set(required).issubset(frame):
        raise ValueError("Alpaca archive is empty or lacks required OHLCV/VWAP fields")
    frame = frame[required]
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    validate_timestamps(frame)
    if not frame.timestamp.between(first, stop, inclusive="left").all():
        raise ValueError("Vendor returned bars outside requested dates")
    rth = filter_regular_hours(frame, timestamp_label="start")
    config = ResearchConfig()
    config.data.timestamp_label = "start"
    config.symbol = symbol
    canonical, quality = validate_bars(rth, config)
    # Audit requested edges too: the generic validator only sees observed dates.
    grid = expected_bar_grid(first, last)
    expected = grid.groupby("session").size()
    observed = canonical.groupby("session").size()
    missing = {
        str(day.date()): int(count - observed.get(day, 0))
        for day, count in expected.items()
        if count != observed.get(day, 0)
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    rth.to_parquet(output, index=False)
    manifest = {
        "vendor": "Alpaca",
        "symbol": symbol,
        "feed": "sip",
        "timeframe": "5Min",
        "adjustment": "raw",
        "asof": "-",
        "timestamp_label": "start",
        "calendar": "XNYS",
        "vwap_policy": "preserved vendor per-bar VWAP",
        "requested_start": start,
        "requested_end": end,
        "vendor_bar_count": len(frame),
        "rth_bar_count": len(rth),
        "excluded_non_rth_or_off_grid": len(frame) - len(rth),
        "requested_session_count": len(expected),
        "observed_session_count": len(observed),
        "missing_bars_by_requested_session": missing,
        "quality": quality.to_dict(),
        "output_file": str(output.resolve()),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "pages": pages,
    }
    write_json(output.with_suffix(".manifest.json"), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start", required=True, help="Inclusive exchange date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Inclusive exchange date YYYY-MM-DD")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/alpaca/archive"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    result = download(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "output_file",
                    "rth_bar_count",
                    "observed_session_count",
                    "missing_bars_by_requested_session",
                    "output_sha256",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
