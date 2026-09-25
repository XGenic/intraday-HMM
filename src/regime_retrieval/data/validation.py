"""Strict market-bar validation and JSON-serializable quality reports."""

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from regime_retrieval.config import ResearchConfig

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


@dataclass
class DataQualityReport:
    number_of_bars: int
    date_range: dict[str, str | None] = field(default_factory=lambda: {"start": None, "end": None})
    number_of_sessions: int = 0
    scheduled_session_count: int = 0
    missing_bars_by_session: dict[str, int] = field(default_factory=dict)
    duplicate_count: int = 0
    largest_timestamp_gaps: list[dict] = field(default_factory=list)
    ohlc_consistency_violations: int = 0
    session_coverage: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class DataValidationError(ValueError):
    """Malformed market data, with its quality report whenever one is available."""

    def __init__(self, message: str, report: DataQualityReport | None = None):
        super().__init__(message)
        self.report = report


def validate_timestamps(bars: pd.DataFrame, report: DataQualityReport | None = None) -> None:
    """Require a nonempty, timezone-aware, strictly increasing timestamp column."""
    if report is None:
        report = DataQualityReport(number_of_bars=len(bars))
    errors = []
    if "timestamp" not in bars.columns:
        errors.append("missing required column: timestamp")
    else:
        stamps = bars["timestamp"]
        report.duplicate_count = int(stamps.duplicated().sum())
        if stamps.isna().any():
            errors.append("timestamps contain NaT or missing values")
        if not isinstance(stamps.dtype, pd.DatetimeTZDtype):
            errors.append("timestamps must be timezone-aware datetime values")
        else:
            valid = stamps.dropna()
            if not valid.empty:
                report.date_range = {
                    "start": valid.min().tz_convert("UTC").isoformat(),
                    "end": valid.max().tz_convert("UTC").isoformat(),
                }
            if not stamps.is_monotonic_increasing:
                errors.append("timestamps must be strictly ordered")
        if report.duplicate_count:
            errors.append(f"duplicate timestamps: {report.duplicate_count}")
    if bars.empty:
        errors.append("no market bars supplied")
    if errors:
        report.errors.extend(errors)
        raise DataValidationError("; ".join(errors), report)


def _numeric_values(bars: pd.DataFrame, name: str, report: DataQualityReport) -> np.ndarray | None:
    series = bars[name]
    if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_complex_dtype(
        series.dtype
    ):
        report.errors.append(f"{name} must contain real numeric values")
        return None
    return series.to_numpy(dtype=float, na_value=np.nan)


def validate_bars(
    bars: pd.DataFrame, config: "ResearchConfig"
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Validate without filling or sorting, returning a detached canonical bar frame."""
    from .sessions import attach_session_metadata, expected_bar_grid

    report = DataQualityReport(number_of_bars=len(bars))
    if not bars.columns.is_unique:
        report.errors.append("column names must be unique")
        raise DataValidationError(report.errors[0], report)
    missing = [column for column in REQUIRED_COLUMNS if column not in bars.columns]
    if missing:
        report.errors.append(f"missing required columns: {', '.join(missing)}")
        raise DataValidationError(report.errors[0], report)
    validate_timestamps(bars, report)
    prepared = bars.copy(deep=True).reset_index(drop=True)
    prepared["timestamp"] = prepared["timestamp"].dt.tz_convert("UTC").astype("datetime64[ns, UTC]")
    if config.data.timestamp_label == "start":
        prepared["timestamp"] = prepared["timestamp"] + pd.Timedelta(minutes=config.bar_minutes)
    report.date_range = {
        "start": prepared["timestamp"].iloc[0].isoformat(),
        "end": prepared["timestamp"].iloc[-1].isoformat(),
    }
    differences = prepared["timestamp"].diff()
    largest = differences.dropna().nlargest(10).index
    report.largest_timestamp_gaps = [
        {
            "start": prepared.at[int(index) - 1, "timestamp"].isoformat(),
            "end": prepared.at[int(index), "timestamp"].isoformat(),
            "minutes": float(differences.loc[index] / pd.Timedelta(minutes=1)),
        }
        for index in largest
    ]

    values = {name: _numeric_values(prepared, name, report) for name in REQUIRED_COLUMNS[1:]}
    for name, data in values.items():
        if data is None:
            continue
        invalid = ~np.isfinite(data) | (data < 0 if name == "volume" else data <= 0)
        if invalid.any():
            requirement = "finite nonnegative" if name == "volume" else "finite positive"
            report.errors.append(f"{name} must be {requirement}: {int(invalid.sum())} invalid bars")
    if all(values[name] is not None for name in ("open", "high", "low", "close")):
        opening, high, low, close = (values[name] for name in ("open", "high", "low", "close"))
        inconsistent = (
            (low > high) | (opening < low) | (opening > high) | (close < low) | (close > high)
        )
        report.ohlc_consistency_violations = int(inconsistent.sum())
        if report.ohlc_consistency_violations:
            report.errors.append(
                f"OHLC consistency violations: {report.ohlc_consistency_violations}"
            )
    if "vwap" in prepared:
        vwap = _numeric_values(prepared, "vwap", report)
        if vwap is not None and (~np.isfinite(vwap) | (vwap <= 0)).any():
            report.errors.append("vendor vwap must contain finite positive values")
    if "symbol" in prepared and not prepared["symbol"].eq(config.symbol).fillna(False).all():
        report.errors.append(f"bar symbol does not match configured symbol {config.symbol!r}")

    grid = expected_bar_grid(
        prepared["timestamp"].iloc[0],
        prepared["timestamp"].iloc[-1],
        calendar=config.data.calendar,
        bar_minutes=config.bar_minutes,
    )
    positions = pd.DatetimeIndex(grid["timestamp"]).get_indexer(prepared["timestamp"])
    matched = positions >= 0
    if not matched.all():
        report.errors.append(
            f"{int((~matched).sum())} bars are outside regular trading hours "
            "or not exactly interval-aligned"
        )
    expected_counts = grid.groupby("session").size()
    observed_counts = grid.iloc[positions[matched]].groupby("session").size()
    report.scheduled_session_count = len(expected_counts)
    report.number_of_sessions = len(observed_counts)
    for session, expected in expected_counts.items():
        key = session.date().isoformat()
        observed = int(observed_counts.get(session, 0))
        coverage = observed / int(expected)
        report.missing_bars_by_session[key] = int(expected) - observed
        report.session_coverage[key] = coverage
        if observed and coverage < config.data.min_session_coverage:
            report.errors.append(
                f"session {key} coverage {coverage:.3f} is below minimum "
                f"{config.data.min_session_coverage:.3f}"
            )
    if report.errors:
        raise DataValidationError("; ".join(report.errors), report)
    return attach_session_metadata(prepared, grid, config.bar_minutes), report


def load_and_validate(config: "ResearchConfig") -> tuple[pd.DataFrame, DataQualityReport]:
    """Read the configured local Parquet file and apply the full RTH data contract."""
    from .providers import ParquetBarDataProvider

    provider = ParquetBarDataProvider(config.data.path, symbol=config.symbol)
    return validate_bars(provider.load(config.symbol), config)
