"""Vendor-independent provider contract and the local, read-only Parquet source."""

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from .sessions import _aware_timestamp
from .validation import DataQualityReport, DataValidationError, validate_timestamps


@runtime_checkable
class BarDataProvider(Protocol):
    def load(
        self,
        symbol: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Load inclusive vendor-labeled timestamps without sorting or filling."""
        ...


class ParquetBarDataProvider:
    """A single-symbol file, explicitly bound to its configured symbol.

    Time bounds apply to the file's original timestamp labels. Conversion from
    bar starts to availability times belongs to validation, not the provider.
    """

    def __init__(self, path: str | Path, symbol: str):
        self.path = Path(path)
        self.symbol = symbol

    def load(
        self,
        symbol: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        if symbol != self.symbol:
            raise DataValidationError(
                f"requested symbol {symbol!r} does not match configured symbol {self.symbol!r}"
            )
        lower = _aware_timestamp(start, "start") if start is not None else None
        upper = _aware_timestamp(end, "end") if end is not None else None
        if lower is not None and upper is not None and lower > upper:
            raise DataValidationError("start must not follow end")
        bars = pd.read_parquet(self.path)
        if not bars.columns.is_unique:
            report = DataQualityReport(
                number_of_bars=len(bars), errors=["column names must be unique"]
            )
            raise DataValidationError(report.errors[0], report)
        validate_timestamps(bars)
        if "symbol" in bars and not bars["symbol"].eq(self.symbol).fillna(False).all():
            report = DataQualityReport(
                number_of_bars=len(bars),
                errors=[f"bar symbol does not match configured symbol {self.symbol!r}"],
            )
            raise DataValidationError(report.errors[0], report)
        selected = pd.Series(True, index=bars.index)
        if lower is not None:
            selected &= bars["timestamp"] >= lower
        if upper is not None:
            selected &= bars["timestamp"] <= upper
        result = bars.loc[selected].copy(deep=True).reset_index(drop=True)
        result["timestamp"] = result["timestamp"].dt.tz_convert("UTC").astype("datetime64[ns, UTC]")
        return result
