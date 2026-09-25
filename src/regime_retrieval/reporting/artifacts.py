"""Streaming audit records keep unconditional baseline history out of process memory."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding="utf-8")


class NeighborWriter:
    def __init__(self, path: Path, buffer_rows: int = 100_000):
        self.path = path
        self.buffer_rows = buffer_rows
        self.pending: list[pd.DataFrame] = []
        self.pending_rows = 0
        self.row_count = 0
        self.writer: pq.ParquetWriter | None = None

    def write(self, frame: pd.DataFrame) -> None:
        self.pending.append(frame)
        self.pending_rows += len(frame)
        if self.pending_rows >= self.buffer_rows:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        frame = pd.concat(self.pending, ignore_index=True)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self.writer.write_table(table)
        self.row_count += len(frame)
        self.pending.clear()
        self.pending_rows = 0

    def __enter__(self) -> "NeighborWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            if exc_type is None:
                self.flush()
        finally:
            if self.writer is not None:
                self.writer.close()
