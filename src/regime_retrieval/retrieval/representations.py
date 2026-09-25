"""Raw contexts never bridge a session boundary, missing bar, or feature warmup."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(frozen=True)
class ContextIndex:
    end_positions: np.ndarray
    vectors: np.ndarray


def build_contexts(
    transformed: pd.DataFrame, feature_columns: list[str], context_bars: int
) -> ContextIndex:
    if context_bars < 1 or not feature_columns:
        raise ValueError("context_bars and feature_columns must be nonempty")
    values = transformed[feature_columns].to_numpy(dtype=float)
    width = context_bars * len(feature_columns)
    if len(values) < context_bars:
        return ContextIndex(np.empty(0, dtype=int), np.empty((0, width)))
    windows = sliding_window_view(values, context_bars, axis=0).transpose(0, 2, 1)
    segments = transformed["segment_id"].to_numpy()
    valid = (segments[context_bars - 1 :] == segments[: len(windows)]) & np.isfinite(windows).all(
        axis=(1, 2)
    )
    ends = np.flatnonzero(valid) + context_bars - 1
    return ContextIndex(ends, windows[valid].reshape(-1, width))
