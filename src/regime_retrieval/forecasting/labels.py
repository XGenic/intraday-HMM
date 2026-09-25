"""Simple close-to-close returns on the actual contiguous same-session bar index."""

import numpy as np
import pandas as pd


def forward_returns(
    bars: pd.DataFrame, horizons_minutes: list[int], bar_minutes: int
) -> pd.DataFrame:
    result = pd.DataFrame(index=bars.index)
    for horizon in horizons_minutes:
        if horizon <= 0 or horizon % bar_minutes:
            raise ValueError("horizon must be a positive multiple of bar_minutes")
        steps = horizon // bar_minutes
        same_segment = bars["segment_id"].eq(bars["segment_id"].shift(-steps))
        result[f"return_{horizon}m"] = (
            bars["close"].shift(-steps).div(bars["close"]).sub(1).where(same_segment)
        )
        result[f"outcome_end_{horizon}m"] = bars["timestamp"].shift(-steps).where(same_segment)
    return result


def recent_return(bars: pd.DataFrame, context_bars: int) -> np.ndarray:
    same_segment = bars["segment_id"].eq(bars["segment_id"].shift(context_bars))
    return (
        bars["close"].div(bars["close"].shift(context_bars)).sub(1).where(same_segment).to_numpy()
    )
