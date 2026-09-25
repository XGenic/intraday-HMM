"""Outcome availability is independent of distance and checked before retrieval."""

import numpy as np
import pandas as pd

from regime_retrieval.retrieval.representations import ContextIndex


class EligibleCandidateIndex:
    """Sorted valid contexts whose complete maximum-horizon outcome exists.

    With zero embargo, outcomes ending exactly at the query are known and eligible.
    A positive N excludes the current session AND the N preceding exchange sessions.
    This deliberately conservative extra embargo is applied after horizon purging.
    Calendar session ordinals (not observed-session counts) handle missing whole days.
    """

    def __init__(
        self,
        bars: pd.DataFrame,
        labels: pd.DataFrame,
        contexts: ContextIndex,
        max_horizon: int,
    ) -> None:
        ends = contexts.end_positions
        known = labels[f"outcome_end_{max_horizon}m"].iloc[ends].notna().to_numpy()
        self.context_rows = np.flatnonzero(known)
        self.end_positions = ends[known]
        self.outcome_ns = pd.DatetimeIndex(
            labels[f"outcome_end_{max_horizon}m"].iloc[self.end_positions]
        ).asi8
        self.session_indices = bars["session_index"].to_numpy()[self.end_positions]
        self.end_ns = pd.DatetimeIndex(bars["timestamp"].iloc[self.end_positions]).asi8

    def eligible(self, query_time: pd.Timestamp, query_session: int, embargo: int) -> np.ndarray:
        if embargo < 0:
            raise ValueError("embargo must be nonnegative")
        available = np.searchsorted(self.outcome_ns, query_time.value, side="right")
        rows = self.context_rows[:available]
        valid = self.end_ns[:available] < query_time.value
        if embargo:
            valid &= self.session_indices[:available] < query_session - embargo
        return rows[valid]
