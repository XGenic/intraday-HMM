"""All historical baselines consume the same purged candidate universe."""

import numpy as np


def same_time_of_day(
    eligible_rows: np.ndarray, candidate_minutes: np.ndarray, query_minute: int
) -> np.ndarray:
    return eligible_rows[candidate_minutes[eligible_rows] == query_minute]


def momentum_forecast(recent_return: float, horizon_bars: int, context_bars: int) -> dict:
    """Fixed log-return extrapolation; probability is a hard direction, not calibration.

    There is no fitted coefficient, threshold optimization or invented return distribution.
    The research report explicitly distinguishes this point-only control.
    """
    forecast = float(np.expm1(np.log1p(recent_return) * horizon_bars / context_bars))
    return {
        "mean": forecast,
        "median": forecast,
        "probability_up": float((forecast > 0) + 0.5 * (forecast == 0)),
        "q10": np.nan,
        "q25": np.nan,
        "q75": np.nan,
        "q90": np.nan,
        "return_std": np.nan,
        "effective_neighbor_count": 0,
        "distance_mean": np.nan,
        "distance_min": np.nan,
        "distance_max": np.nan,
    }
