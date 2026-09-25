"""Equal-weight empirical distributions; the median is the primary point forecast."""

import numpy as np


def summarize_returns(returns: np.ndarray, distances: np.ndarray | None = None) -> dict:
    if not len(returns) or not np.isfinite(returns).all():
        raise ValueError("forecast distribution must contain finite known returns")
    q10, q25, median, q75, q90 = np.quantile(returns, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "mean": float(np.mean(returns)),
        "median": float(median),
        "probability_up": float(np.mean(returns > 0)),
        "q10": float(q10),
        "q25": float(q25),
        "q75": float(q75),
        "q90": float(q90),
        "return_std": float(np.std(returns)),
        "effective_neighbor_count": len(returns),
        "distance_mean": float(np.mean(distances)) if distances is not None else np.nan,
        "distance_min": float(np.min(distances)) if distances is not None else np.nan,
        "distance_max": float(np.max(distances)) if distances is not None else np.nan,
    }
