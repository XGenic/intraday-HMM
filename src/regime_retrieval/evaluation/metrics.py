"""Descriptive, paired-cohort forecast diagnostics (not significance tests)."""

import numpy as np
import pandas as pd

LOG_LOSS_EPSILON = 1e-12


def _check_cohorts(predictions: pd.DataFrame) -> None:
    keys = ["query_timestamp", "horizon_minutes", "method"]
    if predictions.duplicated(keys).any():
        raise ValueError("duplicate method forecasts at a query timestamp and horizon")
    for _, horizon in predictions.groupby("horizon_minutes", sort=False):
        methods = horizon["method"].nunique()
        queries = horizon.groupby("query_timestamp", sort=False)
        if not queries.size().eq(methods).all():
            raise ValueError("methods must be evaluated on identical timestamp cohorts")
        for column in ("actual_return", "query_session", "fold"):
            if not queries[column].nunique(dropna=False).eq(1).all():
                raise ValueError(f"methods disagree on {column} for a query")


def _summary(group: pd.DataFrame) -> dict:
    actual = group["actual_return"].to_numpy(dtype=float)
    predicted = group["median"].to_numpy(dtype=float)
    error = predicted - actual
    probability = group["probability_up"].to_numpy(dtype=float)
    up = (actual > 0).astype(float)
    clipped = np.clip(probability, LOG_LOSS_EPSILON, 1 - LOG_LOSS_EPSILON)
    rank_correlation = float("nan")
    if len(group) > 1 and np.ptp(actual) > 0 and np.ptp(predicted) > 0:
        rank_correlation = float(pd.Series(actual).rank().corr(pd.Series(predicted).rank()))
    lower = group["q10"].to_numpy(dtype=float)
    upper = group["q90"].to_numpy(dtype=float)
    intervals = np.isfinite(lower) & np.isfinite(upper)
    return {
        "n_forecasts": len(group),
        "n_sessions": int(group["query_session"].nunique()),
        "query_start": group["query_timestamp"].min().isoformat(),
        "query_end": group["query_timestamp"].max().isoformat(),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mean_signed_error": float(np.mean(error)),
        "spearman": rank_correlation,
        "brier": float(np.mean((probability - up) ** 2)),
        "log_loss": float(-np.mean(up * np.log(clipped) + (1 - up) * np.log1p(-clipped))),
        "n_intervals": int(intervals.sum()),
        "interval_10_90_width": float(np.mean(upper[intervals] - lower[intervals]))
        if intervals.any()
        else float("nan"),
        "interval_10_90_coverage": float(
            np.mean(
                (actual[intervals] >= lower[intervals]) & (actual[intervals] <= upper[intervals])
            )
        )
        if intervals.any()
        else float("nan"),
        "neighbor_return_std_mean": float(group["return_std"].mean()),
        "effective_neighbor_count_mean": float(group["effective_neighbor_count"].mean()),
        "distance_mean": float(group["distance_mean"].mean()),
        "distance_min_mean": float(group["distance_min"].mean()),
        "distance_max_mean": float(group["distance_max"].mean()),
    }


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return aggregate, chronological fold, and ten-bin probability calibration tables.

    The primary point forecast is the median. Signed error is forecast minus realized
    return; positive outcomes are strictly greater than zero. Brier uses the original
    probabilities, while log loss alone is clipped with ``LOG_LOSS_EPSILON``. Interval
    endpoints are inclusive. Missing momentum intervals remain undefined, not zero.
    All methods must share a timestamp cohort within each horizon.
    """
    if predictions.empty:
        raise ValueError("metrics require at least one forecast")
    _check_cohorts(predictions)
    aggregate = []
    calibration = []
    for (method, horizon), group in predictions.groupby(["method", "horizon_minutes"], sort=True):
        identity = {"method": method, "horizon_minutes": int(horizon)}
        aggregate.append({**identity, **_summary(group)})
        probabilities = group["probability_up"].to_numpy(dtype=float)
        # Left-closed bins; the last bin also includes 1.0.
        bins = np.minimum((probabilities * 10).astype(int), 9)
        for index in range(10):
            selected = group.iloc[np.flatnonzero(bins == index)]
            calibration.append(
                {
                    **identity,
                    "bin": index,
                    "bin_lower": index / 10,
                    "bin_upper": (index + 1) / 10,
                    "n_forecasts": len(selected),
                    "n_sessions": int(selected["query_session"].nunique()),
                    "mean_probability_up": float(selected["probability_up"].mean()),
                    "observed_up_frequency": float((selected["actual_return"] > 0).mean())
                    if len(selected)
                    else float("nan"),
                }
            )
    folds = []
    for (fold, method, horizon), group in predictions.groupby(
        ["fold", "method", "horizon_minutes"], sort=False
    ):
        folds.append(
            {
                "fold": int(fold),
                "method": method,
                "horizon_minutes": int(horizon),
                **_summary(group),
            }
        )
    fold_table = pd.DataFrame(folds).sort_values(
        ["query_start", "fold", "horizon_minutes", "method"], ignore_index=True
    )
    return pd.DataFrame(aggregate), fold_table, pd.DataFrame(calibration)


def paired_error_differences(predictions: pd.DataFrame) -> pd.DataFrame:
    """Mean raw-kNN minus control losses on exactly paired timestamps; no inference."""
    columns = [
        "horizon_minutes",
        "control",
        "n_forecasts",
        "n_sessions",
        "mean_absolute_error_difference",
        "mean_squared_error_difference",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    _check_cohorts(predictions)
    rows = []
    for horizon, group in predictions.groupby("horizon_minutes", sort=True):
        raw = group.loc[group["method"].eq("raw_knn")].set_index("query_timestamp")
        if raw.empty:
            continue
        raw_error = raw["median"] - raw["actual_return"]
        for method, control in group.loc[~group["method"].eq("raw_knn")].groupby("method"):
            control = control.set_index("query_timestamp").loc[raw.index]
            control_error = control["median"] - control["actual_return"]
            rows.append(
                {
                    "horizon_minutes": int(horizon),
                    "control": method,
                    "n_forecasts": len(raw),
                    "n_sessions": int(raw["query_session"].nunique()),
                    "mean_absolute_error_difference": float(
                        (raw_error.abs() - control_error.abs()).mean()
                    ),
                    "mean_squared_error_difference": float(
                        (raw_error**2 - control_error**2).mean()
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)
