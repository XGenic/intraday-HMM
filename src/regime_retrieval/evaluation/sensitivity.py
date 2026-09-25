"""Embargo comparisons on a common, explicitly validated forecast cohort."""

import numpy as np
import pandas as pd

from regime_retrieval.evaluation.metrics import _check_cohorts, metric_tables


def _validate_prediction_cohorts(predictions: pd.DataFrame) -> None:
    """Validate paired methods within each embargo, without intersecting cohorts."""
    required = {
        "embargo_sessions",
        "query_timestamp",
        "query_session",
        "horizon_minutes",
        "method",
        "fold",
        "actual_return",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"missing prediction columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("evaluation requires at least one forecast")
    if predictions[list(required)].isna().any().any():
        raise ValueError("prediction identities and outcomes must not be missing")
    for column, minimum in (("embargo_sessions", 0), ("fold", 1), ("horizon_minutes", 1)):
        values = predictions[column].to_numpy(dtype=float)
        if not (np.isfinite(values) & (values >= minimum) & (values == np.floor(values))).all():
            raise ValueError(f"{column} must contain integers >= {minimum}; fold 0 is aggregate")
    if not np.isfinite(predictions["actual_return"].to_numpy(dtype=float)).all():
        raise ValueError("actual_return must be finite")
    for _, variant in predictions.groupby("embargo_sessions", sort=True):
        _check_cohorts(variant)
        queries = variant.groupby("query_timestamp", sort=False)
        for column in ("query_session", "fold"):
            if not queries[column].nunique().eq(1).all():
                raise ValueError(f"forecasts disagree on {column} for a query")
        if not variant.groupby("query_session")["fold"].nunique().eq(1).all():
            raise ValueError("each query_session must belong to exactly one fold")
        if "actual_outcome_end" in variant:
            outcomes = variant.groupby(["query_timestamp", "horizon_minutes"])
            if not outcomes["actual_outcome_end"].nunique(dropna=False).eq(1).all():
                raise ValueError("methods disagree on actual_outcome_end for a query")


def sensitivity_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize every embargo, rejecting any unpaired variant rather than dropping rows.

    Fold zero contains aggregate metrics; positive fold IDs are preserved. This
    consumes existing forecasts only and never refits models or retrieves neighbors.
    """
    _validate_prediction_cohorts(predictions)
    keys = ["query_timestamp", "horizon_minutes", "method"]
    paired = predictions.groupby(keys, sort=False)
    n_variants = predictions["embargo_sessions"].nunique()
    if not paired.size().eq(n_variants).all():
        raise ValueError("embargo variants must use identical query/horizon/method cohorts")
    outcomes = ["actual_return", "query_session", "fold"]
    if "actual_outcome_end" in predictions:
        outcomes.append("actual_outcome_end")
    for column in outcomes:
        if not paired[column].nunique(dropna=False).eq(1).all():
            raise ValueError(f"embargo variants disagree on {column}")

    tables = []
    for embargo, variant in predictions.groupby("embargo_sessions", sort=True):
        # Canonical ordering also makes floating-point reductions input-order invariant.
        variant = variant.sort_values(keys)
        aggregate, folds, _ = metric_tables(variant)
        aggregate.insert(0, "fold", 0)
        combined = pd.concat([aggregate, folds], ignore_index=True)
        combined.insert(0, "embargo_sessions", int(embargo))
        tables.append(combined)
    return pd.concat(tables, ignore_index=True).sort_values(
        ["embargo_sessions", "fold", "horizon_minutes", "method"], ignore_index=True
    )
