"""Training-only summaries of per-refit latent coordinates, without state labels."""

import numpy as np
import pandas as pd


def _conditional_occupancy(labels: np.ndarray, probabilities: np.ndarray, key: str) -> list[dict]:
    rows: list[dict] = []
    valid = pd.notna(labels) & np.isfinite(probabilities).all(axis=1)
    for label in pd.unique(labels[valid]):
        occupancy = probabilities[valid & (labels == label)].mean(axis=0)
        if isinstance(label, np.generic):
            label = label.item()
        rows.extend(
            {key: label, "state": state, "probability": float(probability)}
            for state, probability in enumerate(occupancy)
        )
    return rows


def training_diagnostics(
    training: pd.DataFrame,
    feature_columns: list[str],
    probabilities: np.ndarray,
    covariances: np.ndarray,
    transition_matrix: np.ndarray,
    segment_starts: np.ndarray,
) -> dict:
    """Summarize filtered training posteriors, never query or smoothed posteriors.

    Feature means use the supplied (standardized) feature coordinates. Volatility
    quantiles use only finite raw volatility at rows with a filtered posterior;
    coincident quantiles collapse into fewer buckets. State numbers are local to
    this fit, and must not be aligned or pooled with independently fitted models.
    """
    finite = np.isfinite(probabilities).all(axis=1)
    posterior = probabilities[finite]
    values = training.loc[:, feature_columns].to_numpy(dtype=float)[finite]
    weight = posterior.sum(axis=0)
    occupancy = weight / len(posterior)
    feature_means = np.divide(
        posterior.T @ values,
        weight[:, None],
        out=np.full((posterior.shape[1], values.shape[1]), np.nan),
        where=weight[:, None] > 0,
    )
    flags = []
    for state, fraction in enumerate(occupancy):
        if fraction < 0.01:
            flags.append(f"low_occupancy:state={state}")
        if fraction > 0.95:
            flags.append(f"dominant_occupancy:state={state}")
        eigenvalues = np.linalg.eigvalsh(covariances[state])
        if eigenvalues[0] < 1e-8 or eigenvalues[0] / eigenvalues[-1] < 1e-6:
            flags.append(f"near_singular_covariance:state={state}")
        if transition_matrix[state].max() >= 0.995:
            flags.append(f"near_deterministic_transition:state={state}")

    by_minute = []
    if "minute_of_session" in training:
        by_minute = _conditional_occupancy(
            training["minute_of_session"].to_numpy(), probabilities, "minute_of_session"
        )
    else:
        flags.append("missing_minute_of_session")

    edges = np.array([], dtype=float)
    by_volatility = []
    if "raw_realized_volatility" in training:
        volatility = training["raw_realized_volatility"].to_numpy(dtype=float)
        available = finite & np.isfinite(volatility)
        if available.any():
            edges = np.unique(np.quantile(volatility[available], [0, 1 / 3, 2 / 3, 1]))
            buckets = np.full(len(training), np.nan)
            buckets[available] = np.searchsorted(edges[1:-1], volatility[available], side="right")
            by_volatility = _conditional_occupancy(buckets, probabilities, "bucket")
    if not by_volatility:
        flags.append("unavailable_raw_realized_volatility")

    returns = np.full(len(training), np.nan)
    if "raw_log_return" in training:
        returns = training["raw_log_return"].to_numpy(dtype=float)
    elif "close" in training:
        close = training["close"].to_numpy(dtype=float)
        valid_pair = (
            ~segment_starts[1:]
            & np.isfinite(close[1:])
            & np.isfinite(close[:-1])
            & (close[1:] > 0)
            & (close[:-1] > 0)
        )
        positions = np.flatnonzero(valid_pair) + 1
        returns[positions] = np.log(close[positions]) - np.log(close[positions - 1])
    signs = np.where(np.isfinite(returns), np.sign(returns), np.nan)
    by_sign = _conditional_occupancy(signs, probabilities, "sign")
    if not by_sign:
        flags.append("unavailable_raw_return_sign")

    return {
        "state_occupancy": occupancy.tolist(),
        "mean_features_by_state": [
            [float(value) if np.isfinite(value) else None for value in row] for row in feature_means
        ],
        "occupancy_by_minute": by_minute,
        "occupancy_by_volatility": by_volatility,
        "volatility_bucket_edges": edges.tolist(),
        "occupancy_by_return_sign": by_sign,
        "flags": flags,
    }
