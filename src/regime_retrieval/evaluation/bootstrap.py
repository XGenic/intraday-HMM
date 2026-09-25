"""Paired session-cluster confidence intervals for HMM minus raw-kNN losses."""

import numpy as np
import pandas as pd

from regime_retrieval.config import BootstrapConfig
from regime_retrieval.evaluation.sensitivity import _validate_prediction_cohorts

COLUMNS = [
    "embargo_sessions",
    "fold",
    "horizon_minutes",
    "model",
    "reference",
    "loss",
    "n_forecasts",
    "n_sessions",
    "mean_difference",
    "ci_lower",
    "ci_upper",
    "confidence",
    "n_resamples",
    "status",
]
MODELS = ("hmm_current", "hmm_trajectory")
LOSSES = ("absolute_error", "squared_error", "brier")


def _losses(frame: pd.DataFrame) -> np.ndarray:
    actual = frame["actual_return"].to_numpy(dtype=float)
    error = frame["median"].to_numpy(dtype=float) - actual
    probability = frame["probability_up"].to_numpy(dtype=float)
    return np.column_stack((np.abs(error), error**2, (probability - (actual > 0)) ** 2))


def paired_session_bootstrap(
    predictions: pd.DataFrame, config: BootstrapConfig, random_state: int
) -> pd.DataFrame:
    """Return percentile intervals for forecast-weighted mean paired loss differences.

    Resampling units are whole query sessions, not bars. A draw contains as many
    sessions as the original cohort, with replacement; unequal session lengths
    contribute their original forecast counts on each appearance. Both HMM models
    and all losses share draws. Historical neighbors are not resampling units.
    """
    _validate_prediction_cohorts(predictions)
    required = {"median", "probability_up"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"missing prediction columns: {sorted(missing)}")
    compared = predictions.loc[predictions["method"].isin((*MODELS, "raw_knn"))]
    if not np.isfinite(compared[["median", "probability_up"]].to_numpy(dtype=float)).all():
        raise ValueError("paired point forecasts and probabilities must be finite")
    if not compared["probability_up"].between(0, 1).all():
        raise ValueError("probability_up must lie in [0, 1]")

    rng = np.random.default_rng(random_state)
    alpha = (1 - config.confidence) / 2
    rows = []
    for (embargo, horizon), group in predictions.groupby(
        ["embargo_sessions", "horizon_minutes"], sort=True
    ):
        if not {*MODELS, "raw_knn"}.issubset(set(group["method"])):
            raise ValueError("each embargo/horizon requires raw_knn, hmm_current, hmm_trajectory")
        indexed = {
            method: group.loc[group["method"].eq(method)].set_index("query_timestamp").sort_index()
            for method in ("raw_knn", *MODELS)
        }
        reference = indexed["raw_knn"]
        reference_loss = _losses(reference)
        differences = np.stack(
            [_losses(indexed[model]) - reference_loss for model in MODELS], axis=1
        )
        if not np.isfinite(differences).all():
            raise ValueError("paired losses must be finite")
        for fold in (0, *sorted(reference["fold"].unique())):
            selected = (
                np.ones(len(reference), dtype=bool)
                if fold == 0
                else reference["fold"].eq(fold).to_numpy()
            )
            cohort = reference.loc[selected]
            loss_differences = differences[selected]
            session_codes, sessions = pd.factorize(cohort["query_session"], sort=True)
            n_sessions = len(sessions)
            n_forecasts = len(cohort)
            counts = np.bincount(session_codes, minlength=n_sessions)
            sums = np.zeros((n_sessions, len(MODELS), len(LOSSES)))
            np.add.at(sums, session_codes, loss_differences)
            mean_difference = sums.sum(axis=0) / n_forecasts
            lower = np.full((len(MODELS), len(LOSSES)), np.nan)
            upper = lower.copy()
            status = "insufficient_sessions"
            if n_sessions >= 2:
                resampled = np.empty((config.n_resamples, len(MODELS), len(LOSSES)))
                # Multinomial multiplicities are exactly whole-session draws with
                # replacement. Only cluster sums/counts enter the bootstrap loop.
                probabilities = np.full(n_sessions, 1 / n_sessions)
                flat_sums = sums.reshape(n_sessions, -1)
                for draw in range(config.n_resamples):
                    multiplicities = rng.multinomial(n_sessions, probabilities)
                    resampled[draw] = (
                        (multiplicities @ flat_sums) / (multiplicities @ counts)
                    ).reshape(len(MODELS), len(LOSSES))
                lower, upper = np.quantile(resampled, [alpha, 1 - alpha], axis=0)
                status = "ok"
            for model_index, model in enumerate(MODELS):
                for loss_index, loss in enumerate(LOSSES):
                    rows.append(
                        {
                            "embargo_sessions": int(embargo),
                            "fold": int(fold),
                            "horizon_minutes": int(horizon),
                            "model": model,
                            "reference": "raw_knn",
                            "loss": loss,
                            "n_forecasts": n_forecasts,
                            "n_sessions": n_sessions,
                            "mean_difference": float(mean_difference[model_index, loss_index]),
                            "ci_lower": float(lower[model_index, loss_index]),
                            "ci_upper": float(upper[model_index, loss_index]),
                            "confidence": config.confidence,
                            "n_resamples": config.n_resamples,
                            "status": status,
                        }
                    )
    return pd.DataFrame(rows, columns=COLUMNS).sort_values(
        ["embargo_sessions", "fold", "horizon_minutes", "model", "loss"], ignore_index=True
    )
