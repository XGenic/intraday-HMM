"""Frozen past-only models, causal state filtering, and paired analogue evaluation."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from regime_retrieval.config import ResearchConfig, refit_key
from regime_retrieval.features.causal import compute_causal_features
from regime_retrieval.features.scaling import FeatureTransformer
from regime_retrieval.forecasting.analog_forecast import summarize_returns
from regime_retrieval.forecasting.baselines import momentum_forecast, same_time_of_day
from regime_retrieval.forecasting.labels import forward_returns, recent_return
from regime_retrieval.models.hmm import HMMFitError, LatentStateModel
from regime_retrieval.retrieval.eligibility import EligibleCandidateIndex
from regime_retrieval.retrieval.neighbors import exact_knn
from regime_retrieval.retrieval.representations import build_contexts


@dataclass
class EvaluationResult:
    predictions: pd.DataFrame
    neighbors: pd.DataFrame
    refits: list[dict]
    skipped_blocks: list[dict]
    state_diagnostics: list[dict]
    representation_examples: pd.DataFrame


class NoEvaluableQueries(ValueError):
    def __init__(self, refits: list[dict], skipped_blocks: list[dict]):
        super().__init__(
            "No evaluable queries: require sufficient training history, a converged HMM, "
            "contiguous contexts/outcomes, and same-time history under every requested embargo. "
            "Inspect saved refit diagnostics and skipped blocks."
        )
        self.refits = refits
        self.skipped_blocks = skipped_blocks


def _example_contexts(
    transformed: pd.DataFrame,
    posterior: np.ndarray,
    feature_columns: list[str],
    context_bars: int,
    position: int,
    ends: np.ndarray,
    cohorts: dict,
    shared: dict,
) -> list[pd.DataFrame]:
    """Small, deterministic first-query examples, not outcome-selected success stories."""
    examples = []
    selections = [("query", position, 0)]
    for method in ("raw_knn", "hmm_current", "hmm_trajectory"):
        rows, _ = cohorts[method]
        selections.extend((method, int(ends[row]), rank + 1) for rank, row in enumerate(rows[:3]))
    for role, end, rank in selections:
        start = end - context_bars + 1
        context = transformed.iloc[start : end + 1]
        frame = context[feature_columns].reset_index(drop=True).copy()
        frame["query_timestamp"] = shared["query_timestamp"]
        frame["refit_id"] = shared["refit_id"]
        frame["embargo_sessions"] = shared["embargo_sessions"]
        frame["role"] = role
        frame["neighbor_timestamp"] = transformed["timestamp"].iloc[end]
        frame["rank"] = rank
        frame["context_timestamp"] = context["timestamp"].to_numpy()
        frame["context_step"] = np.arange(context_bars)
        frame["close"] = context["close"].to_numpy()
        frame["normalized_close"] = frame["close"] / frame["close"].iloc[0] - 1
        for state in range(posterior.shape[1]):
            frame[f"posterior_{state}"] = posterior[start : end + 1, state]
        examples.append(frame)
    return examples


def evaluate(
    bars: pd.DataFrame,
    config: ResearchConfig,
    neighbor_sink: Callable[[pd.DataFrame], None] | None = None,
    *,
    embargoes: list[int] | None = None,
    diagnostic_sink: Callable[[dict], None] | None = None,
) -> EvaluationResult:
    """Six methods and all requested embargoes share an identical scored query cohort.

    Each block fits preprocessing and ONE HMM before its first query. History and
    current windows are forward-filtered under that same HMM, never smoothed over
    future query observations. A failed HMM skips the entire paired block, including
    raw controls. CLI streams audits and diagnostics, including unsuccessful fits.
    """
    embargoes = sorted(
        set(embargoes if embargoes is not None else [config.retrieval.embargo_sessions])
    )
    if not embargoes or any(value < 0 for value in embargoes):
        raise ValueError("at least one nonnegative embargo is required")
    if config.retrieval.embargo_sessions not in embargoes:
        raise ValueError("embargoes must include the configured primary embargo")
    causal = compute_causal_features(bars, config.features, config.bar_minutes)
    labels = forward_returns(bars, config.horizons_minutes, config.bar_minutes)
    returns = labels[[f"return_{h}m" for h in config.horizons_minutes]].to_numpy()
    outcome_known = np.isfinite(returns).all(axis=1)
    momentum = recent_return(bars, config.context_bars)
    session_ordinals = bars["session_index"].to_numpy()
    minutes = bars["minute_of_session"].to_numpy()
    timestamps = bars["timestamp"]
    sessions = bars["session"]
    keys = sessions.map(lambda date: refit_key(date, config.walk_forward.refit_frequency))
    records: list[dict] = []
    audits: list[pd.DataFrame] = []
    refits: list[dict] = []
    skipped: list[dict] = []
    diagnostics: list[dict] = []
    examples: list[pd.DataFrame] = []
    planned_test_sessions: list[pd.Timestamp] = []
    emit = neighbor_sink if neighbor_sink is not None else audits.append

    for block, block_rows in bars.groupby(keys, sort=False):
        first, last = int(block_rows.index[0]), int(block_rows.index[-1])
        training = causal.iloc[:first]
        n_sessions = training["session"].nunique()
        if n_sessions < config.walk_forward.min_training_sessions:
            skipped.append({"block": block, "reason": "insufficient_training_sessions"})
            continue
        # Folds depend on scheduled evaluation dates, never on later fit/outcome success.
        planned_test_sessions.extend(block_rows["session"].drop_duplicates().tolist())
        processor = FeatureTransformer(config.features).fit(training)
        block_start = timestamps.iloc[first]
        if not processor.fit_end < block_start:
            raise RuntimeError("preprocessing fit cutoff is not strictly before block start")
        transformed = processor.transform(causal.iloc[: last + 1])
        refit = {
            "refit_id": block,
            "block_start": block_start.isoformat(),
            "block_end": timestamps.iloc[last].isoformat(),
            "fit_start": timestamps.iloc[0].isoformat(),
            "fit_end": processor.fit_end.isoformat(),
            "training_sessions": n_sessions,
            "training_bars": first,
            "feature_columns": processor.feature_columns,
        }
        model = LatentStateModel(config.hmm)
        try:
            model.fit(transformed.iloc[:first], processor.feature_columns)
        except HMMFitError as exc:
            diagnostic = {**exc.diagnostics, **refit, "status": "failed"}
            diagnostics.append(diagnostic)
            refits.append({**refit, "status": "failed"})
            skipped.append({"block": block, "reason": "hmm_fit_failed", "details": str(exc)})
            if diagnostic_sink is not None:
                diagnostic_sink(diagnostic)
            continue
        if not model.fit_end < block_start:
            raise RuntimeError("HMM fit cutoff is not strictly before block start")
        diagnostic = {
            **model.diagnostics_,
            **refit,
            "hmm_fit_end": model.fit_end.isoformat(),
            "preprocessing": {
                "mean": processor.scaler.mean_.to_dict(),
                "scale": processor.scaler.scale_.to_dict(),
                "volume_statistics": processor.seasonality.statistics_.reset_index().to_dict(
                    "records"
                )
                if processor.seasonality is not None
                else [],
            },
        }
        diagnostics.append(diagnostic)
        if diagnostic_sink is not None:
            diagnostic_sink(diagnostic)
        refits.append({**refit, "status": "accepted", "hmm_fit_end": model.fit_end.isoformat()})
        posterior = model.filter(transformed)
        posterior_columns = [f"posterior_{state}" for state in range(config.hmm.n_states)]
        latent_frame = pd.DataFrame(posterior, columns=posterior_columns, index=transformed.index)
        latent_frame["segment_id"] = transformed["segment_id"]
        contexts = build_contexts(transformed, processor.feature_columns, config.context_bars)
        latent_contexts = build_contexts(latent_frame, posterior_columns, config.context_bars)
        if not np.array_equal(contexts.end_positions, latent_contexts.end_positions):
            raise RuntimeError("raw and latent representations do not share the same context index")
        ends = contexts.end_positions
        representations = {
            "raw_knn": contexts.vectors,
            "hmm_current": posterior[ends],
            "hmm_trajectory": latent_contexts.vectors,
        }
        candidates = EligibleCandidateIndex(bars, labels, contexts, max(config.horizons_minutes))
        context_minutes = minutes[ends]
        queries_before = len(records)
        captured_example = False
        for query_row in np.flatnonzero(ends >= first):
            position = int(ends[query_row])
            if not outcome_known[position] or not np.isfinite(momentum[position]):
                continue
            query_time = timestamps.iloc[position]
            eligible_by_embargo = {
                embargo: candidates.eligible(query_time, int(session_ordinals[position]), embargo)
                for embargo in embargoes
            }
            tod_by_embargo = {
                embargo: same_time_of_day(eligible, context_minutes, int(minutes[position]))
                for embargo, eligible in eligible_by_embargo.items()
            }
            if any(not len(tod) for tod in tod_by_embargo.values()):
                continue
            for embargo in embargoes:
                eligible = eligible_by_embargo[embargo]
                cohorts = {
                    "unconditional": (eligible, None),
                    "same_time_of_day": (tod_by_embargo[embargo], None),
                }
                for method, vectors in representations.items():
                    selected, distances = exact_knn(
                        vectors[query_row],
                        vectors[eligible],
                        session_ordinals[ends[eligible]],
                        config.retrieval.k,
                        config.retrieval.max_neighbors_per_session,
                        config.retrieval.distance,
                    )
                    cohorts[method] = (eligible[selected], distances)
                shared = {
                    "query_timestamp": query_time,
                    "query_session": sessions.iloc[position],
                    "query_session_index": int(session_ordinals[position]),
                    "minute_of_session": int(minutes[position]),
                    "context_start": timestamps.iloc[position - config.context_bars + 1],
                    "latest_feature_timestamp": query_time,
                    "refit_id": block,
                    "refit_block_start": block_start,
                    "scaler_fit_end": processor.fit_end,
                    "hmm_fit_end": model.fit_end,
                    "query_model_id": block,
                    "embargo_sessions": embargo,
                }
                if not captured_example and embargo == config.retrieval.embargo_sessions:
                    examples.extend(
                        _example_contexts(
                            transformed,
                            posterior,
                            processor.feature_columns,
                            config.context_bars,
                            position,
                            ends,
                            cohorts,
                            shared,
                        )
                    )
                    captured_example = True
                for method, (cohort, distances) in cohorts.items():
                    neighbor_ends = ends[cohort]
                    audit = pd.DataFrame(
                        {
                            "query_timestamp": query_time,
                            "query_session_index": int(session_ordinals[position]),
                            "method": method,
                            "refit_id": block,
                            "scaler_fit_end": processor.fit_end,
                            "hmm_fit_end": model.fit_end,
                            "query_model_id": block,
                            "neighbor_model_id": block,
                            "embargo_sessions": embargo,
                            "neighbor_timestamp": timestamps.iloc[neighbor_ends].to_numpy(),
                            "neighbor_context_start": timestamps.iloc[
                                neighbor_ends - config.context_bars + 1
                            ].to_numpy(),
                            "neighbor_session": sessions.iloc[neighbor_ends].to_numpy(),
                            "neighbor_session_index": session_ordinals[neighbor_ends],
                            "rank": np.arange(1, len(neighbor_ends) + 1),
                            "distance": distances if distances is not None else np.nan,
                        }
                    )
                    for horizon_index, horizon in enumerate(config.horizons_minutes):
                        historical = returns[neighbor_ends, horizon_index]
                        actual = float(returns[position, horizon_index])
                        records.append(
                            {
                                **shared,
                                "method": method,
                                "horizon_minutes": horizon,
                                **summarize_returns(historical, distances),
                                "actual_return": actual,
                                "actual_outcome_end": labels[f"outcome_end_{horizon}m"].iloc[
                                    position
                                ],
                                "realized_percentile": float(np.mean(historical <= actual)),
                            }
                        )
                        audit[f"return_{horizon}m"] = historical
                        audit[f"outcome_end_{horizon}m"] = (
                            labels[f"outcome_end_{horizon}m"].iloc[neighbor_ends].to_numpy()
                        )
                    emit(audit)
                for horizon_index, horizon in enumerate(config.horizons_minutes):
                    records.append(
                        {
                            **shared,
                            "method": "momentum",
                            "horizon_minutes": horizon,
                            **momentum_forecast(
                                float(momentum[position]),
                                horizon // config.bar_minutes,
                                config.context_bars,
                            ),
                            "actual_return": float(returns[position, horizon_index]),
                            "actual_outcome_end": labels[f"outcome_end_{horizon}m"].iloc[position],
                            "realized_percentile": np.nan,
                        }
                    )
        if len(records) == queries_before:
            skipped.append({"block": block, "reason": "no_common_eligible_queries"})
    if not records:
        raise NoEvaluableQueries(refits, skipped)
    predictions = pd.DataFrame(records)
    test_sessions = np.asarray(planned_test_sessions, dtype="datetime64[ns]")
    fold_map = {
        date: fold + 1
        for fold, dates in enumerate(
            np.array_split(test_sessions, min(config.walk_forward.n_folds, len(test_sessions)))
        )
        for date in dates
    }
    predictions["fold"] = predictions["query_session"].map(fold_map)
    neighbors = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()
    return EvaluationResult(
        predictions,
        neighbors,
        refits,
        skipped,
        diagnostics,
        pd.concat(examples, ignore_index=True),
    )
