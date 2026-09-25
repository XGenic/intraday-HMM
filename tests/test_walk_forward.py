import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from regime_retrieval.config import ResearchConfig
from regime_retrieval.evaluation.walk_forward import evaluate
from regime_retrieval.models.hmm import HMMFitError, LatentStateModel


def prepared_history():
    frames = []
    generator = np.random.default_rng(127)
    for session_index, session in enumerate(pd.bdate_range("2024-01-02", periods=14)):
        count = 22
        close = 100 * np.exp(np.cumsum(generator.normal(0, 0.001, count)))
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        session.tz_localize("UTC") + pd.Timedelta(hours=14, minutes=35),
                        periods=count,
                        freq="5min",
                    ),
                    "open": close * 0.9999,
                    "high": close * 1.001,
                    "low": close * 0.999,
                    "close": close,
                    "volume": generator.integers(10, 100, count).astype(float),
                    "session": session,
                    "session_index": session_index,
                    "minute_of_session": np.arange(count) * 5,
                    "segment_id": session_index,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def experiment_config():
    return ResearchConfig(
        features={"realized_vol_window": 3},
        context_bars=3,
        horizons_minutes=[5, 15],
        retrieval={"k": 6, "max_neighbors_per_session": 2, "embargo_sessions": 1},
        walk_forward={"min_training_sessions": 3, "refit_frequency": "weekly", "n_folds": 2},
        hmm={"n_states": 2, "covariance_type": "diag", "n_iter": 500, "tol": 0.001},
    )


def test_walk_forward_cutoffs_and_audited_forecasts():
    config = experiment_config()
    result = evaluate(prepared_history(), config)
    p, n = result.predictions, result.neighbors
    assert (p.scaler_fit_end < p.refit_block_start).all()
    assert (p.hmm_fit_end < p.refit_block_start).all()
    assert (p.latest_feature_timestamp <= p.query_timestamp).all()
    assert (n.neighbor_timestamp < n.query_timestamp).all()
    assert (n.outcome_end_15m <= n.query_timestamp).all()
    assert (n.neighbor_session_index < n.query_session_index - 1).all()
    assert set(p.method) == {
        "raw_knn",
        "unconditional",
        "same_time_of_day",
        "momentum",
        "hmm_current",
        "hmm_trajectory",
    }
    assert p.groupby(["query_timestamp", "horizon_minutes"]).method.nunique().eq(6).all()
    assert (n.query_model_id == n.neighbor_model_id).all()
    for (_, method), audit in n.groupby(["query_timestamp", "method"]):
        if method in {"raw_knn", "hmm_current", "hmm_trajectory"}:
            assert audit.groupby("neighbor_session_index").size().le(2).all()
    for _, forecast in p[p.method != "momentum"].sample(12, random_state=5).iterrows():
        audit = n[(n.query_timestamp == forecast.query_timestamp) & (n.method == forecast.method)]
        observed = audit[f"return_{forecast.horizon_minutes}m"]
        assert forecast["median"] == observed.median()
        assert forecast.probability_up == (observed > 0).mean()
        assert forecast.effective_neighbor_count == len(audit)


def test_future_perturbation_preserves_forecasts_and_neighbors():
    bars = prepared_history()
    config = experiment_config()
    cutoff = bars.loc[bars.session_index == 6, "timestamp"].iloc[10]
    changed = bars.copy()
    future = changed.timestamp > cutoff
    changed.loc[future, ["open", "high", "low", "close"]] *= 4
    changed.loc[future, "volume"] *= 100
    original = evaluate(bars, config)
    perturbed = evaluate(changed, config)
    # Realized outcomes are scoring data, not predictions, and may straddle the cutoff.
    scoring = ["actual_return", "actual_outcome_end", "realized_percentile"]
    before = original.predictions.query("query_timestamp <= @cutoff").drop(columns=scoring)
    after = perturbed.predictions.query("query_timestamp <= @cutoff").drop(columns=scoring)
    assert_frame_equal(before, after, check_exact=True)
    assert_frame_equal(
        original.neighbors.query("query_timestamp <= @cutoff"),
        perturbed.neighbors.query("query_timestamp <= @cutoff"),
        check_exact=True,
    )


def test_embargo_sensitivities_share_queries_models_and_reconstruct_latent_distances():
    config = experiment_config()
    result = evaluate(prepared_history(), config, embargoes=[0, 1, 5])
    p, n = result.predictions, result.neighbors
    query_sets = [set(frame.query_timestamp) for _, frame in p.groupby("embargo_sessions")]
    assert query_sets[0] == query_sets[1] == query_sets[2]
    assert (n.outcome_end_15m <= n.query_timestamp).all()
    embargoed = n[n.embargo_sessions > 0]
    assert (
        embargoed.neighbor_session_index
        < embargoed.query_session_index - embargoed.embargo_sessions
    ).all()
    examples = result.representation_examples
    for query_time, frame in examples.groupby("query_timestamp"):
        query = frame[frame.role == "query"].sort_values("context_step")
        posterior_cols = [column for column in frame if column.startswith("posterior_")]
        np.testing.assert_allclose(query[posterior_cols].sum(axis=1), 1, atol=1e-12)
        for method in ("hmm_current", "hmm_trajectory"):
            query_vector = query[posterior_cols].to_numpy()
            if method == "hmm_current":
                query_vector = query_vector[-1]
            for rank, context in frame[frame.role == method].groupby("rank"):
                vector = context.sort_values("context_step")[posterior_cols].to_numpy()
                if method == "hmm_current":
                    vector = vector[-1]
                observed = n[
                    (n.query_timestamp == query_time)
                    & (n.method == method)
                    & (n.embargo_sessions == 1)
                    & (n["rank"] == rank)
                ]
                assert observed.distance.iloc[0] == pytest.approx(
                    np.linalg.norm(query_vector.ravel() - vector.ravel()), abs=1e-12
                )


def test_failed_hmm_skips_whole_paired_block_and_records_failure(monkeypatch):
    original_fit = LatentStateModel.fit

    def fail_first_block(self, training, feature_columns):
        if training.session.nunique() == 4:
            raise HMMFitError(
                "deliberate numerical failure",
                {
                    "status": "failed",
                    "reason": "deliberate numerical failure",
                    "attempts": [],
                    "fit_end": training.timestamp.max().isoformat(),
                },
            )
        return original_fit(self, training, feature_columns)

    monkeypatch.setattr(LatentStateModel, "fit", fail_first_block)
    diagnostics = []
    result = evaluate(prepared_history(), experiment_config(), diagnostic_sink=diagnostics.append)
    failed = [row for row in result.refits if row["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["refit_id"] not in set(result.predictions.refit_id)
    assert {row["status"] for row in diagnostics} == {"failed", "accepted"}
    assert result.predictions.groupby("query_timestamp").method.nunique().eq(6).all()
