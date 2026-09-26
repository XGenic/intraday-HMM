"""Audit saved real-data forecasts and references without refitting any model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from threadpoolctl import threadpool_limits

from regime_retrieval.config import load_config
from regime_retrieval.evaluation.sensitivity import _validate_prediction_cohorts
from regime_retrieval.features.causal import compute_causal_features
from regime_retrieval.features.scaling import FeatureScaler, FeatureTransformer
from regime_retrieval.features.seasonality import VolumeSeasonality
from regime_retrieval.forecasting.analog_forecast import summarize_returns
from regime_retrieval.forecasting.labels import forward_returns
from regime_retrieval.models.hmm import _forward_filter, _layout
from regime_retrieval.reporting.artifacts import write_json


def reconstruct_examples(run, config, bars, dataset):
    examples = pd.read_parquet(run / "representation_examples.parquet")
    refits = examples.refit_id.drop_duplicates().tolist()
    selected = list(dict.fromkeys([refits[0], refits[len(refits) // 2], refits[-1]]))
    causal = compute_causal_features(bars, config.features, config.bar_minutes)
    checked = 0
    for refit in selected:
        diagnostic = json.loads((run / "state_diagnostics" / f"{refit}.json").read_text())
        processor = FeatureTransformer(config.features)
        processor.scaler = FeatureScaler()
        processor.scaler.mean_ = pd.Series(diagnostic["preprocessing"]["mean"])
        processor.scaler.scale_ = pd.Series(diagnostic["preprocessing"]["scale"])
        processor.seasonality = VolumeSeasonality()
        processor.seasonality.statistics_ = pd.DataFrame(
            diagnostic["preprocessing"]["volume_statistics"]
        ).set_index("minute_of_session")
        transformed = processor.transform(
            causal.loc[causal.timestamp <= pd.Timestamp(diagnostic["block_end"])]
        )
        values, finite, starts, _ = _layout(
            transformed, diagnostic["feature_columns"], pd.Timedelta(minutes=5)
        )
        with threadpool_limits(limits=1):
            posterior = _forward_filter(
                values,
                finite,
                starts,
                *[
                    np.asarray(diagnostic[key])
                    for key in (
                        "state_means",
                        "state_covariances",
                        "transition_matrix",
                        "start_probabilities",
                    )
                ],
            )
        saved = examples.loc[examples.refit_id.eq(refit)]
        positions = pd.DatetimeIndex(transformed.timestamp).get_indexer(saved.context_timestamp)
        np.testing.assert_allclose(
            saved[diagnostic["feature_columns"]], values[positions], rtol=1e-12, atol=1e-12
        )
        posterior_columns = [f"posterior_{state}" for state in range(config.hmm.n_states)]
        np.testing.assert_allclose(
            saved[posterior_columns], posterior[positions], rtol=1e-10, atol=1e-10
        )
        query = saved.loc[saved.role.eq("query")].sort_values("context_step")
        query_stamp = query.query_timestamp.iloc[0]
        members = dataset.to_table(
            filter=(ds.field("query_timestamp") == query_stamp)
            & (ds.field("embargo_sessions") == 1)
            & ds.field("method").isin(["raw_knn", "hmm_current", "hmm_trajectory"])
        ).to_pandas()
        for (role, rank), neighbor in saved.loc[saved.role.ne("query")].groupby(["role", "rank"]):
            neighbor = neighbor.sort_values("context_step")
            columns = diagnostic["feature_columns"] if role == "raw_knn" else posterior_columns
            left, right = query[columns].to_numpy(), neighbor[columns].to_numpy()
            if role == "hmm_current":
                left, right = left[-1], right[-1]
            distance = np.linalg.norm(left.ravel() - right.ravel())
            audit_distance = members.loc[
                members.method.eq(role) & members["rank"].eq(rank), "distance"
            ].iloc[0]
            np.testing.assert_allclose(distance, audit_distance, rtol=1e-10, atol=1e-12)
        checked += len(saved)
    return {
        "refits_reconstructed_without_fitting": selected,
        "representation_rows_checked": checked,
    }


def audit(run: Path, prefix_run: Path | None = None):
    config = load_config(run / "config.yaml")
    manifest = json.loads((run / "data_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert hashlib.sha256(config.data.path.read_bytes()).hexdigest() == manifest["raw_sha256"]
    predictions = pd.read_parquet(run / "predictions.parquet")
    _validate_prediction_cohorts(predictions)
    assert predictions.groupby(["query_timestamp", "embargo_sessions"]).size().eq(24).all()
    assert (predictions.scaler_fit_end < predictions.refit_block_start).all()
    assert (predictions.hmm_fit_end < predictions.refit_block_start).all()
    assert (predictions.latest_feature_timestamp <= predictions.query_timestamp).all()
    assert predictions.query_model_id.eq(predictions.refit_id).all()
    bars = pd.read_parquet(run / "bars.parquet")
    labels = forward_returns(bars, config.horizons_minutes, config.bar_minutes)
    stamps = pd.DatetimeIndex(bars.timestamp)
    positions = stamps.get_indexer(predictions.query_timestamp)
    assert (positions >= 0).all()
    for horizon in config.horizons_minutes:
        mask = predictions.horizon_minutes.eq(horizon)
        np.testing.assert_allclose(
            predictions.loc[mask, "actual_return"],
            labels[f"return_{horizon}m"].iloc[positions[mask]],
            rtol=0,
            atol=0,
        )

    dataset = ds.dataset(run / "neighbors.parquet", format="parquet")
    count = 0
    for batch in dataset.scanner(batch_size=100_000).to_batches():
        frame = batch.to_pandas()
        count += len(frame)
        assert (frame.neighbor_timestamp < frame.query_timestamp).all()
        assert (frame.scaler_fit_end < frame.query_timestamp).all()
        assert (frame.hmm_fit_end < frame.query_timestamp).all()
        assert frame.query_model_id.eq(frame.neighbor_model_id).all()
        assert frame.query_model_id.eq(frame.refit_id).all()
        embargoed = frame.embargo_sessions.gt(0)
        assert (
            frame.loc[embargoed, "neighbor_session_index"]
            < frame.loc[embargoed, "query_session_index"] - frame.loc[embargoed, "embargo_sessions"]
        ).all()
        positions = stamps.get_indexer(frame.neighbor_timestamp)
        assert (positions >= 0).all()
        expected = labels.iloc[positions]
        for horizon in config.horizons_minutes:
            assert (frame[f"outcome_end_{horizon}m"] <= frame.query_timestamp).all()
            np.testing.assert_allclose(
                frame[f"return_{horizon}m"], expected[f"return_{horizon}m"], rtol=0, atol=0
            )
            assert np.array_equal(
                frame[f"outcome_end_{horizon}m"].array.asi8,
                expected[f"outcome_end_{horizon}m"].array.asi8,
            )
    assert count == manifest["neighbor_record_count"]

    retrieval = ("raw_knn", "hmm_current", "hmm_trajectory")
    columns = [
        "query_timestamp",
        "method",
        "embargo_sessions",
        "neighbor_session",
        "neighbor_timestamp",
        "rank",
    ]
    tail = pd.DataFrame()
    max_cap = 0

    def check_caps(frame):
        if frame.empty:
            return 0
        keys = ["query_timestamp", "method", "embargo_sessions"]
        assert not frame.duplicated(keys + ["neighbor_timestamp"]).any()
        assert frame.groupby(keys).size().le(config.retrieval.k).all()
        cap = int(frame.groupby(keys + ["neighbor_session"]).size().max())
        assert cap <= config.retrieval.max_neighbors_per_session
        return cap

    for batch in dataset.scanner(
        columns=columns, filter=ds.field("method").isin(retrieval)
    ).to_batches():
        frame = pd.concat([tail, batch.to_pandas()], ignore_index=True)
        last = frame.query_timestamp.iloc[-1]
        max_cap = max(max_cap, check_caps(frame.loc[frame.query_timestamp.ne(last)]))
        tail = frame.loc[frame.query_timestamp.eq(last)].copy()
    max_cap = max(max_cap, check_caps(tail))

    query_times = predictions.query_timestamp.drop_duplicates().sort_values()
    selected = query_times.iloc[[0, len(query_times) // 2, -1]]
    cohorts = dataset.to_table(
        filter=ds.field("query_timestamp").isin(selected.tolist())
    ).to_pandas()
    samples = predictions.loc[
        predictions.query_timestamp.isin(selected) & predictions.method.ne("momentum")
    ]
    for _, prediction in samples.iterrows():
        cohort = cohorts.loc[
            cohorts.query_timestamp.eq(prediction.query_timestamp)
            & cohorts.method.eq(prediction.method)
            & cohorts.embargo_sessions.eq(prediction.embargo_sessions)
        ]
        assert len(cohort) == prediction.effective_neighbor_count
        distances = cohort.distance.to_numpy() if prediction.method in retrieval else None
        summary = summarize_returns(
            cohort[f"return_{prediction.horizon_minutes}m"].to_numpy(), distances
        )
        for key, value in summary.items():
            np.testing.assert_allclose(
                prediction[key], value, rtol=1e-12, atol=1e-15, equal_nan=True
            )
    result = {
        "status": "passed",
        "forecast_rows_checked": len(predictions),
        "neighbor_rows_checked": count,
        "maximum_neighbors_per_historical_session": max_cap,
        "reconstructed_distributions": len(samples),
        "checks": [
            "source checksum",
            "paired cohorts",
            "feature and model cutoffs",
            "neighbor availability and embargo",
            "same-model identities",
            "saved outcomes versus canonical bars",
            "unique capped kNN cohorts",
            "sample forecast distribution reconstruction",
        ],
    }
    result.update(reconstruct_examples(run, config, bars, dataset))
    if prefix_run is not None:
        prefix = pd.read_parquet(prefix_run / "predictions.parquet")
        keys = ["query_timestamp", "embargo_sessions", "method", "horizon_minutes"]
        earlier = predictions.loc[predictions.query_timestamp <= prefix.query_timestamp.max()]
        # Fold assignments depend on the full planned evaluation period, not on forecasts.
        pd.testing.assert_frame_equal(
            prefix.drop(columns="fold").sort_values(keys).reset_index(drop=True),
            earlier.drop(columns="fold").sort_values(keys).reset_index(drop=True),
            check_exact=True,
        )
        result["future_extension_identical_forecast_rows"] = len(prefix)
    snapshot = run / "source_snapshot" / "regime_retrieval"
    if snapshot.exists():
        for name, digest in manifest["source_sha256"].items():
            assert hashlib.sha256((snapshot / name).read_bytes()).hexdigest() == digest
        result["source_snapshot_files_verified"] = len(manifest["source_sha256"])
    write_json(run / "audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--prefix-run", type=Path)
    args = parser.parse_args()
    audit(args.run, args.prefix_run)
