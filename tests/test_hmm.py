import itertools
import json
import warnings

import numpy as np
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM

from regime_retrieval.config import HMMConfig
from regime_retrieval.models import HMMFitError, LatentStateModel


def observations(values):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    frame = pd.DataFrame(values, columns=[f"x{i}" for i in range(values.shape[1])])
    frame["timestamp"] = pd.date_range("2024-01-02T14:35Z", periods=len(frame), freq="5min")
    frame["session"] = "2024-01-02"
    frame["segment_id"] = 0
    frame["minute_of_session"] = np.arange(len(frame)) * 5
    return frame


def analytical_model():
    # Known Gaussian parameters make the posterior independently calculable;
    # these are not substituted for a real fit in fitting/failure-path tests.
    model = GaussianHMM(n_components=2, covariance_type="full")
    model.n_features = 1
    model.startprob_ = np.array([0.6, 0.4])
    model.transmat_ = np.array([[0.9, 0.1], [0.2, 0.8]])
    model.means_ = np.array([[-1.0], [1.0]])
    model.covars_ = np.array([[[1.0]], [[0.5]]])
    wrapper = LatentStateModel(HMMConfig(n_states=2))
    wrapper.model_ = model
    wrapper.feature_columns = ["x0"]
    wrapper._interval = pd.Timedelta(minutes=5)
    return wrapper


def hand_forward(values):
    means = np.array([-1.0, 1.0])
    variance = np.array([1.0, 0.5])
    emissions = np.exp(-0.5 * (np.asarray(values)[:, None] - means) ** 2 / variance)
    emissions /= np.sqrt(2 * np.pi * variance)
    transitions = np.array([[0.9, 0.1], [0.2, 0.8]])
    prior = np.array([0.6, 0.4])
    posteriors = []
    for emission in emissions:
        posterior = prior * emission
        posterior /= posterior.sum()
        posteriors.append(posterior)
        prior = posterior @ transitions
    return np.array(posteriors), emissions


def synthetic_training():
    rng = np.random.default_rng(412)
    sessions = 6
    per_session = 60
    means = np.array([[-1.6, -0.8], [1.6, 0.8]])
    covariance = np.array([[0.3, 0.08], [0.08, 0.2]])
    frames = []
    for session in range(sessions):
        state = int(rng.integers(2))
        values = []
        for _ in range(per_session):
            if rng.random() < 0.06:
                state = 1 - state
            values.append(rng.multivariate_normal(means[state], covariance))
        frame = observations(values)
        frame["timestamp"] += pd.Timedelta(days=session)
        frame["session"] = (pd.Timestamp("2024-01-02") + pd.Timedelta(days=session)).date()
        frame["segment_id"] = session
        frame["raw_realized_volatility"] = 0.01 + np.abs(frame.x1) * 0.01
        frame["raw_log_return"] = frame.x0 * 0.001
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_filter_matches_hand_forward_not_smoothed_posterior():
    values = [0.1, 0.4, -0.7, 2.0]
    expected, emissions = hand_forward(values)
    result = analytical_model().filter(observations(values))
    np.testing.assert_allclose(result, expected, atol=1e-14)
    np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-14)

    # Exhaustively sum state paths to obtain the *smoothed* first posterior.
    transition = np.array([[0.9, 0.1], [0.2, 0.8]])
    first_state_mass = np.zeros(2)
    for path in itertools.product(range(2), repeat=len(values)):
        mass = [0.6, 0.4][path[0]] * emissions[0, path[0]]
        for row in range(1, len(values)):
            mass *= transition[path[row - 1], path[row]] * emissions[row, path[row]]
        first_state_mass[path[0]] += mass
    smoothed_first = first_state_mass / first_state_mass.sum()
    assert abs(result[0, 0] - smoothed_first[0]) > 0.01


def test_future_observations_and_truncation_cannot_change_filtered_prefix():
    model = analytical_model()
    frame = observations([-0.5, 0.1, 0.8, -0.2, 1.2, 0.5])
    prefix = model.filter(frame.iloc[:3])
    np.testing.assert_allclose(model.filter(frame)[:3], prefix, atol=1e-14)
    changed = frame.copy()
    changed.loc[3:, "x0"] = [-1e3, np.nan, 1e3]
    np.testing.assert_allclose(model.filter(changed)[:3], prefix, atol=1e-14)


def test_session_segment_nan_and_timestamp_gaps_reset_to_start_distribution():
    model = analytical_model()
    frame = observations([-2, -1, np.nan, 0.7, 1.5, 0.7, 1.5, 0.7, 1.5, 0.7])
    frame.loc[5:, "session"] = "2024-01-03"
    frame.loc[5:, "timestamp"] += pd.Timedelta(days=1)
    frame.loc[7:, "segment_id"] = 1
    frame.loc[9:, "timestamp"] += pd.Timedelta(minutes=5)
    result = model.filter(frame)
    assert np.isnan(result[2]).all()
    first_posterior = hand_forward([0.7])[0][0]
    for row in [3, 5, 7, 9]:
        np.testing.assert_allclose(result[row], first_posterior, atol=1e-14)
    np.testing.assert_allclose(result[[0, 1, 3, 4, 5, 6, 7, 8, 9]].sum(axis=1), 1.0)
    assert not np.allclose(result[4], first_posterior)


def test_genuine_multivariate_fit_is_deterministic_and_diagnostics_are_training_only():
    training = synthetic_training()
    config = HMMConfig(n_states=2, n_iter=200, tol=1e-3, random_state=27)
    first = LatentStateModel(config).fit(training, ["x0", "x1"])
    second = LatentStateModel(config).fit(training, ["x0", "x1"])
    posterior = first.filter(training)
    np.testing.assert_array_equal(posterior, second.filter(training))
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-12)
    assert np.isfinite(first.diagnostics_["log_likelihood"])
    assert np.mean(posterior.max(axis=1)) > 0.9
    assert first.fit_end == training.timestamp.max()
    before = json.dumps(first.diagnostics_, sort_keys=True, allow_nan=False)
    future = training.iloc[-20:].copy()
    future["timestamp"] += pd.Timedelta(days=30)
    future[["x0", "x1"]] += 100
    future["raw_realized_volatility"] = 1e6
    first.filter(future)
    assert json.dumps(first.diagnostics_, sort_keys=True, allow_nan=False) == before
    np.testing.assert_allclose(first.diagnostics_["state_occupancy"], posterior.mean(axis=0))
    expected_edges = np.unique(np.quantile(training.raw_realized_volatility, [0, 1 / 3, 2 / 3, 1]))
    np.testing.assert_allclose(first.diagnostics_["volatility_bucket_edges"], expected_edges)
    minute = training.minute_of_session.iloc[0]
    minute_rows = [
        row
        for row in first.diagnostics_["occupancy_by_minute"]
        if row["minute_of_session"] == minute
    ]
    np.testing.assert_allclose(
        [row["probability"] for row in minute_rows],
        posterior[training.minute_of_session.eq(minute)].mean(axis=0),
    )


def test_iteration_exhaustion_is_not_convergence_and_does_not_trigger_diag_fallback():
    config = HMMConfig(n_states=2, n_iter=2, tol=1e-12, retry_seed_offsets=[0, 1])
    model = LatentStateModel(config)
    with pytest.raises(HMMFitError) as caught:
        model.fit(synthetic_training(), ["x0", "x1"])
    attempts = caught.value.diagnostics["attempts"]
    assert len(attempts) == 2
    assert all(attempt["monitor_converged"] for attempt in attempts)
    assert all(not attempt["converged"] for attempt in attempts)
    assert all(attempt["covariance_type"] == "full" for attempt in attempts)
    assert caught.value.diagnostics["fallback_reason"] is None
    with pytest.raises(RuntimeError, match="fitted"):
        model.filter(synthetic_training())


def test_numerical_full_failures_retry_before_genuine_diagonal_fit(monkeypatch):
    original_fit = GaussianHMM.fit

    def unstable_full(self, values, lengths=None):
        if self.covariance_type == "full":
            warnings.warn("Covariance factorization failed", RuntimeWarning)
            raise np.linalg.LinAlgError("Covariance is not positive definite")
        return original_fit(self, values, lengths=lengths)

    monkeypatch.setattr(GaussianHMM, "fit", unstable_full)
    config = HMMConfig(n_states=2, n_iter=200, tol=1e-3, retry_seed_offsets=[0, 3])
    training = synthetic_training()
    model = LatentStateModel(config).fit(training, ["x0", "x1"])
    assert model.diagnostics_["covariance_type"] == "diag"
    assert "positive definite" in model.diagnostics_["fallback_reason"]
    attempts = model.diagnostics_["attempts"]
    assert [attempt["seed"] for attempt in attempts[:2]] == [42, 45]
    assert all(attempt["status"] == "failed" for attempt in attempts[:2])
    assert all(attempt["warnings"] for attempt in attempts[:2])
    assert attempts[-1]["status"] == "accepted"
    np.testing.assert_allclose(model.filter(training).sum(axis=1), 1.0)
    assert np.mean(model.filter(training).max(axis=1)) > 0.9


def test_numerical_seed_retry_can_recover_without_changing_covariance_type(monkeypatch):
    original_fit = GaussianHMM.fit

    def unstable_first_seed(self, values, lengths=None):
        if self.random_state == 42:
            raise np.linalg.LinAlgError("Singular covariance")
        return original_fit(self, values, lengths=lengths)

    monkeypatch.setattr(GaussianHMM, "fit", unstable_first_seed)
    config = HMMConfig(n_states=2, n_iter=200, tol=1e-3, retry_seed_offsets=[0, 1])
    model = LatentStateModel(config).fit(synthetic_training(), ["x0", "x1"])
    assert model.diagnostics_["covariance_type"] == "full"
    assert model.diagnostics_["fallback_reason"] is None
    assert model.diagnostics_["selected_seed"] == 43
    assert np.isfinite(model.filter(synthetic_training())).all()


def test_exhausted_numerical_retries_fail_with_serializable_audit(monkeypatch):
    def unstable_fit(self, values, lengths=None):
        raise np.linalg.LinAlgError("Singular covariance")

    monkeypatch.setattr(GaussianHMM, "fit", unstable_fit)
    config = HMMConfig(n_states=2, retry_seed_offsets=[0, 1])
    training = synthetic_training()
    with pytest.raises(HMMFitError) as caught:
        LatentStateModel(config).fit(training, ["x0", "x1"])
    diagnostics = caught.value.diagnostics
    assert [attempt["covariance_type"] for attempt in diagnostics["attempts"]] == [
        "full",
        "full",
        "diag",
        "diag",
    ]
    assert all(attempt["numerical_failure"] for attempt in diagnostics["attempts"])
    assert diagnostics["status"] == "failed"
    assert pd.Timestamp(diagnostics["fit_end"]) == training.timestamp.max()
    json.dumps(diagnostics, allow_nan=False)


def test_fit_likelihood_is_sum_of_independent_finite_contiguous_sequences():
    training = synthetic_training()
    warmups = [row for start in range(0, len(training), 60) for row in range(start, start + 3)]
    training.loc[warmups + [25], ["x0", "x1"]] = np.nan
    # The timestamp gap must reset even if stale segment metadata does not.
    training.loc[40:59, "timestamp"] += pd.Timedelta(minutes=5)
    config = HMMConfig(n_states=2, n_iter=200, tol=1e-3, random_state=27)
    model = LatentStateModel(config).fit(training, ["x0", "x1"])
    slices = [(3, 25), (26, 40), (40, 60)]
    slices += [(start + 3, start + 60) for start in range(60, len(training), 60)]
    independently_scored = sum(
        model.model_.score(training.iloc[start:stop][["x0", "x1"]].to_numpy())
        for start, stop in slices
    )
    assert model.diagnostics_["log_likelihood"] == pytest.approx(independently_scored)
    posterior = model.filter(training)
    assert np.isnan(posterior[warmups + [25]]).all()
    np.testing.assert_allclose(posterior[40], model.filter(training.iloc[40:60])[0], atol=1e-12)


def test_log_space_filter_does_not_underflow_on_unlikely_observations():
    posterior = analytical_model().filter(observations([1e3, -1e3, 1e3]))
    assert np.isfinite(posterior).all()
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-12)
