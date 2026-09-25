"""Deterministically fitted Gaussian HMMs and causal, segment-reset filtering."""

import logging
import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from regime_retrieval.config import HMMConfig
from regime_retrieval.models.diagnostics import training_diagnostics


class HMMFitError(ValueError):
    """No acceptable fit; ``diagnostics`` contains the complete retry audit."""

    def __init__(self, reason: str, diagnostics: dict) -> None:
        super().__init__(reason)
        self.diagnostics = diagnostics


class _FitRejection(ValueError):
    def __init__(self, reason: str, *, numerical: bool) -> None:
        super().__init__(reason)
        self.numerical = numerical


class _CaptureLog(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(f"{record.levelname}: {record.getMessage()}")


def _layout(
    frame: pd.DataFrame, feature_columns: list[str], interval: pd.Timedelta | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    required = ["timestamp", "session", "segment_id", *feature_columns]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"HMM input is missing columns: {missing}")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("HMM timestamps must be present and sorted")
    if timestamps.duplicated().any():
        raise ValueError("HMM timestamps must be unique")
    if frame[["session", "segment_id"]].isna().any().any():
        raise ValueError("HMM session and segment identities must be present")
    values = frame.loc[:, feature_columns].to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=1)
    starts = np.ones(len(frame), dtype=bool)
    if len(frame) > 1:
        sessions = frame["session"].to_numpy()
        segments = frame["segment_id"].to_numpy()
        starts[1:] = (
            (sessions[1:] != sessions[:-1]) | (segments[1:] != segments[:-1]) | ~finite[:-1]
        )
        if interval is not None:
            starts[1:] |= timestamps.diff().iloc[1:].ne(interval).to_numpy()
    return values, finite, starts, timestamps


def _parameters(model: GaussianHMM, n_features: int) -> tuple[np.ndarray, ...]:
    """Validate actual fitted parameters; never repair or jitter them silently."""
    states = model.n_components
    means = np.asarray(model.means_, dtype=float)
    covariances = np.asarray(model.covars_, dtype=float)
    transitions = np.asarray(model.transmat_, dtype=float)
    start = np.asarray(model.startprob_, dtype=float)
    for name, values, shape in (
        ("means", means, (states, n_features)),
        ("covariances", covariances, (states, n_features, n_features)),
        ("transitions", transitions, (states, states)),
        ("start probabilities", start, (states,)),
    ):
        if values.shape != shape or not np.isfinite(values).all():
            raise _FitRejection(f"Invalid or nonfinite {name}", numerical=True)
    for name, values, total in (
        ("transitions", transitions, transitions.sum(axis=1)),
        ("start probabilities", start, start.sum()),
    ):
        if (values < 0).any() or not np.allclose(total, 1.0, rtol=1e-8, atol=1e-10):
            raise _FitRejection(f"Unnormalized {name}", numerical=True)
    for covariance in covariances:
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
            raise _FitRejection("Nonsymmetric covariance", numerical=True)
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise _FitRejection("Covariance is not positive definite", numerical=True) from error
    return means, covariances, transitions, start


def _forward_filter(
    values: np.ndarray,
    finite: np.ndarray,
    starts: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    transitions: np.ndarray,
    start: np.ndarray,
) -> np.ndarray:
    """Forward recursion only: no backward pass and no smoothed predict_proba."""
    positions = np.flatnonzero(finite)
    posterior = np.full((len(values), len(start)), np.nan)
    if not len(positions):
        return posterior
    observations = values[positions]
    emissions = np.empty((len(positions), len(start)))
    normalizer = values.shape[1] * np.log(2 * np.pi)
    for state in range(len(start)):
        cholesky = np.linalg.cholesky(covariances[state])
        residual = solve_triangular(
            cholesky, (observations - means[state]).T, lower=True, check_finite=False
        )
        emissions[:, state] = -0.5 * (
            normalizer
            + 2 * np.log(np.diag(cholesky)).sum()
            + np.einsum("ij,ij->j", residual, residual)
        )
    if not np.isfinite(emissions).all():
        raise _FitRejection("Nonfinite Gaussian log emissions", numerical=True)
    with np.errstate(divide="ignore"):
        log_transitions = np.log(transitions)
        log_start = np.log(start)
    previous = None
    for index, position in enumerate(positions):
        if starts[position] or previous is None:
            prior = log_start
        else:
            prior = logsumexp(previous[:, None] + log_transitions, axis=0)
        filtered = prior + emissions[index]
        evidence = logsumexp(filtered)
        if not np.isfinite(evidence):
            raise _FitRejection("Nonfinite filtering evidence", numerical=True)
        previous = filtered - evidence
        posterior[position] = np.exp(previous)
    return posterior


def _numerical_exception(error: Exception) -> bool:
    if isinstance(error, _FitRejection):
        return error.numerical
    if isinstance(error, (np.linalg.LinAlgError, FloatingPointError, OverflowError)):
        return True
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "covar",
            "cholesky",
            "positive definite",
            "positive-definite",
            "singular",
            "ill-conditioned",
            "nan",
            "infinity",
            "infs",
            "nonfinite",
            "finite values",
            "must sum to 1",
            "sum to 1.0",
        )
    )


class LatentStateModel:
    """One frozen model for both historical and query observations in one refit.

    Inputs retain prepared ``timestamp``, ``session`` and ``segment_id`` columns.
    A call starts from the fitted start distribution, then resets on a session,
    segment, time gap, or any nonfinite selected feature. To carry a posterior
    into a query, filter its preceding contiguous rows in the same call. State
    coordinates are specific to this fit, never aligned to independent refits.
    """

    def __init__(self, config: HMMConfig) -> None:
        self.config = config.model_copy(deep=True)
        self.feature_columns: list[str] = []
        self.model_: GaussianHMM | None = None
        self.fit_end: pd.Timestamp | None = None
        self.diagnostics_: dict = {}
        self._interval: pd.Timedelta | None = None

    def fit(self, training: pd.DataFrame, feature_columns: list[str]) -> "LatentStateModel":
        self.model_ = None
        self.fit_end = None
        self._interval = None
        self.feature_columns = list(feature_columns)
        self.diagnostics_ = {
            "status": "failed",
            "converged": False,
            "fit_start": None,
            "fit_end": None,
            "feature_columns": self.feature_columns.copy(),
            "attempts": [],
            "fallback_reason": None,
        }
        try:
            if not self.feature_columns or len(set(feature_columns)) != len(feature_columns):
                raise ValueError("HMM feature columns must be nonempty and unique")
            values, finite, starts, timestamps = _layout(training, self.feature_columns, None)
            if training.empty:
                raise ValueError("Cannot fit HMM on empty training rows")
            self.fit_end = timestamps.max()
            self.diagnostics_.update(
                fit_start=timestamps.min().isoformat(),
                fit_end=self.fit_end.isoformat(),
                training_rows=len(training),
                finite_training_rows=int(finite.sum()),
            )
            same_segment = training["session"].eq(training["session"].shift()) & training[
                "segment_id"
            ].eq(training["segment_id"].shift())
            differences = timestamps.diff()[same_segment]
            if not differences.empty:
                self._interval = differences.min()
                starts |= timestamps.diff().ne(self._interval).to_numpy()
            if finite.sum() < self.config.n_states:
                raise ValueError("Fewer finite training rows than HMM states")
        except (ValueError, TypeError, KeyError) as error:
            self.diagnostics_["reason"] = str(error)
            raise HMMFitError(str(error), self.diagnostics_) from error

        positions = np.flatnonzero(finite)
        boundaries = np.flatnonzero(starts[positions])
        lengths = np.diff(np.append(boundaries, len(positions))).tolist()
        observations = values[finite]
        self.diagnostics_["sequence_lengths"] = lengths
        self.diagnostics_["bar_interval_seconds"] = (
            self._interval.total_seconds() if self._interval is not None else None
        )
        covariance_types = [self.config.covariance_type]
        if self.config.covariance_type == "full":
            covariance_types.append("diag")
        numerical_reasons = []
        for covariance_type in covariance_types:
            if covariance_type != self.config.covariance_type:
                if not numerical_reasons:
                    break
                self.diagnostics_["fallback_reason"] = "; ".join(dict.fromkeys(numerical_reasons))
            for offset in self.config.retry_seed_offsets:
                seed = self.config.random_state + offset
                candidate = GaussianHMM(
                    n_components=self.config.n_states,
                    covariance_type=covariance_type,
                    n_iter=self.config.n_iter,
                    tol=self.config.tol,
                    min_covar=self.config.min_covar,
                    random_state=seed,
                    implementation="log",
                )
                attempt = {
                    "seed": seed,
                    "covariance_type": covariance_type,
                    "status": "failed",
                    "converged": False,
                    "log_likelihood": None,
                    "iterations": 0,
                    "warnings": [],
                    "logging": [],
                }
                captured_log = _CaptureLog()
                logger = logging.getLogger("hmmlearn")
                logger.addHandler(captured_log)
                caught: list = []
                try:
                    with (
                        warnings.catch_warnings(record=True) as caught,
                        threadpool_limits(limits=1),
                    ):
                        warnings.simplefilter("always")
                        candidate.fit(observations, lengths=lengths)
                        means, covariances, transitions, start = _parameters(
                            candidate, len(self.feature_columns)
                        )
                        history = np.asarray(candidate.monitor_.history, dtype=float)
                        attempt["iterations"] = int(candidate.monitor_.iter)
                        attempt["monitor_converged"] = bool(candidate.monitor_.converged)
                        attempt["likelihood_history"] = [
                            float(value) if np.isfinite(value) else None for value in history
                        ]
                        if not np.isfinite(history).all():
                            raise _FitRejection("Nonfinite EM likelihood", numerical=True)
                        gain = float(history[-1] - history[-2]) if len(history) >= 2 else None
                        attempt["final_improvement"] = gain
                        tolerance = (
                            32 * np.finfo(float).eps * max(1.0, abs(history[-1]))
                            if len(history)
                            else 0.0
                        )
                        if gain is None or gain < -tolerance or gain >= self.config.tol:
                            reason = (
                                "EM likelihood decreased"
                                if gain is not None and gain < -tolerance
                                else "EM did not converge before iteration exhaustion"
                            )
                            raise _FitRejection(reason, numerical=False)
                        attempt["converged"] = True
                        likelihood = float(candidate.score(observations, lengths=lengths))
                        if not np.isfinite(likelihood):
                            raise _FitRejection("Nonfinite fitted likelihood", numerical=True)
                        attempt["log_likelihood"] = likelihood
                        posterior = _forward_filter(
                            values, finite, starts, means, covariances, transitions, start
                        )
                        summaries = training_diagnostics(
                            training,
                            self.feature_columns,
                            posterior,
                            covariances,
                            transitions,
                            starts,
                        )
                        numerical_warnings = [
                            str(item.message)
                            for item in caught
                            if issubclass(item.category, RuntimeWarning)
                        ]
                        if numerical_warnings:
                            raise _FitRejection("; ".join(numerical_warnings), numerical=True)
                    attempt["status"] = "accepted"
                except (
                    ValueError,
                    FloatingPointError,
                    OverflowError,
                    np.linalg.LinAlgError,
                ) as error:
                    numerical = _numerical_exception(error) or any(
                        issubclass(item.category, RuntimeWarning) for item in caught
                    )
                    attempt.update(reason=str(error), numerical_failure=numerical)
                    if numerical and covariance_type == "full":
                        numerical_reasons.append(str(error))
                finally:
                    logger.removeHandler(captured_log)
                    attempt["warnings"] = [
                        f"{item.category.__name__}: {item.message}" for item in caught
                    ]
                    attempt["logging"] = captured_log.messages
                    if hasattr(candidate, "monitor_"):
                        attempt["iterations"] = int(candidate.monitor_.iter)
                    self.diagnostics_["attempts"].append(attempt)
                if attempt["status"] == "accepted":
                    self.model_ = candidate
                    self.diagnostics_.update(
                        status="accepted",
                        converged=True,
                        log_likelihood=likelihood,
                        iterations=attempt["iterations"],
                        covariance_type=covariance_type,
                        selected_seed=seed,
                        state_means=means.tolist(),
                        state_covariances=covariances.tolist(),
                        transition_matrix=transitions.tolist(),
                        start_probabilities=start.tolist(),
                        **summaries,
                    )
                    return self
        reason = "No acceptable HMM fit after deterministic retries"
        self.diagnostics_["reason"] = reason
        raise HMMFitError(reason, self.diagnostics_)

    def filter(self, transformed: pd.DataFrame) -> np.ndarray:
        """Return aligned P(S_t | X_<=t), with missing feature rows left as NaN."""
        if self.model_ is None:
            raise RuntimeError("LatentStateModel must be fitted before filtering")
        values, finite, starts, _ = _layout(transformed, self.feature_columns, self._interval)
        with threadpool_limits(limits=1):
            parameters = _parameters(self.model_, len(self.feature_columns))
            return _forward_filter(values, finite, starts, *parameters)
