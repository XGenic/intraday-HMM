# Implemented methodology

The question is whether HMM posterior trajectories improve the similarity space
for historical-analogue forecasting relative to standardized raw-feature
trajectories. The frozen configuration is [alpaca_2019.yaml](../configs/alpaca_2019.yaml).
The [original plan](../experiment-plan.md) and [results](../experiment-results.md)
separate the intended experiment from the empirical outcome.

## Data and features

- SPY five-minute bars on the XNYS regular-session calendar, including holidays,
  daylight-saving changes, and early closes. Canonical timestamps label bar-end
  availability. Vendor start labels are shifted forward five minutes.
- Unsorted or duplicated timestamps, invalid OHLCV, and off-grid rows fail.
  Observed-session coverage below 80% fails. Missing bars and whole sessions are
  reported, never filled; session ordinals include missing scheduled sessions.
- Features: log return; 12-bar sample standard deviation of log returns;
  `(high - low) / close`; volume surprise; and session VWAP distance.
- Volume surprise uses training-slot medians and 1.4826 times the median absolute
  deviation. A zero dispersion uses `max(abs(median), 1)`. Unseen slots stay missing.
- Session VWAP cumulatively weights per-bar vendor VWAP by volume. If unavailable,
  typical price is used. Feature means, scales, and volume statistics are fitted
  before each evaluation block and then frozen.
- Returns and volatility reset at gaps and session boundaries. Contexts require
  12 consecutive finite-feature bars; outcomes require contiguous bars through
  the maximum 60-minute horizon. Evaluation consequently excludes early-session
  warmup and late-session queries, rather than representing the entire trading day.

## Models and retrieval

After at least 126 training sessions, refit weekly using expanding history. The
Gaussian HMM uses six states, full covariance, 500 iterations, tolerance `1e-4`,
and seed 42. Retry offsets are `[0, 1, 2]`. Iteration exhaustion is not convergence.
Numerical covariance failures can trigger a recorded diagonal fallback; ordinary
nonconvergence alone cannot. A failed fit excludes the entire paired block.

Inference is the causal forward recursion `P(S_t | X_<=t)`, not a backward-smoothed
posterior. Filtering resets at missing features, gaps, and session boundaries.
Historical and query representations use the same fitted HMM within each block.
State numbers have no shared meaning across independently fitted models.

Raw kNN flattens 12 × 5 standardized features. HMM trajectory flattens 12 × 6
posterior probabilities; HMM current-state uses the final six probabilities.
Posteriors remain on their probability scale. Frozen retrieval uses exact Euclidean
distance, 50 neighbors, and a maximum of three neighbors per historical session.
Fewer legal neighbors are retained without padding. Cosine distance is supported
but was not used in the reported experiment.

A candidate's full maximum-horizon outcome must already be known at the query.
For embargo zero this availability constraint is sufficient. A positive embargo
`N` also requires `candidate_session_index < query_session_index - N`, excluding
the current and preceding `N` exchange sessions. Primary embargo is one;
sensitivities are zero, one, and five. All six methods and embargo variants share
the same scored query cohort and refit models.

The unconditional control uses all eligible candidate contexts. The time-of-day
control additionally matches the session-minute offset. Momentum extrapolates
the recent context log return; its direction score is a fixed sign indicator,
not a calibrated probability.

## Scoring and inference

Each historical distribution supplies a median, mean, probability of a positive
return, quantiles, dispersion, and effective sample size. The primary point
forecast is the median. Horizons are 5, 15, 30, and 60 minutes.

Primary comparison: HMM trajectory minus raw kNN in absolute error. Reported
metrics also include RMSE, signed error, Spearman correlation, Brier score,
log loss, and interval coverage/width. HMM current-state is a descriptive ablation.

Bootstrap intervals resample whole query sessions, preserving all paired
observations and their original session sizes. There are 1,000 draws and 95%
percentile intervals. Three chronological folds are assigned from planned
evaluation dates independently of later fit success.

A favorable report conclusion requires an aggregate interval strictly below zero
at every horizon and favorable point differences in every configured fold.
Consistently adverse evidence yields a negative directional conclusion; missing
folds or mixed evidence yield an inconclusive classification. Synthetic inputs
always produce demonstration-only conclusions. The decision to discontinue this
configuration is distinct from a statistical claim of equivalence.

Intervals are exploratory and pointwise, without multiple-comparison correction.
Dependence between sessions and overlapping training history can make them
optimistic. A single historical period cannot establish broader market stability.
Raw vendor history is a retrospective snapshot and may include later revisions.

## Audit and optional execution

Every historical-distribution forecast retains its selected members, returns,
outcome endpoints, fit cutoffs, ranks, distances, and model identities. Large
unconditional cohorts are streamed to Parquet. Saved examples retain standardized
features and filtered probabilities; model parameters are JSON, not pickles.

The test suite covers deterministic fitting, forward filtering versus smoothing,
eligibility, future perturbations, common cohorts, bootstrap pairing, report
conclusion guards, and data/calendar edge cases. The real-data artifact audit
independently reconstructs outcomes and selected forecasts from persisted files.

Optional long/flat simulation uses a fixed median threshold, next contiguous bar
open entry, horizon-close exit, no overlapping positions per method/horizon, and
no overnight positions. Costs default to one basis point per side. Execution is
idealized without extra latency. Trading was disabled in the reported experiment
and never determines the research conclusion.
