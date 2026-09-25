# Intraday Latent-Regime Analog Retrieval
## Codex Handoff / Implementation Specification

### Project goal

Build a reproducible research pipeline that tests whether a **latent-state representation of intraday market behavior improves historical-analogue retrieval** relative to matching directly on normalized raw features.

The project is inspired by Nguyen (2018), *Hidden Markov Model for Stock Trading*. That paper fits an HMM to monthly S&P 500 OHLC data, compresses the current observation window into a likelihood, searches history for a window with a similar likelihood, and uses the subsequent historical move as a forecast. The useful idea is not the paper's exact implementation. The useful idea is:

```text
market sequence
    -> latent representation
    -> historical analogue retrieval
    -> conditional future-return distribution
```

The modern experiment should test that architecture on shorter intraday horizons with stricter methodology.

The central research question is:

> **Does an HMM-derived latent-state trajectory create a better similarity space for short-horizon historical analogue forecasting than k-nearest-neighbor retrieval on normalized raw intraday features?**

A negative answer is a successful research outcome. The objective is to build a system that makes the comparison difficult to fool ourselves about.

---

## 1. Scope

### V1 target

Use a single highly liquid instrument, preferably **SPY 5-minute regular-session bars**, and compare several forecasting approaches over multiple forward horizons.

Initial forecast horizons:

```text
+5 minutes
+15 minutes
+30 minutes
+60 minutes
```

Initial recent-context window:

```text
12 bars = 60 minutes
```

Initial retrieval count:

```text
k = 50
```

All of these must live in configuration rather than being hard-coded.

### V1 is NOT

Do not turn this into:

- an HFT execution engine;
- an options strategy;
- a live trading bot;
- a giant hyperparameter search;
- a deep-learning project;
- an HSMM project before the HMM experiment works;
- a dashboard-first project;
- a claim that HMM "predicts the market."

The first deliverable is a clean empirical comparison.

---

## 2. Scientific hypothesis

### Primary hypothesis

Let `X_t` be a normalized sequence of recent intraday features and let `Z_t` be the corresponding sequence of posterior HMM state probabilities.

Compare:

```text
raw retrieval:
X_t -> nearest historical X_i -> future returns

latent retrieval:
Z_t -> nearest historical Z_i -> future returns
```

The primary hypothesis is:

```text
Historical analogues selected in latent-state space contain more
out-of-sample information about subsequent returns than analogues
selected directly in normalized raw-feature space.
```

"More information" should be evaluated by forecast-quality metrics, not only trading P&L.

### Null

The HMM adds no useful representation. Any apparent improvement is explained by:

- ordinary similarity in the original features;
- time-of-day effects;
- volatility clustering;
- momentum / mean reversion;
- leakage;
- model tuning;
- chance.

The pipeline must make this null easy to support if that is what the data says.

---

## 3. Data contract

### Preferred V1 input

Do **not** make the project dependent on one market-data vendor.

The canonical input should be a local Parquet file with one row per 5-minute bar:

```text
timestamp
open
high
low
close
volume
```

Optional fields should be supported if present:

```text
vwap
trade_count
bid
ask
spread
```

Assumptions:

- timestamps are timezone-aware;
- bars are ordered;
- duplicate timestamps are rejected;
- V1 uses U.S. regular trading hours only;
- session calendar should use an exchange-calendar library rather than naïve weekday logic;
- raw data files should never be modified in place.

Implement a provider interface, but only the local-file provider is required for V1.

Example abstraction:

```python
class BarDataProvider(Protocol):
    def load(
        self,
        symbol: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        ...
```

Later providers can be added for Polygon/Massive, Databento, Alpaca, etc. without changing the research pipeline.

### Data validation

Fail loudly on:

- unsorted timestamps;
- duplicated bars;
- nonpositive prices;
- negative volume;
- bars outside the configured session when RTH-only mode is active;
- missing required columns;
- obviously malformed session lengths.

Generate a short data-quality report:

```text
date range
number of sessions
number of bars
missing bars by session
duplicate count
largest timestamp gaps
OHLC consistency violations
```

Do not silently forward-fill missing market bars.

---

## 4. Feature construction

Features must be computable using information available **at or before the current bar only**.

Start with a deliberately small set.

### Core features

#### 1. Log return

```python
log_return_t = log(close_t / close_t_minus_1)
```

#### 2. Realized volatility

Use rolling standard deviation of 5-minute log returns.

Default:

```text
window = 12 bars
```

Do not center rolling windows.

#### 3. Normalized intrabar range

A reasonable starting definition:

```python
range_frac = (high - low) / close
```

Then standardize using training-only statistics.

An ATR-normalized variant can be added later.

#### 4. Volume surprise

Intraday volume has strong time-of-day seasonality.

Do **not** z-score volume globally.

Fit minute-of-session volume statistics using training data only:

```text
for each 5-minute slot:
    estimate historical mean/median and dispersion
```

Then compute a time-of-day-adjusted volume surprise.

A robust implementation using median and MAD is preferable if convenient.

#### 5. VWAP distance

If vendor VWAP is absent, compute cumulative session VWAP using only bars observed so far.

Example:

```python
typical_price = (high + low + close) / 3
session_vwap_t =
    cumulative_sum(typical_price * volume)
    / cumulative_sum(volume)

vwap_distance = (close - session_vwap_t) / close
```

Then standardize with training-only statistics.

### Optional context fields

Store, but do not necessarily feed into the first HMM:

```text
minute_of_session
day_of_week
overnight_gap
previous_session_return
```

These are useful for controls and stratification.

### Feature preprocessing

Implement a fitted transformer object.

All scalers, minute-of-day seasonal statistics, clipping thresholds, etc. must be fit on training data only.

Recommended flow:

```text
raw bars
 -> causal feature calculation
 -> training-only seasonality fit
 -> training-only standardization
 -> transformed feature matrix
```

Do not fit a scaler on the full dataset before walk-forward evaluation.

---

## 5. Market-state model

### V1 model

Use a multivariate Gaussian Hidden Markov Model.

Good initial library choice:

```text
hmmlearn
```

Default configuration:

```text
n_states = 6
covariance_type = "full"
n_iter = 500
tol = 1e-4
random_state = fixed
```

These values are starting defaults, not claims of optimality.

If full covariance is numerically unstable, fall back to diagonal covariance and record the reason in the report.

### Important: posterior probabilities, not only hard state labels

For every bar, compute:

```python
P(S_t = j | observed_sequence)
```

The main representation should retain the posterior probability vector.

For a 12-bar context and six states:

```text
posterior trajectory shape = (12, 6)
flattened dimension = 72
```

A hard sequence such as:

```text
2,2,2,5,5,5,1,1,1,1,1,1
```

throws information away and should be secondary.

### State-label switching

HMM state labels are arbitrary and may permute between refits.

Therefore:

**Do not compare state label "3" from one independently fitted model with state label "3" from another model.**

At every walk-forward refit:

1. fit the HMM using history available at that time;
2. transform the eligible historical analogue pool under that same fitted HMM;
3. transform the current query window under that same HMM;
4. compare query and candidate trajectories within this common model coordinate system.

This avoids requiring global state-label alignment across refits.

It is computationally heavier but scientifically cleaner.

### Refit cadence

Do not refit the HMM every 5 minutes in V1.

Configurable default:

```text
refit once per trading week
```

Sensitivity tests can later compare daily vs weekly vs monthly refits.

Within each refit block, use one model fitted only on data available before the block begins.

---

## 6. Analogue representations

Implement at least two comparable representations.

### A. Raw-feature trajectory

For the most recent `context_bars` bars:

```text
shape = context_bars x n_features
```

Flatten to one vector after training-only standardization.

Example:

```text
12 bars x 5 features = 60 dimensions
```

### B. HMM posterior trajectory

For the same context:

```text
shape = context_bars x n_states
```

Flatten to one vector.

Example:

```text
12 bars x 6 states = 72 dimensions
```

### Distance

Start with Euclidean distance on standardized vectors.

Also support cosine distance by config.

Do not add dynamic time warping until the basic experiment is complete.

---

## 7. Historical-neighbor eligibility

This is one of the most important parts of the project.

At query time `t`, a candidate historical analogue ending at `i` is eligible only if its entire forward outcome is already known at `t`.

For maximum horizon `H`:

```text
candidate_end + H <= query_time
```

This prevents a candidate's "future return" from accidentally using information that was not yet known at the query time.

The candidate context window must also be fully available.

### Purging / embargo

Add a configurable additional embargo.

Default experiment:

```text
embargo = 1 trading session
```

Run a sensitivity comparison with:

```text
embargo = max_forward_horizon only
embargo = 1 trading session
embargo = 5 trading sessions
```

This tests whether apparent skill is merely retrieval of nearly adjacent autocorrelated windows.

### Neighbor overlap

Do not allow many near-identical overlapping historical windows from the same local episode to dominate the `k` neighbors.

Implement one of:

```text
minimum time separation between selected neighbors
```

or

```text
at most N neighbors per trading session
```

Default:

```text
max 3 neighbors from the same session
```

Make configurable.

---

## 8. Retrieval methods to compare

The minimum comparison suite:

### Baseline 0: unconditional

Predict using historical forward-return statistics available at that time.

At minimum:

```text
historical mean return
historical median return
historical probability(return > 0)
```

### Baseline 1: same-time-of-day

Same as unconditional, but condition on the same 5-minute slot or a small time-of-day bucket.

This is important because intraday return distributions vary by clock time.

### Baseline 2: simple momentum / mean reversion

Use recent 60-minute return as a deliberately simple predictive signal.

Do not optimize complicated thresholds.

### Baseline 3: raw-feature kNN

Retrieve nearest historical windows using flattened normalized raw features.

### Model 1: HMM current-state only

Use only the latest posterior state vector.

This tests whether trajectory information matters.

### Model 2: HMM posterior-trajectory kNN

Main method:

```text
12-bar posterior trajectory -> k nearest historical posterior trajectories
```

### Optional later comparisons

After V1:

```text
PCA + kNN
autoencoder embedding + kNN
UMAP for visualization only
HSMM
DTW trajectory distance
contrastive representation learning
```

Do not implement these before the core experiment is working.

---

## 9. Forecast construction

For each method and each horizon, retrieve an eligible historical comparison set and compute:

```text
mean forward return
median forward return
probability(forward return > 0)
10th percentile
25th percentile
75th percentile
90th percentile
neighbor-distance statistics
effective neighbor count
```

The primary point forecast should initially be:

```text
median forward return
```

The binary probability forecast should be:

```text
fraction of neighbor forward returns > 0
```

Store the full neighbor IDs/timestamps used for every prediction so predictions are auditable.

---

## 10. Walk-forward evaluation

### Principle

Everything that can learn from data must learn from the past only.

At a high level:

```text
for each refit block:
    training history = all eligible data before block start
    fit preprocessing on training history
    fit HMM on transformed training history
    transform historical candidate windows with same HMM
    for each query in block:
        build query representation
        select only eligible historical analogues
        produce forecasts
```

### Minimum history

Do not begin predictions until enough training data exists.

Configurable default:

```text
minimum training history = 6 months
```

Prefer one year if the available dataset is large enough.

### Validation structure

V1 should produce results by chronological test fold, not only one giant aggregate.

Example:

```text
Fold A: earliest OOS segment
Fold B: middle OOS segment
Fold C: latest OOS segment
```

Exact dates depend on the dataset.

The report must show whether any advantage is stable or concentrated in one period.

### Hyperparameters

Avoid a giant parameter search.

For V1, freeze:

```text
context_bars = 12
k = 50
n_states = 6
distance = euclidean
refit = weekly
```

Only after V1 runs should a small sensitivity grid be allowed.

If tuning is added later, it must be nested inside the historical training period rather than using final OOS results.

---

## 11. Primary evaluation metrics

Trading return is secondary.

### Continuous forecast metrics

For each horizon:

```text
MAE
RMSE
mean signed error
Spearman correlation between forecast and realized return
```

### Directional probability metrics

For:

```text
P(return > 0)
```

compute:

```text
Brier score
log loss if probabilities are clipped safely
calibration table / reliability plot
```

### Conditional-distribution diagnostics

For analogue methods:

```text
neighbor distance
neighbor return dispersion
10-90% interval width
realized return percentile within neighbor distribution
```

This helps determine when the retrieval system is confident versus merely finding weak matches.

### Statistical comparison

Use paired forecast errors because all methods predict the same timestamps.

At minimum:

- compare model-vs-raw-kNN error differences;
- report session-clustered or block-bootstrap confidence intervals;
- avoid treating every 5-minute bar as independent.

Bootstrap whole trading sessions rather than individual bars.

A Diebold-Mariano-style test can be added later, but it is not required for the first working version.

---

## 12. Secondary trading simulation

Only after forecast evaluation works.

Use a deliberately simple rule so strategy tuning does not become the project.

Example:

```text
long if predicted median return > threshold
short if predicted median return < -threshold
otherwise flat
```

Default V1 can be long/flat only if simplicity is preferred.

Include:

```text
spread/slippage cost in basis points per side
turnover
gross return
net return
Sharpe
max drawdown
hit rate
average holding-period return
time in market
```

Costs must be configurable.

Do not call an approach economically useful because gross P&L is positive.

The main scientific conclusion must still come from forecast comparison.

---

## 13. Leakage tests

Write explicit automated tests for leakage.

These are not optional.

### Required invariants

For every prediction timestamp:

```text
latest feature input timestamp <= query timestamp
scaler fit_end < query/refit-block start
HMM fit_end < query/refit-block start
historical neighbor end < query timestamp
neighbor forward-outcome end <= query timestamp
```

If a 60-minute outcome is used:

```text
neighbor_end + 60 minutes <= query_timestamp
```

### Test future perturbation

Create a test where data after some cutoff is modified drastically.

Predictions before the cutoff must remain bit-for-bit or numerically identical.

This is one of the strongest pipeline sanity checks.

### Test candidate auditability

Randomly sample predictions and verify from stored metadata that every retrieved neighbor was legally available at forecast time.

---

## 14. Time-of-day controls

An HMM can "discover" trivial states such as:

```text
market open
lunch
closing hour
```

That is not necessarily useless, but we need to know whether it is happening.

The report should include:

```text
state occupancy by minute of session
state occupancy by realized-volatility bucket
state occupancy by return sign
state transition matrix
average feature vector by state
```

If one state is overwhelmingly an "opening bell" state, say so.

Also compare performance against the same-time-of-day baseline.

A useful latent model should add information beyond clock time alone.

---

## 15. Model diagnostics

For every refit block, save:

```text
convergence status
log likelihood
number of iterations
transition matrix
state means
state covariance matrices
state occupancy
```

Flag degenerate states:

```text
very low occupancy
near-singular covariance
one state absorbing almost everything
transition matrix nearly deterministic for implausible reasons
```

Do not silently accept a failed HMM fit.

If a fit fails:

1. retry with deterministic alternative initialization seeds;
2. log the failure;
3. if still unsuccessful, skip the block rather than injecting a future-informed fix.

---

## 16. Reproducibility

Use deterministic seeds wherever possible.

Every research run should write a run directory:

```text
runs/<timestamp_or_run_id>/
    config.yaml
    data_manifest.json
    predictions.parquet
    neighbors.parquet
    metrics.json
    fold_metrics.csv
    state_diagnostics/
    figures/
    report.html
```

`data_manifest.json` should include:

```text
symbol
raw file path or hash
date range
bar interval
row count
feature schema version
git commit if available
```

A run must be reproducible from its config plus raw input data.

---

## 17. Suggested repository structure

```text
intraday-regime-retrieval/
├─ README.md
├─ pyproject.toml
├─ configs/
│  └─ baseline.yaml
├─ data/
│  ├─ raw/
│  └─ processed/
├─ src/
│  └─ regime_retrieval/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ data/
│     │  ├─ providers.py
│     │  ├─ validation.py
│     │  └─ sessions.py
│     ├─ features/
│     │  ├─ causal.py
│     │  ├─ seasonality.py
│     │  └─ scaling.py
│     ├─ models/
│     │  ├─ hmm.py
│     │  └─ diagnostics.py
│     ├─ retrieval/
│     │  ├─ representations.py
│     │  ├─ neighbors.py
│     │  └─ eligibility.py
│     ├─ forecasting/
│     │  ├─ baselines.py
│     │  └─ analog_forecast.py
│     ├─ evaluation/
│     │  ├─ walk_forward.py
│     │  ├─ metrics.py
│     │  ├─ bootstrap.py
│     │  └─ trading.py
│     ├─ reporting/
│     │  ├─ figures.py
│     │  └─ report.py
│     └─ cli.py
└─ tests/
   ├─ test_data_validation.py
   ├─ test_causal_features.py
   ├─ test_neighbor_eligibility.py
   ├─ test_no_future_leakage.py
   ├─ test_future_perturbation.py
   └─ test_small_end_to_end.py
```

Keep modules small and research logic explicit.

---

## 18. Suggested dependencies

Keep the dependency set conservative.

Core:

```text
python >= 3.11
numpy
pandas
pyarrow
scipy
scikit-learn
hmmlearn
matplotlib
pydantic
pyyaml
exchange-calendars
jinja2
```

Testing:

```text
pytest
pytest-cov
```

Optional later:

```text
polars
numba
plotly
duckdb
```

Do not bring in a vector database. The candidate set is small enough for exact nearest-neighbor search in V1.

---

## 19. Configuration

Example `configs/baseline.yaml`:

```yaml
symbol: SPY
bar_interval: 5m
regular_hours_only: true

data:
  provider: parquet
  path: data/raw/SPY_5m.parquet

features:
  log_return: true
  realized_vol_window: 12
  range_fraction: true
  volume_tod_adjusted: true
  vwap_distance: true

context_bars: 12

hmm:
  n_states: 6
  covariance_type: full
  n_iter: 500
  tol: 0.0001
  random_state: 42
  refit_frequency: weekly

retrieval:
  k: 50
  distance: euclidean
  max_neighbors_per_session: 3
  embargo_sessions: 1

horizons_minutes:
  - 5
  - 15
  - 30
  - 60

walk_forward:
  min_training_sessions: 126

costs:
  bps_per_side: 1.0

report:
  output_dir: runs
```

All timestamps and horizons should be resolved through the actual session/bar index rather than assuming that 60 clock minutes always equals 12 valid bars across breaks or session boundaries.

---

## 20. CLI

Aim for a simple interface.

Examples:

```bash
python -m regime_retrieval.cli validate-data \
  --config configs/baseline.yaml
```

```bash
python -m regime_retrieval.cli run \
  --config configs/baseline.yaml
```

```bash
python -m regime_retrieval.cli report \
  --run runs/<run_id>
```

One `run` command may perform the full pipeline once stable.

---

## 21. Output report

Generate a self-contained HTML report.

### Header

```text
run ID
symbol
date range
number of sessions
config summary
git commit
```

### Data quality

Show:

```text
missing bars
session coverage
feature distributions
```

### HMM interpretation

Show:

```text
state means
state covariance summaries
transition matrix
state occupancy
state occupancy by time of day
```

Do not assign narrative names like "bull state" automatically. If labels are added, keep them descriptive and data-based.

### Forecast comparison

For each horizon compare:

```text
unconditional
same-time-of-day
momentum baseline
raw-feature kNN
HMM current-state
HMM trajectory kNN
```

Show:

```text
MAE
RMSE
Spearman
Brier score
fold-by-fold values
bootstrap confidence intervals for key pairwise differences
```

The primary pairwise comparison is:

```text
HMM trajectory kNN vs raw-feature kNN
```

### Retrieval examples

Include several audit examples:

```text
query timestamp
query recent price/feature trajectory
nearest raw-feature matches
nearest latent-state matches
their historical dates
their distances
their subsequent returns
```

This is important for understanding what the representation is actually doing.

### Trading section

Secondary.

Show gross/net metrics with clearly stated cost assumptions.

---

## 22. Acceptance criteria for V1

The project is "done" when all of the following are true:

1. A local 5-minute SPY dataset can be validated and ingested from Parquet.
2. Causal features are generated with tests confirming no future bars are used.
3. Time-of-day volume normalization is fit using training data only.
4. Raw-feature kNN forecasts work in a strict walk-forward loop.
5. An HMM can be fit at scheduled refit points and produces posterior state probabilities.
6. Historical candidate windows and the current query are transformed under the same fitted HMM.
7. HMM posterior-trajectory kNN forecasts work.
8. Neighbor eligibility guarantees that the candidate's full forward outcome was already observable at query time.
9. Future-perturbation leakage tests pass.
10. Forecast metrics are reported by chronological fold for all required baselines.
11. The report explicitly compares HMM trajectory retrieval with raw-feature retrieval.
12. The pipeline produces an auditable `neighbors.parquet` showing exactly which historical analogues created every forecast.
13. A self-contained HTML report is generated.
14. The conclusion can honestly say one of:
    - latent retrieval improved OOS forecasting;
    - latent retrieval did not improve OOS forecasting;
    - evidence is unstable / inconclusive.

Do not redefine "success" to mean profitable backtest.

---

## 23. Milestone plan

### Milestone 1 — data + causal features

Build:

```text
repo skeleton
config loading
Parquet provider
exchange-session filtering
data validation
causal features
time-of-day normalization
tests
```

Stop and verify the data pipeline before any HMM work.

### Milestone 2 — raw retrieval baseline

Build:

```text
context-window builder
eligible historical-candidate index
raw-feature representation
exact kNN
forward-return labels
unconditional baseline
same-time-of-day baseline
simple momentum baseline
walk-forward evaluator
```

At this point the project should already produce a legitimate research report without an HMM.

### Milestone 3 — HMM representation

Build:

```text
HMM wrapper
weekly refit schedule
posterior state probabilities
trajectory representation
state diagnostics
HMM current-state baseline
HMM trajectory kNN
```

### Milestone 4 — leakage gauntlet

Implement and pass:

```text
neighbor outcome availability test
scaler/HMM fit-cutoff tests
future-perturbation test
sampled neighbor audit
embargo sensitivity
```

### Milestone 5 — evaluation + report

Add:

```text
fold metrics
session bootstrap
retrieval examples
state/time-of-day diagnostics
optional simple trading simulation
self-contained HTML report
```

Only after this milestone should we consider additional models.

---

## 24. First implementation sequence for Codex

Start here and keep commits logically separated.

1. Create `pyproject.toml`, package structure, config model, and baseline YAML.
2. Implement Parquet ingestion and strict data validation.
3. Implement exchange-session handling.
4. Implement the causal feature pipeline.
5. Add unit tests for every feature using tiny hand-constructed bar sequences.
6. Implement training-only time-of-day volume normalization.
7. Implement forward-return labels for 5/15/30/60 minutes.
8. Implement historical-candidate eligibility with max-horizon purging.
9. Implement raw-feature context vectors and exact kNN.
10. Build a small walk-forward baseline before touching HMM code.
11. Add leakage tests, especially future perturbation.
12. Once raw retrieval is trustworthy, implement the HMM wrapper.
13. Compute posterior trajectories for history and query under the same model.
14. Add HMM retrieval to the same evaluator.
15. Generate the first comparison report.
16. Only then consider performance optimization or optional extensions.

---

## 25. Engineering principles

The research design owns the code, not the reverse.

Keep these rules:

> **No future data in preprocessing, fitting, candidate selection, or outcome construction.**

> **Every prediction must be auditable back to the historical neighbors that generated it.**

> **Raw-feature retrieval is the primary control. If HMM retrieval cannot beat it, the latent representation did not earn its complexity.**

> **Do not tune against the final test period.**

> **Do not optimize the trading rule before establishing forecast improvement.**

> **A negative result is a valid finished result.**

> **Prefer explicit code over clever abstraction in leakage-sensitive sections.**

---

## 26. Potential failure modes to watch

### HMM merely learns time of day

Diagnose via state occupancy by minute-of-session.

### HMM merely learns volatility buckets

This can still be meaningful, but compare against raw volatility-conditioned retrieval.

### Candidate leakage

The most dangerous bug. The candidate's future return must end before the query timestamp.

### Overlapping-neighbor pseudo-sample size

Limit repeated neighbors from one historical episode/session.

### State collapse

Watch occupancy and covariance matrices.

### State-label switching

Never compare labels across independent model fits without a common coordinate system.

### Unstable posterior trajectories

Measure retrieval-distance distributions and performance by refit block.

### False edge from one crisis period

Report chronological folds and remove-one-regime sensitivity later.

### Transaction-cost illusion

Trading simulation is secondary and must show conservative costs.

### Hyperparameter fishing

Freeze the V1 config before looking at the final OOS comparison.

---

## 27. Post-V1 extensions

Only if V1 is informative.

### Representation comparisons

```text
PCA
autoencoder
contrastive encoder
transformer embedding
HSMM
switching autoregression
```

### Better distances

```text
Mahalanobis distance
dynamic time warping
learned metric
```

### Better market context

```text
SPY + QQQ + IWM
ES / NQ futures
VIX changes
breadth
order-flow imbalance
spread
depth
auction imbalance
```

### Conditional retrieval

Require neighbors to match on:

```text
time of day
volatility quartile
overnight gap regime
macro-event flag
```

### Cross-asset analogues

Ask whether latent market-state trajectories transfer across SPY, QQQ, ES, or NQ after normalization.

### Online use

Only after research validation:

```text
stream bars
update features
infer posterior state
retrieve historical analogues
display conditional return distribution
```

Still not an auto-trader by default.

---

## 28. Relationship to the original paper

The original paper should be treated as inspiration, not an implementation blueprint.

Keep:

```text
latent market-state modeling
historical analogue idea
out-of-sample comparison
```

Replace:

```text
monthly raw price levels
OHLC treated as independent sequences
single scalar sequence-likelihood matching
single historical analogue
weak historical-average benchmark
partially contaminated model-selection procedure
```

With:

```text
normalized intraday features
multivariate HMM
posterior-state trajectory
k historical analogues
strong raw-feature retrieval baseline
strict walk-forward refitting and purging
```

The study question is therefore not:

> "Can we reproduce Nguyen's reported trading returns?"

It is:

> **"Does latent regime representation improve historical similarity search for intraday forecasting when tested under strict causal evaluation?"**

That is the project.

---

## Implementation notes — Milestones 1–5

All five milestones are implemented: the six-method walk-forward comparison,
explicit leakage tests, paired session-bootstrap intervals, and self-contained
report. A secondary long/flat simulation is available but disabled by default.
There is no tuning grid, live trading, or automatic claim of market skill.

### Running the implemented pipeline

```bash
uv sync --extra dev
uv run python -m regime_retrieval.cli validate-data --config configs/baseline.yaml
uv run python -m regime_retrieval.cli run --config configs/baseline.yaml
uv run python -m regime_retrieval.cli report --run runs/<run_id>
uv run pytest -q
```

Supply your own Parquet data at `data/raw/SPY_5m.parquet`, or change `data.path`.
Paths in YAML resolve relative to the YAML file, not the working directory.
The baseline retains 126 training sessions, weekly refits, 12-bar contexts,
50 requested neighbors, and the 5/15/30/60-minute horizons. A missing source file
fails explicitly; the program never substitutes synthetic data.

### Resolved data and leakage contracts

- `data.timestamp_label` explicitly selects `start` or `end`; default `end`.
  Internally every timestamp is a UTC bar-end availability time. Naive,
  duplicated, unsorted, nonfinite, off-grid, and inconsistent OHLCV data fail.
- `XNYS` supplies holidays, DST, and early closes. The ingestion command rejects
  non-RTH rows; `filter_regular_hours` is a separate explicit opt-in helper.
- Missing bars and whole sessions are reported, never filled. Observed sessions
  below `data.min_session_coverage` (default 80%) fail, including partial edge
  sessions. Entirely missing sessions retain their exchange-calendar ordinals.
- Returns and rolling sample volatility reset at session boundaries and missing
  bars. Contexts require finite features on consecutive bars in one session.
  Forward labels are simple close-to-close returns on that same contiguous
  session index; they never borrow the next day's bars or skip gaps.
- Session VWAP accumulates observed volume only. Optional vendor `vwap` is
  interpreted as a **per-bar** volume-weighted price and then accumulated;
  otherwise typical price is used. Zero cumulative volume leaves VWAP undefined.
  Optional context currently includes day of week; overnight-gap fields are not
  synthesized when prior scheduled-close information may be missing.
- Volume surprise uses training-only per-session-minute median and scaled MAD.
  Zero MAD uses `max(abs(training_slot_median), 1)`; unseen slots remain missing.
  Feature standardization is frozen before each refit block. Numerically constant
  features use scale one, rather than amplifying floating-point roundoff.
- A historical candidate's maximum-horizon outcome must end at or before the
  query timestamp. With `embargo_sessions: 0`, this availability purge is the only
  embargo. Positive `N` additionally excludes the current and preceding `N`
  exchange sessions: `candidate_session_index < query_session_index - N`.
- Exact Euclidean or cosine kNN applies the configured per-session cap; fewer than
  `k` neighbors are retained when the legal pool is small, never padded or reused.
  Cosine distance treats two zero vectors as equal and one zero vector as orthogonal.
- All six methods and requested embargo variants use identical scored timestamps.
  Each block is fitted once; sensitivity runs reuse its preprocessing and HMM.
  An unusable HMM fit excludes the whole block, including raw controls, and is
  recorded rather than replaced with an untrained or future-informed model.
  Chronological folds use planned evaluation dates, independent of fit success.
  Unconditional and
  same-time-of-day baselines use the eligible context universe, with the latter
  additionally matching exact session-minute offsets. Momentum extrapolates the
  recent context log return over the horizon without fitting coefficients.
  Its hard-sign 0/0.5/1 direction score is not a calibrated probability, and it
  has no invented empirical quantiles.
- The Gaussian HMM defaults to six states, full covariance, 500 iterations,
  tolerance `1e-4`, and seed 42. Retries use deterministic seed offsets `[0, 1, 2]`.
  Iteration exhaustion is **not** convergence. Numerical full-covariance failures
  can trigger a recorded diagonal-covariance fallback after seed retries;
  ordinary nonconvergence alone cannot. Failed attempts retain warnings and reasons.
- Inference implements the forward recursion `P(S_t | X_<=t)`, never smoothed
  `hmmlearn.predict_proba`. It resets at session boundaries, missing features, and
  gaps. Historical candidates and queries are filtered under one common refit.
  Posterior vectors remain on their probability scale; HMM current-state uses
  the final vector, while HMM trajectory flattens the same context length as raw kNN.
  State numbers are not aligned or pooled across refits.
- Model parameters, likelihood/convergence history, occupancy, covariance flags,
  transitions, and state means are saved for each accepted refit. Minute-of-session,
  raw-volatility-bucket, and return-sign occupancy controls use training rows only.
  Low occupancy, concentration, and near-singular states are flagged descriptively,
  not relabelled as bullish/bearish or excluded based on OOS results.
- The default sensitivity set is `sensitivity.embargo_sessions: [0, 1, 5]`;
  `retrieval.embargo_sessions` selects the primary comparison. The common cohort
  must be eligible under every requested embargo, so early observations can be
  excluded when a stronger embargo lacks enough history.
- Bootstrap replicates resample whole query sessions with replacement, retaining
  all paired observations and their original session sizes. Defaults are 1,000
  resamples and 95% percentile intervals. Both HMM methods are compared against
  raw kNN using absolute error, squared error, and Brier loss, by aggregate and fold.
  Fewer than two sessions yields an undefined interval. Intervals are exploratory
  and pointwise, not multiplicity-adjusted; cross-session dependence remains a risk.
- The report's primary conclusion uses HMM-trajectory versus raw-kNN absolute
  error: a directional conclusion requires consistently signed aggregate intervals
  and fold differences at all configured horizons; otherwise it is inconclusive.
  `data.synthetic: true` overrides this with a demonstration-only conclusion.
  Missing configured folds force an inconclusive result, even when the remaining
  folds look favorable.
- Enable `trading.enabled` for the optional long/flat simulation. Median forecasts
  above the fixed `trading.threshold` enter at the **next contiguous bar open**,
  with exit at the horizon's close and no overlapping positions per method/horizon.
  Boundary execution is idealized without added latency. Per-side costs use
  `costs.bps_per_side`; net return is
  `(exit / entry) * (1 - cost) / (1 + cost) - 1`, with cost in fractional units.
  Drawdown uses bar-level marked equity; Sharpe uses whole-session net returns,
  including observed no-trade sessions. Trading never determines research success.

### Saved research artifacts

Each successful run writes `config.yaml`, `data_manifest.json`,
`data_quality.json`, `predictions.parquet`, `neighbors.parquet`, `refits.json`,
`metrics.json`, `fold_metrics.csv`, `calibration.csv`, `bootstrap.csv`,
`sensitivity_metrics.csv`, `representation_examples.parquet`, `bars.parquet`,
`state_diagnostics/<refit_id>.json`, and `report.html`.
Enabled trading additionally writes `trading_metrics.csv`, `trades.parquet`, and
`equity.parquet`. The manifest records raw SHA-256, timestamp convention, feature
and artifact schema versions, runtime package versions, date range/counts, synthetic
status, and Git commit when available. Failed runs are marked failed and retain
fit diagnostics. `uv.lock` records the environment; NumPy is bounded below 2.4 for
compatibility with the selected pandas 2.x stack.

Neighbor records contain exact historical members for **all five**
historical-distribution methods and every embargo, with context starts, model IDs,
fit cutoffs, and outcome endpoints. They are streamed to Parquet; unconditional
reference sets can make this artifact large. Momentum has no historical neighbors.
Primary metrics use only the configured primary embargo; sensitivity metrics and
bootstrap artifacts retain the embargo identity.

Saved examples select the first eligible primary-embargo query in each accepted
block and the nearest three raw/current-state/trajectory neighbors. Their standardized
features and filtered posterior trajectories use that query's frozen model.
The report reads only selected neighbor cohorts, embeds all figures, and can be
regenerated without the original source dataset or any model fitting.

### Verification

The complete implementation passed 149 regression tests, including analytical
forward-filter versus smoother checks, deterministic fitting, numerical fallback,
nonconvergence rejection, leakage cutoffs, future perturbation, paired session
bootstrap, execution/cost accounting, and incomplete-evidence conclusion guards.

A synthetic CLI smoke used 2,381 bars across 31 exchange sessions, with DST, an
early close, and one missing bar. Five weekly six-state HMM fits were accepted.
The common cohort contained 867 queries and produced 62,424 forecasts across six
methods, four horizons, three embargoes, and three folds, with 2,311,154 audited
historical references. All availability/cap/model-identity checks passed.
Sixty sampled forecast distributions reconstructed exactly; persisted Gaussian
parameters reproduced the example posteriors and raw/latent distances without HMM
refitting. The audit checked 288 bootstrap records and 3,023 optional trades.
Input hashes remained unchanged. All 15 embedded report figures loaded, and
state/retrieval/trading panels were visually checked. These synthetic results
verify software behavior, **not** predictive market skill.
