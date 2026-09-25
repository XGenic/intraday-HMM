# Working notes

Last updated: 2026-09-25.

## Project and current state

This is a research pipeline testing whether HMM latent-state trajectories improve
intraday historical-analogue forecasting over normalized raw-feature kNN. It is
not a live trading system. A negative or inconclusive result is a valid outcome.

**All five implementation milestones are complete.** The pipeline includes strict
local Parquet ingestion, causal features, six-method walk-forward evaluation,
leakage regressions, embargo sensitivity, session-bootstrap confidence intervals,
state diagnostics, auditable artifacts, and a self-contained HTML report. An
optional cost-aware long/flat simulation is implemented and disabled by default.

**No real market dataset has been supplied or evaluated.** Verification so far
uses hand-constructed fixtures and synthetic market bars. Do not interpret those
results as evidence of forecasting skill or profitability.

The detailed research specification is
`Intraday_Latent_Regime_Retrieval_Codex_Handoff.md`. Its final
“Implementation notes — Milestones 1–5” section records the implemented contracts
and deliberate choices. Read that section before changing methodological behavior.

Repository: `https://github.com/XGenic/intraday-HMM` (private, branch `main`).

## Next-session priorities

1. Obtain a properly licensed, timezone-aware SPY 5-minute OHLCV Parquet dataset.
   Use at least 126 training sessions plus a meaningful out-of-sample period;
   preferably use enough data for a year of training and several market regimes.
2. Confirm whether vendor timestamps label bar starts or ends, and whether an
   optional vendor VWAP is per-bar. Set the configuration explicitly; do not guess.
3. Validate the real dataset, inspect missing sessions/bars and coverage, then run
   the frozen baseline configuration. Do not tune against final OOS results.
4. Inspect refit failures/degeneracy, clock-time state concentration, neighbor
   audit examples, fold stability, and HMM-trajectory versus raw-kNN intervals.
5. Record the actual empirical conclusion, including a negative or inconclusive
   result. Additional representations or execution work should wait until this
   real-data comparison is trustworthy.

There is no unfinished HMM/report scaffold and no known failing test at handoff.
The missing prerequisite for an empirical market conclusion is real input data.

## Environment and commands

Python >=3.11; development and smoke verification used Python 3.12.3. Dependency
versions are locked in `uv.lock`. NumPy is bounded below 2.4 for compatibility with
the selected pandas 2.x stack. Do not casually upgrade numerical dependencies
without rerunning deterministic-fit and leakage checks.

```bash
uv sync --locked --extra dev
mkdir -p data/raw data/processed
# Supply your own dataset here, or change data.path in the configuration:
# data/raw/SPY_5m.parquet
uv run regime-retrieval validate-data --config configs/baseline.yaml
uv run regime-retrieval run --config configs/baseline.yaml
uv run regime-retrieval report --run runs/<run_id>

uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

`python -m regime_retrieval.cli` is the equivalent module entrypoint. YAML paths
resolve relative to the YAML file, not the shell's working directory. Saved run
configs contain resolved paths; adjust them if moving the raw data to another
machine. Missing input fails explicitly; there is no synthetic-data substitution.

## Frozen defaults and scientific invariants

- SPY, 5-minute XNYS regular-session bars; input timestamps default to **end**.
  Set `data.timestamp_label: start` for start-stamped vendor bars. Canonical
  internal timestamps are timezone-aware UTC bar-end availability times.
- Exchange calendars handle holidays, DST, and early closes. Unsorted/duplicate
  timestamps, off-grid rows, malformed OHLCV, and insufficient observed-session
  coverage fail. Default minimum coverage is 80%, including partial edge days.
  Missing bars and whole sessions are reported, never filled; calendar session
  ordinals include missing whole sessions.
- Five core features: log return, 12-bar realized volatility, range fraction,
  training-slot-adjusted volume surprise, and session VWAP distance. Vendor VWAP
  means **per-bar** VWAP and is cumulatively volume-weighted. Otherwise typical
  price is used. Rolling returns/volatility reset at gaps and session boundaries.
- Context length 12; horizons 5/15/30/60 minutes. Contexts and outcomes cannot
  bridge missing bars or sessions. Feature warmups plus complete contexts exclude
  early-session queries; maximum-horizon availability excludes late-session ones.
- All seasonality and scaling statistics are fitted before the refit block and
  frozen. Volume uses per-minute-of-session median/scaled MAD; unseen slots stay
  missing. No full-dataset preprocessing or forward-filled market bars.
- Weekly expanding-history refits after 126 training sessions. HMM defaults:
  six Gaussian states, full covariance, 500 iterations, tolerance 1e-4, seed 42.
  Retry offsets are [0, 1, 2]. Iteration exhaustion is not convergence. Numerical
  full-covariance failures may trigger a documented diagonal fallback; ordinary
  nonconvergence alone does not. Failed fits skip the entire paired block.
- **Use causal forward filtering, not smoothed `hmmlearn.predict_proba`.** Each
  posterior is P(S_t | X_<=t). Filtering resets at gaps, missing features, and
  sessions. Include preceding contiguous observations when filtering a query.
- Historical and query representations use the **same fitted HMM** per block.
  Posterior vectors stay on their probability scale. State IDs are arbitrary and
  must never be compared or averaged across independently fitted models.
- Exact Euclidean kNN by default; cosine is supported. Request 50 neighbors,
  cap at three per historical session, and retain fewer if the legal pool is
  small. Do not pad or reuse neighbors.
- A candidate's complete maximum-horizon outcome must already be known at the
  query. With embargo 0, this availability purge is the only embargo. Positive N
  additionally excludes the current session and the preceding N exchange
  sessions: `candidate_session_index < query_session_index - N`.
- Primary embargo is 1; sensitivities are 0/1/5. All six methods and all requested
  embargo variants use the same scored timestamps and refit models. Stronger
  embargoes can reduce the common query cohort. Fold boundaries depend on planned
  evaluation dates, not later fit success.
- Methods: `unconditional`, `same_time_of_day`, `momentum`, `raw_knn`,
  `hmm_current`, `hmm_trajectory`. The main comparison is HMM trajectory versus
  raw kNN. Momentum is fixed log-return extrapolation with a hard-sign direction
  score, not a calibrated probabilistic model.
- Primary point forecast is the empirical median. Bootstrap compares paired
  absolute-error, squared-error, and Brier differences using whole query-session
  resampling: 1,000 resamples, 95% percentile intervals. Fold 0 means aggregate.
  Fewer than two sessions gives an undefined interval. Intervals are pointwise,
  exploratory, and not adjusted for multiple testing or cross-session dependence.
- Report conclusions require consistent aggregate intervals and fold signs at
  every configured horizon. Missing configured folds force “inconclusive”; this
  has a regression test. `data.synthetic: true` forces demonstration-only output.
- Optional trading: `trading.enabled: true`; fixed median threshold, long/flat,
  next contiguous bar open, exit at horizon close, no overlapping positions per
  method/horizon, no overnight trades. Default cost is 1 bp per side. Execution
  at the query-close/next-open boundary is idealized with no added latency.
  Net holding multiplier is `(exit / entry) * (1 - cost) / (1 + cost)`.
  Drawdown uses marked equity; Sharpe uses whole-session returns, not bar/trade
  pseudo-independent samples. Trading never determines research success.

## Code map

- `config.py`, `configs/baseline.yaml`: validated configuration and defaults.
- `cli.py`: validation, research orchestration, artifact writing, report command.
- `data/`: local provider, session grid, validation and quality reports.
- `features/`: causal calculations, slot seasonality, training-only scaling.
- `models/hmm.py`: segmented HMM fit/retries and causal log-space filtering.
- `models/diagnostics.py`: training-only state/clock-time/volatility/sign summaries.
- `retrieval/`: contiguous contexts, availability index, exact constrained kNN.
- `forecasting/`: forward labels, empirical distributions, simple controls.
- `evaluation/walk_forward.py`: common-model/common-cohort six-method evaluator.
- `evaluation/bootstrap.py`, `sensitivity.py`: paired session intervals and embargo tables.
- `evaluation/trading.py`: optional execution and portfolio accounting.
- `reporting/artifacts.py`: streaming neighbor Parquet writer and safe JSON output.
- `reporting/report.py`: standalone HTML generation and evidence-sufficiency guards.
- `tests/`: feature/calendar/eligibility tests plus HMM, future-perturbation,
  bootstrap, sensitivity, trading, and report-conclusion regressions.

Package paths above are relative to `src/regime_retrieval/` unless stated otherwise.
No Python language server was configured during implementation; Ruff and pytest
were used for final static/style and behavioral verification.

## Artifacts, storage, and reproducibility

A successful run writes resolved config, raw-file SHA-256/provenance, data-quality
report, canonical `bars.parquet`, all-embargo `predictions.parquet` and
`neighbors.parquet`, refit records, per-refit `state_diagnostics/*.json`, primary
metrics/folds/calibration, `bootstrap.csv`, `sensitivity_metrics.csv`, saved
`representation_examples.parquet`, and `report.html`. Enabled trading adds
`trading_metrics.csv`, `trades.parquet`, and `equity.parquet`.

Neighbor records retain exact historical members for all five historical methods,
all outcome endpoints, model identities, fit cutoffs, ranks, and distances. They
are streamed, but unconditional cohorts can make the file large. Reports read
selected cohorts and embed figures; regeneration needs saved run artifacts, not
original vendor data or model refitting. Model parameters are JSON, not pickles.

Raw/processed market data, generated runs, virtual environments, and caches are
ignored by Git. They are **not uploaded to GitHub**. Empty data directories are
also absent from a fresh clone. Keep licensed input data and research outputs in
appropriate separate storage.

## Last completed verification

- **149 tests passed**; Ruff lint and formatting checks passed.
- Full synthetic CLI smoke: 2,381 bars across 31 exchange sessions, including DST,
  an early close, and a missing bar. Five weekly six-state HMM fits accepted.
- 867 common queries, six methods, four horizons, three embargoes, three folds:
  **62,424 forecasts** and **2,311,154 audited historical references**.
- All cutoff, session-cap, cohort, and model-identity checks passed. Sixty sampled
  forecast distributions reconstructed exactly from saved references.
- Persisted Gaussian parameters reproduced saved example posteriors and raw/latent
  distances without refitting an HMM. Checked 288 bootstrap records and 3,023
  optional trades; source-file hashes remained unchanged.
- All 15 embedded figures loaded. Report header, state diagnostics, retrieval
  trajectories, and gross/net trading panels were visually checked in Chromium.
- A missing-fold conclusion bug was reproduced and fixed: favorable evidence
  from only a subset of configured folds cannot claim directional improvement.

Latest local full report:
`runs/full-synthetic-smoke/20260925T043153-a0dbe0b9/report.html`

Its input/config live under `runs/full-synthetic-smoke/`. This smoke overrides
minimum training history to 10 sessions, bootstrap repetitions to 200, and enables
trading; the shipped baseline retains 126/1,000/trading-disabled defaults.

Earlier Milestone-2-only output remains under `runs/synthetic-smoke/`; it is not
current-format full-pipeline evidence. Temporary smoke-generation/audit scripts
were removed after verification. None of these ignored local artifacts accompany
a fresh clone. There is no committed synthetic-data generator.
