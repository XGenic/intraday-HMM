**First real-data experiment — completed 2026-09-26**

The frozen 2019 experiment is **inconclusive: no reliable forecasting advantage
for HMM-trajectory retrieval over raw-feature kNN**. Every primary 95% paired
whole-session bootstrap interval includes zero, and fold signs are mixed. This
does not establish equivalence or rule out an effect in other periods.

**Research decision: stop development of this configuration and retain a scoped
negative finding.** Observed MAE improvements are only about 0.15–0.35% at 15–60
minutes, while five-minute MAE worsens by 0.39%. Simpler controls perform better
descriptively. Further tuning or the proposed seven-year run is not justified by
this evidence. The pipeline is preserved as a reproducible research reference.

The dataset contains 252 SPY sessions from 2019. The first eligible weekly refit
used 128 training sessions, satisfying the frozen 126-session minimum. Evaluation
covers July 8–December 31: **124 OOS sessions, 5,255 common query timestamps,
378,360 forecasts**, six methods, four horizons, and embargoes 0/1/5. All model,
retrieval, bootstrap, and trading settings stayed frozen; trading remained disabled.
The period was recorded in [experiment-plan.md](experiment-plan.md) before results.

Errors below are basis points of return. Differences are HMM trajectory minus
raw kNN; negative favors the HMM. Intervals use 1,000 whole-session resamples and
the primary one-session embargo.

| Horizon | Raw kNN MAE | HMM trajectory MAE | Difference | 95% interval |
|---|---:|---:|---:|---:|
| 5 min | 3.5095 | 3.5232 | +0.0138 | [-0.0140, +0.0416] |
| 15 min | 6.0490 | 6.0376 | -0.0114 | [-0.0895, +0.0585] |
| 30 min | 8.3610 | 8.3487 | -0.0123 | [-0.1491, +0.1227] |
| 60 min | 12.2056 | 12.1632 | -0.0424 | [-0.3681, +0.2360] |

The 5-minute difference favors raw kNN in all three folds; the other horizons
change sign across folds. Aggregate squared-error and Brier differences also have
intervals spanning zero at every horizon. MAE intervals still span zero with
embargoes 0 and 5. These are exploratory pointwise intervals without correction
for multiple comparisons or dependence between different sessions.

The unconditional and same-time-of-day controls have lower observed MAE than both
raw kNN and HMM trajectory at all four horizons. Unconditional MAE is
3.4832 / 5.9472 / 8.2499 / 11.8505 bp for 5/15/30/60 minutes. These control rankings
are descriptive; the primary paired test compares HMM trajectory with raw kNN.
There is no basis here for a trading or profitability claim.

HMM-trajectory Brier scores are 0.2551 / 0.2537 / 0.2540 / 0.2584, above the 0.25
score of a constant 50% forecast. Its nominal 80% return intervals cover about
78–79% of outcomes, closer to nominal than raw kNN's 75–78%, but they are also
wider. This descriptive calibration hint does not establish better overall
distribution forecasts or replace the primary criterion after observing results.

All **26 weekly fits were accepted with full covariance**. Twenty-five accepted
seed 42; 2019-W52 exhausted 500 iterations on seed 42 and then converged on the
predeclared retry seed 43 in 237 iterations. No entire block was dropped, no
covariance fallback occurred, and no configured degeneracy or clock-time
concentration flags were raised. Lack of a clock flag is not proof of clock-time
independence. The three folds contain 42, 41, and 41 evaluated sessions.

The saved-artifact audit passed for **all 131,310,766 historical references**,
including availability, embargo, model identities, source hashes, and outcomes
recomputed from canonical bars. It verified unique kNN cohorts with at most three
neighbors per historical session, reconstructed 180 forecast distributions, and
reproduced 360 representation rows plus neighbor distances using saved parameters
from the first, middle, and last refits. Extending the pilot data through year-end
left all **30,960 earlier forecast rows exactly unchanged**, apart from the expected
change in fold assignment. All **153 tests**, Ruff lint, and formatting checks pass.

The run took **1,321.5 seconds (22.0 minutes)** and produced **2.85 GB** before the
additional audit/figure outputs. The full HTML report contains 35 embedded figures.
Full run artifacts are retained locally and are not distributed with this repository.

The original 2019–2025 archive contained 136,727 regular-session bars
across all 1,760 scheduled sessions, with 13 missing bars left unfilled. This run
uses only 2019; it does not establish performance over 2020–2025. The snapshot
contains retrospective vendor history, not a point-in-time tape of later revisions.
The full archive would generate approximately 8.62 billion reference rows; pilot
extrapolation suggested about 178 GB and 21 hours. That proposed expansion was
not run. Resource cost was secondary to the weak evidence in the decision to stop.
Any future replication must be identified as a separate experiment, with its
criteria fixed before inspecting new outcomes.

- [Published aggregate evidence](docs/results/2019/README.md)
- [Primary confidence-interval figure](docs/results/2019/hmm-vs-raw-mae.png)
- [Audit summary](docs/results/2019/audit.json)
- [Bootstrap comparisons](docs/results/2019/bootstrap.csv)
- [Configuration](configs/alpaca_2019.yaml)

The original local run retains source snapshots, vendor provenance, predictions,
references, and refit diagnostics. Those artifacts and market inputs are excluded
from the public source distribution. Reproduction commands and the source-data
checksum are in [reproduction details](docs/reproduction.md). Published aggregate
tables alone are insufficient to repeat the individual-forecast audit.
