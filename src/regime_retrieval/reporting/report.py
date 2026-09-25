"""Render a portable six-method research report from saved run artifacts only."""

import base64
import json
from io import BytesIO
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import yaml
from jinja2 import Environment
from matplotlib.figure import Figure

from regime_retrieval.evaluation.metrics import LOG_LOSS_EPSILON

matplotlib.use("Agg")

_RETRIEVAL_METHODS = ("raw_knn", "hmm_current", "hmm_trajectory")

_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Latent regime retrieval: six-method V1 research report</title>
<style>
body {font:15px/1.5 system-ui,sans-serif;color:#17212b;margin:2rem auto;max-width:1250px;
 padding:0 1rem;background:#fff} h1,h2,h3 {line-height:1.2} .warning {background:#fff0c2;
 border-left:5px solid #9b6600;padding:1rem} .synthetic {background:#ffe0e0;border:3px solid
 #a51d1d;padding:1rem;font-size:1.4rem;font-weight:bold} table {border-collapse:collapse;
 font-size:12px;width:100%} th,td {border:1px solid #ccd2d8;padding:6px;text-align:right;
 white-space:nowrap} th {background:#eaf0f5} .scroll {overflow:auto;margin:1rem 0}
pre {white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f4f6;padding:1rem}
img {max-width:100%;height:auto} details {margin:1rem 0} .muted {color:#52606d}
</style></head><body>
{% macro table(data) -%}
{% if data.rows %}<div class="scroll"><table><thead><tr>
{% for column in data.columns %}<th>{{ column }}</th>{% endfor %}</tr></thead><tbody>
{% for row in data.rows %}<tr>{% for value in row %}<td>{{ value|display_value }}</td>
{% endfor %}</tr>{% endfor %}</tbody></table></div>
{% else %}<p>No rows available.</p>{% endif %}
{%- endmacro %}
<h1>Latent regime retrieval: six-method V1 comparison</h1>
{% if synthetic %}<div class="synthetic">SYNTHETIC DATA — pipeline demonstration only.
These results are not evidence of market predictability or trading skill.</div>{% endif %}
<p>Run: <strong>{{ run_id }}</strong>. Symbol: <strong>{{ symbol }}</strong>.
Forecast query range: {{ start }} to {{ end }}. {{ session_count }} evaluated sessions;
{{ query_count }} distinct query timestamps. Primary embargo: {{ primary_embargo }} sessions.</p>
<h2>Primary conclusion</h2>
<div class="warning" data-outcome="{{ conclusion.outcome }}">
<strong>{{ conclusion.label }}</strong><p>{{ conclusion.reason }}</p></div>
<p>The primary comparison is <strong>hmm_trajectory minus raw_knn</strong> in absolute error;
negative differences favor latent retrieval. An improved conclusion requires aggregate interval
upper bounds below zero at every configured horizon and negative differences in every
chronological fold. Every configured fold must be represented. A not-improved conclusion
requires consistently adverse intervals and fold signs; incomplete evidence is inconclusive.
The hmm_current comparison is a descriptive current-state ablation, not a second primary claim.
Forecast quality, not trading P&amp;L, determines the conclusion.</p>
<div class="warning"><strong>Exploratory, pointwise uncertainty.</strong> Paired bootstrap
replicates resample whole sessions and retain all paired intraday forecasts in each sampled
session. This preserves within-session dependence, but sessions themselves need not be independent;
serial dependence across sessions and overlapping history can make these intervals optimistic.
Intervals are pointwise, not simultaneous; there is no multiple-testing adjustment across horizons,
losses, folds, methods or embargoes. Favorable intervals do not prove market skill.</div>
<h2>Methods and common comparison cohort</h2>
<p>All six methods and all embargo variants share common evaluable query timestamps.
Unconditional uses all eligible historical contexts; same_time_of_day restricts candidates to
the query's session-minute offset. Raw kNN compares standardized feature windows; hmm_current
compares the current filtered posterior vector, while hmm_trajectory compares the recent sequence
of filtered posterior vectors. Historical and query posteriors share one frozen refit-block model.
No smoothed posteriors or state-label alignment across independent refits are used.
The current-state ablation isolates the contribution of posterior trajectories.</p>
<p><strong>Momentum is a point-only control:</strong> recent context log return is extrapolated
over the horizon. Its direction probability is the hard sign 0, 1, or 0.5, not an estimated
probability. Other methods use equal-weight empirical outcomes. Their 10–90% return bands are
historical outcome quantiles, not confidence intervals for expected returns. Point errors use
the median forecast and simple forward returns. Up means strictly positive realized return.
Distances have method-specific representations and are not comparable across methods.</p>
<p>With positive embargo N, both the current and preceding N exchange sessions are excluded.
With N = 0, a historical candidate is permitted only if its maximum-horizon outcome is already
known at the query. Contexts and outcomes remain contiguous within a session. Preprocessing and
HMM parameters are fitted before each block, then frozen. A failed HMM fit excludes the entire
block for every method, preserving paired comparisons; failures remain in the diagnostic audit.</p>
<h2>Configuration and data provenance</h2>
<details><summary>Saved configuration</summary><pre>{{ config_text }}</pre></details>
<details><summary>Data manifest (including source identity and Git commit)</summary>
<pre>{{ manifest_text }}</pre></details>
<h2>Data quality, session coverage, and missing bars</h2>
<p>Saved source diagnostics describe acceptance, gaps and session coverage. The table describes
evaluated queries only, excluding initial training sessions and incomplete contexts/outcomes.</p>
<pre>{{ quality_text }}</pre>{{ table(session_coverage) }}
<h2>Aggregate forecast comparison — primary embargo</h2>
<p>Return errors and widths are fractional returns (0.01 = 1%). Signed error is forecast minus
realized return. Brier uses unclipped probabilities; log loss clips only for the logarithm at
ε = {{ epsilon }}. Spearman is undefined for constant series. An em dash denotes an undefined
value. Coverage includes both endpoints. Neighbor dispersion is within-cohort outcome variation,
not uncertainty in the reported loss.</p>
<img alt="Mean absolute error by horizon for all six methods"
 src="data:image/png;base64,{{ error_plot }}">
{{ table(aggregate) }}
<h2>Primary paired whole-session bootstrap intervals and fold stability</h2>
<p>Every difference pairs identical query timestamps and horizons. Fold 0 is the aggregate;
positive fold IDs are chronological subsets, not independent replicates or additional training
splits. Fold boundaries depend on planned evaluation dates, not subsequent fit success.
All differences are model minus raw_knn. Status insufficient_sessions means an interval
cannot be estimated, not evidence of equivalence.</p>
{{ table(primary_bootstrap) }}
<h3>Descriptive current-state ablation and other loss intervals</h3>
{{ table(secondary_bootstrap) }}
<h3>All six methods by chronological fold</h3>{{ table(folds) }}
<h2>Embargo sensitivity on the common query cohort</h2>
<p>Requested embargoes: {{ embargoes }}. The standard 0/1/5-session comparison tests whether
results rely on temporally adjacent analogues. Zero still purges unknown maximum-horizon
outcomes; positive values exclude whole preceding exchange sessions. These are exploratory
sensitivities, not opportunities to select the best-looking setting. Fold 0 denotes aggregate.</p>
<img alt="Absolute error across embargo settings by horizon and method"
 src="data:image/png;base64,{{ sensitivity_plot }}">
{{ table(sensitivity) }}
<details><summary>Paired bootstrap intervals for every embargo and fold</summary>
{{ table(all_bootstrap) }}</details>
<h2>Probability calibration</h2>
<p>Ten bins are left-closed and right-open, except the final bin includes 1. Empty bins remain
in the table but are omitted from plots. Momentum's sign is not a fitted probability model.</p>
<img alt="Observed up frequency versus predicted up probability"
 src="data:image/png;base64,{{ calibration_plot }}">
{{ table(calibration) }}
<h2>Refit cutoffs, failures and state diagnostics</h2>
<p>Fit cutoffs precede block starts; candidate outcomes are separately purged at each query.
All records below are saved artifacts, not refitted models. State indices are local coordinates
of one refit: <strong>never compare or average state number across refits</strong>. No narrative
bull/bear names are assigned. Occupancy and its clock-time, volatility and return-sign controls
are training diagnostics, not out-of-sample predictive evidence. Compare the same_time_of_day
baseline above before attributing clock-time structure to added forecasting information.</p>
{{ table(refits) }}{{ table(skipped) }}
{% for diagnostic in diagnostics %}
<details><summary>Refit {{ diagnostic.refit_id }} — {{ diagnostic.status }}</summary>
<pre>{{ diagnostic.summary }}</pre>
<h3>Fit attempts, retry and fallback details</h3>{{ table(diagnostic.attempts) }}
{% if diagnostic.accepted %}
<p><strong>Saved diagnostic flags:</strong> {{ diagnostic.flags }}</p>
<p>{{ diagnostic.clock_note }}</p>
<img alt="Per-refit transition matrix and state occupancy controls"
 src="data:image/png;base64,{{ diagnostic.image }}">
<h3>State occupancy and covariance summaries</h3>{{ table(diagnostic.states) }}
<h3>Gaussian state means in fitted feature coordinates</h3>{{ table(diagnostic.means) }}
<h3>Empirical mean feature vector by state</h3>{{ table(diagnostic.empirical_means) }}
<h3>Transition probabilities (row = source, column = destination)</h3>
{{ table(diagnostic.transitions) }}
<details><summary>Full covariance matrices and conditional occupancy rows</summary>
<pre>{{ diagnostic.covariances }}</pre>
{{ table(diagnostic.minute) }}{{ table(diagnostic.volatility) }}{{ table(diagnostic.sign) }}
</details>{% endif %}</details>
{% else %}<p>No state diagnostic records were saved.</p>{% endfor %}
<h2>Audited raw and latent retrieval examples</h2>
<p>Examples are selected deterministically across saved first-evaluable-query-per-refit contexts,
not by forecast success. Only those queries' primary-embargo retrieval cohorts are filtered-read
from neighbors.parquet. Timestamp is the neighbor ID; complete saved cohorts retain outcome
endpoints and model IDs. Up to twelve neighbors per method are displayed. All horizons share a
cohort. Outcome histograms use the complete selected-query cohorts. Realized outcomes are
shown for evaluation, never supplied as forecast inputs.</p>
<p>Trajectory panels compare the query (solid) with the nearest neighbor of each method
(dashed). Price is close / first context close − 1, read from saved canonical bars. Features use
the saved query-refit's standardization. Posterior curves are saved causal filtered probabilities
under that same model for both historical and query contexts; they are not reconstructed or
smoothed. State colors have meaning only within that example's refit.</p>
{% for example in examples %}
<h3>Query {{ example.timestamp }} — refit {{ example.refit_id }}</h3>
{{ table(example.predictions) }}
<img alt="Query and nearest-neighbor price, feature and posterior trajectories"
 src="data:image/png;base64,{{ example.trajectory_plot }}">
<img alt="Historical neighbor return distributions and actual query outcomes"
 src="data:image/png;base64,{{ example.return_plot }}">
{% for cohort in example.cohorts %}
<h4>{{ cohort.method }}: {{ cohort.count }} audited neighbors; {{ cohort.shown }} displayed</h4>
{{ table(cohort.neighbors) }}{% endfor %}
{% else %}<p>No representation examples were saved.</p>{% endfor %}
<h2>Secondary cost-aware trading simulation</h2>
{% if trading.enabled %}
<p>Trading is enabled for the primary embargo only. Median forecasts above {{ trading.threshold }}
go long; all other signals stay flat. There are no shorts or leverage. Signals use query-close
information; entry is at the next contiguous bar open, not the signal close. This is idealized
boundary execution without additional latency: the query bar-end and next open timestamps
coincide. Exit is at the forecast outcome-end close. No overlapping positions are allowed within
each method/horizon stream, and trades never cross sessions or gaps. A signal at the prior
exit boundary may enter the next trade. Costs are {{ trading.cost }} basis points per side.
The net holding multiplier is (exit / entry) × (1 − cost) / (1 + cost), with cost in fractional
units. Gross gains alone do not establish economic value.</p>
<p>Each method/horizon account starts at one. Equity is marked every observed bar, including
entry fees at the execution boundary. Net daily Sharpe uses all observed sessions in the query
span (including no-trade sessions), sample standard deviation and √252 annualization; missing
whole sessions are not fabricated. Drawdown is a positive fraction using net marks and initial
capital. Hit rate and average holding return use net trades. Turnover is two-way net-account
notional divided by initial capital; time in market is held bars / observed bars. Undefined risk
statistics are not zero. Trading does not change the forecast-quality conclusion.</p>
{{ table(trading.metrics) }}
{% if trading.image %}
<img alt="Secondary gross and net equity by method and horizon"
 src="data:image/png;base64,{{ trading.image }}">{% endif %}
<details><summary>First {{ trading.shown }} of {{ trading.count }} saved trades</summary>
{{ table(trading.trades) }}</details>
{% else %}<p>Disabled (the default). No strategy return or economic-value claim is made.
Enabling the secondary simulation requires next-open execution, no overlap and configured
entry/exit costs; the forecast comparison above is independent of that choice.</p>{% endif %}
<p class="muted">Rendered solely from standalone saved run artifacts. All figures are embedded;
no network resources, source dataset access, model refits or graphical display are required.</p>
</body></html>"""


def _table(frame: pd.DataFrame) -> dict:
    return {
        "columns": frame.columns.tolist(),
        "rows": list(frame.itertuples(index=False, name=None)),
    }


def _display_value(value: object) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    return str(value)


def _image(figure: Figure) -> str:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    figure.clear()
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _error_plot(aggregate: pd.DataFrame) -> str:
    figure = Figure(figsize=(10, 4), layout="constrained")
    axes = figure.subplots()
    for method, group in aggregate.groupby("method", sort=True):
        group = group.sort_values("horizon_minutes")
        axes.plot(group["horizon_minutes"], group["mae"], marker="o", label=method)
    axes.set(xlabel="Horizon (minutes)", ylabel="Mean absolute error (fractional return)")
    axes.legend()
    axes.grid(alpha=0.2)
    return _image(figure)


def _calibration_plot(calibration: pd.DataFrame) -> str:
    horizons = sorted(calibration["horizon_minutes"].unique())
    columns = min(2, len(horizons))
    rows = (len(horizons) + columns - 1) // columns
    figure = Figure(figsize=(6 * columns, 4 * rows), layout="constrained")
    axes = figure.subplots(rows, columns, squeeze=False).ravel()
    for axis, horizon in zip(axes, horizons, strict=False):
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
        current = calibration.loc[calibration["horizon_minutes"].eq(horizon)]
        for method, group in current.groupby("method", sort=True):
            group = group.loc[group["n_forecasts"].gt(0)].sort_values("bin")
            axis.plot(
                group["mean_probability_up"],
                group["observed_up_frequency"],
                marker="o",
                label=method,
            )
        axis.set(
            title=f"{horizon} minute horizon",
            xlabel="Mean predicted P(up)",
            ylabel="Observed up frequency",
            xlim=(-0.02, 1.02),
            ylim=(-0.02, 1.02),
        )
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    for axis in axes[len(horizons) :]:
        axis.set_visible(False)
    return _image(figure)


def _conclusion(primary: pd.DataFrame, horizons: list[int], synthetic: bool, n_folds: int) -> dict:
    if synthetic:
        return {
            "outcome": "synthetic",
            "label": "Demonstration only — no market-skill conclusion",
            "reason": (
                "Synthetic data verifies the pipeline, not market predictability, "
                "regardless of favorable intervals or fold signs."
            ),
        }
    expected = {(int(fold), int(horizon)) for fold in range(n_folds + 1) for horizon in horizons}
    observed = set(zip(primary["fold"], primary["horizon_minutes"], strict=True))
    aggregate = primary.loc[primary["fold"].eq(0)]
    folds = primary.loc[primary["fold"].ne(0)]
    complete = (
        observed == expected
        and len(primary) == len(expected)
        and aggregate["status"].eq("ok").all()
        and np.isfinite(aggregate[["mean_difference", "ci_lower", "ci_upper"]].to_numpy()).all()
        and np.isfinite(folds["mean_difference"].to_numpy()).all()
    )
    if complete and aggregate["ci_upper"].lt(0).all() and folds["mean_difference"].lt(0).all():
        return {
            "outcome": "improved",
            "label": "Latent retrieval improved out-of-sample forecasting in this run",
            "reason": (
                "Every configured horizon has a favorable aggregate absolute-error interval "
                "and favorable fold differences. This is exploratory evidence for this run, "
                "not a guarantee of generalization."
            ),
        }
    if complete and aggregate["ci_lower"].gt(0).all() and folds["mean_difference"].gt(0).all():
        return {
            "outcome": "not_improved",
            "label": "Latent retrieval did not improve out-of-sample forecasting in this run",
            "reason": (
                "Every configured horizon has an aggregate absolute-error interval favoring "
                "raw kNN, with adverse fold differences for latent retrieval."
            ),
        }
    return {
        "outcome": "inconclusive",
        "label": "Evidence is unstable / inconclusive",
        "reason": (
            "The configured horizons do not consistently favor one representation. "
            "Intervals crossing zero, mixed fold signs, or insufficient sessions "
            "prevent a directional conclusion."
        ),
    }


def _sensitivity_plot(sensitivity: pd.DataFrame) -> str:
    aggregate = sensitivity.loc[sensitivity["fold"].eq(0)]
    horizons = sorted(aggregate["horizon_minutes"].unique())
    figure = Figure(
        figsize=(6 * min(2, len(horizons)), 4 * ((len(horizons) + 1) // 2)), layout="constrained"
    )
    axes = figure.subplots((len(horizons) + 1) // 2, min(2, len(horizons)), squeeze=False).ravel()
    for axis, horizon in zip(axes, horizons, strict=False):
        for method, group in aggregate.loc[aggregate["horizon_minutes"].eq(horizon)].groupby(
            "method", sort=True
        ):
            group = group.sort_values("embargo_sessions")
            axis.plot(group["embargo_sessions"], group["mae"], marker="o", label=method)
        axis.set(
            title=f"{horizon} minute horizon",
            xlabel="Embargo (exchange sessions)",
            ylabel="Mean absolute error",
        )
        axis.set_xticks(sorted(aggregate["embargo_sessions"].unique()))
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    for axis in axes[len(horizons) :]:
        axis.set_visible(False)
    return _image(figure)


def _state_frame(values: list, features: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(values, columns=features)
    frame.insert(0, "state", np.arange(len(frame)))
    return frame


def _diagnostic_plot(diagnostic: dict) -> str:
    figure = Figure(figsize=(12, 8), layout="constrained")
    axes = figure.subplots(2, 2).ravel()
    matrix = np.asarray(diagnostic["transition_matrix"])
    image = axes[0].imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    axes[0].set(
        title="Transition probabilities",
        xlabel="Destination state",
        ylabel="Source state",
        xticks=np.arange(len(matrix)),
        yticks=np.arange(len(matrix)),
    )
    figure.colorbar(image, ax=axes[0])
    for axis, key, coordinate, title in zip(
        axes[1:],
        ("occupancy_by_minute", "occupancy_by_volatility", "occupancy_by_return_sign"),
        ("minute_of_session", "bucket", "sign"),
        (
            "Occupancy by minute of session",
            "Occupancy by volatility bucket",
            "Occupancy by return sign",
        ),
        strict=True,
    ):
        rows = pd.DataFrame(diagnostic[key])
        if rows.empty:
            axis.set_title(title)
            axis.text(
                0.5,
                0.5,
                "Unavailable — see saved diagnostic flags",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            continue
        for state, group in rows.groupby("state", sort=True):
            group = group.sort_values(coordinate)
            axis.plot(group[coordinate], group["probability"], marker=".", label=f"State {state}")
        axis.set(title=title, xlabel=coordinate, ylabel="Mean filtered probability", ylim=(0, 1))
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    return _image(figure)


def _clock_note(minute: pd.DataFrame) -> str:
    if minute.empty:
        return "Clock-time occupancy is unavailable; concentration cannot be assessed."
    notes = []
    for state, rows in minute.groupby("state", sort=True):
        dominant = rows.loc[rows["probability"].ge(0.8)]
        if not dominant.empty and rows["probability"].max() - rows["probability"].min() >= 0.5:
            offsets = ", ".join(
                str(value) for value in dominant["minute_of_session"].sort_values().tolist()
            )
            notes.append(
                f"State {state}: mean posterior ≥0.8 at offsets [{offsets}], "
                "with within-day occupancy range ≥0.5."
            )
    if notes:
        return (
            "Descriptive clock-time concentration flag: "
            + " ".join(notes)
            + " Descriptive thresholds, not a significance test or semantic state label."
        )
    return (
        "No state meets both descriptive thresholds: mean posterior ≥0.8 and within-day "
        "range ≥0.5. This does not establish clock-time independence."
    )


def _diagnostics(run_dir: Path) -> tuple[list[dict], dict[str, list[str]]]:
    rendered = []
    feature_columns = {}
    for path in sorted((run_dir / "state_diagnostics").glob("*.json")):
        diagnostic = json.loads(path.read_text(encoding="utf-8"))
        accepted = diagnostic["status"] == "accepted"
        summary_keys = (
            "refit_id",
            "block_start",
            "block_end",
            "status",
            "converged",
            "log_likelihood",
            "iterations",
            "covariance_type",
            "selected_seed",
            "fallback_reason",
            "reason",
            "fit_end",
        )
        item = {
            "refit_id": diagnostic["refit_id"],
            "status": diagnostic["status"],
            "accepted": accepted,
            "summary": json.dumps(
                {key: diagnostic[key] for key in summary_keys if key in diagnostic}, indent=2
            ),
            "attempts": _table(pd.DataFrame(diagnostic["attempts"])),
        }
        if accepted:
            features = diagnostic["feature_columns"]
            feature_columns[diagnostic["refit_id"]] = features
            covariance = np.asarray(diagnostic["state_covariances"], dtype=float)
            states = []
            for state, matrix in enumerate(covariance):
                eigenvalues = np.linalg.eigvalsh(matrix)
                states.append(
                    {
                        "state": state,
                        "occupancy": diagnostic["state_occupancy"][state],
                        "covariance_trace": float(np.trace(matrix)),
                        "min_eigenvalue": float(eigenvalues.min()),
                        "max_eigenvalue": float(eigenvalues.max()),
                        "condition_number": float(np.linalg.cond(matrix)),
                    }
                )
            minute = pd.DataFrame(diagnostic["occupancy_by_minute"])
            item.update(
                {
                    "flags": "; ".join(diagnostic["flags"]) or "None recorded",
                    "clock_note": _clock_note(minute),
                    "image": _diagnostic_plot(diagnostic),
                    "states": _table(pd.DataFrame(states)),
                    "means": _table(_state_frame(diagnostic["state_means"], features)),
                    "empirical_means": _table(
                        _state_frame(diagnostic["mean_features_by_state"], features)
                    ),
                    "transitions": _table(
                        _state_frame(
                            diagnostic["transition_matrix"],
                            [f"to_state_{i}" for i in range(len(covariance))],
                        )
                    ),
                    "covariances": json.dumps(diagnostic["state_covariances"], indent=2),
                    "minute": _table(minute),
                    "volatility": _table(pd.DataFrame(diagnostic["occupancy_by_volatility"])),
                    "sign": _table(pd.DataFrame(diagnostic["occupancy_by_return_sign"])),
                }
            )
        rendered.append(item)
    return rendered, feature_columns


def _trajectory_plot(contexts: pd.DataFrame, prices: pd.Series, features: list[str]) -> str:
    rows = len(features) + 2
    figure = Figure(figsize=(15, 2.4 * rows), layout="constrained")
    axes = figure.subplots(rows, len(_RETRIEVAL_METHODS), squeeze=False)
    query = contexts.loc[contexts["role"].eq("query")].sort_values("context_step")
    posterior_columns = sorted(
        (column for column in contexts if column.startswith("posterior_")),
        key=lambda column: int(column.split("_")[-1]),
    )
    for column, method in enumerate(_RETRIEVAL_METHODS):
        neighbor = contexts.loc[contexts["role"].eq(method) & contexts["rank"].eq(1)].sort_values(
            "context_step"
        )
        for frame, label, style in ((query, "Query", "-"), (neighbor, "Nearest neighbor", "--")):
            closes = prices.reindex(frame["context_timestamp"]).to_numpy(dtype=float)
            axes[0, column].plot(
                frame["context_step"], closes / closes[0] - 1, linestyle=style, label=label
            )
            for row, feature in enumerate(features, start=1):
                axes[row, column].plot(
                    frame["context_step"], frame[feature], linestyle=style, label=label
                )
            for state, posterior in enumerate(posterior_columns):
                axes[-1, column].plot(
                    frame["context_step"],
                    frame[posterior],
                    linestyle=style,
                    color=f"C{state % 10}",
                    label=f"{label}: state {state}",
                )
        axes[0, column].set_title(f"{method}\n{neighbor['neighbor_timestamp'].iloc[0]}")
        axes[0, column].set_ylabel("Normalized close")
        for row, feature in enumerate(features, start=1):
            axes[row, column].set_ylabel(feature)
        axes[-1, column].set(ylabel="Filtered posterior", ylim=(0, 1))
        for axis in axes[:, column]:
            axis.set_xlabel("Context step")
            axis.legend(fontsize=6, ncol=2 if axis is axes[-1, column] else 1)
            axis.grid(alpha=0.2)
    return _image(figure)


def _return_plot(cohort: pd.DataFrame, query: pd.DataFrame) -> str:
    horizons = sorted(query["horizon_minutes"].unique())
    columns = min(2, len(horizons))
    figure = Figure(
        figsize=(6 * columns, 3.5 * ((len(horizons) + columns - 1) // columns)),
        layout="constrained",
    )
    axes = figure.subplots((len(horizons) + columns - 1) // columns, columns, squeeze=False).ravel()
    for axis, horizon in zip(axes, horizons, strict=False):
        outcome = f"return_{horizon}m"
        bins = np.histogram_bin_edges(cohort[outcome].to_numpy(), bins=15)
        for method, rows in cohort.groupby("method", sort=True):
            axis.hist(rows[outcome], bins=bins, histtype="step", density=True, label=method)
        actual = query.loc[query["horizon_minutes"].eq(horizon), "actual_return"].iloc[0]
        axis.axvline(actual, color="black", linestyle=":", label="Actual query return")
        axis.set(
            title=f"{horizon} minute outcomes", xlabel="Forward return", ylabel="Empirical density"
        )
        axis.legend(fontsize=8)
    for axis in axes[len(horizons) :]:
        axis.set_visible(False)
    return _image(figure)


def _audit_examples(
    run_dir: Path,
    predictions: pd.DataFrame,
    primary_embargo: int,
    feature_columns: dict[str, list[str]],
) -> list[dict]:
    representations = pd.read_parquet(
        run_dir / "representation_examples.parquet",
        filters=[("embargo_sessions", "==", primary_embargo)],
    )
    timestamps = representations["query_timestamp"].drop_duplicates().sort_values().tolist()
    if not timestamps:
        return []
    dataset = ds.dataset(run_dir / "neighbors.parquet", format="parquet")
    timestamp_type = dataset.schema.field("query_timestamp").type
    bars = ds.dataset(run_dir / "bars.parquet", format="parquet")
    positions = np.unique(np.linspace(0, len(timestamps) - 1, min(3, len(timestamps)), dtype=int))
    examples = []
    for position in positions:
        timestamp = timestamps[position]
        query_filter = (
            ds.field("method").isin(_RETRIEVAL_METHODS)
            & (ds.field("embargo_sessions") == primary_embargo)
            & (
                ds.field("query_timestamp")
                == pa.scalar(timestamp.to_pydatetime(), type=timestamp_type)
            )
        )
        cohort = dataset.to_table(filter=query_filter).to_pandas().sort_values(["method", "rank"])
        query = predictions.loc[predictions["query_timestamp"].eq(timestamp)].sort_values(
            ["horizon_minutes", "method"]
        )
        contexts = representations.loc[representations["query_timestamp"].eq(timestamp)]
        refit_id = contexts["refit_id"].iloc[0]
        context_times = pa.array(
            contexts["context_timestamp"].drop_duplicates(),
            type=bars.schema.field("timestamp").type,
        )
        prices = (
            bars.to_table(
                columns=["timestamp", "close"], filter=ds.field("timestamp").isin(context_times)
            )
            .to_pandas()
            .set_index("timestamp")["close"]
        )
        examples.append(
            {
                "timestamp": timestamp.isoformat(),
                "refit_id": refit_id,
                "predictions": _table(query),
                "cohorts": [
                    {
                        "method": method,
                        "count": len(rows),
                        "shown": min(12, len(rows)),
                        "neighbors": _table(rows.iloc[:12]),
                    }
                    for method, rows in cohort.groupby("method", sort=True)
                ],
                "trajectory_plot": _trajectory_plot(contexts, prices, feature_columns[refit_id]),
                "return_plot": _return_plot(cohort, query),
            }
        )
    return examples


def _trading(run_dir: Path, config: dict) -> dict:
    if not config["trading"]["enabled"]:
        return {"enabled": False}
    metrics = pd.read_csv(run_dir / "trading_metrics.csv")
    trades = pd.read_parquet(run_dir / "trades.parquet").sort_values(
        ["query_timestamp", "method", "horizon_minutes"]
    )
    equity = pd.read_parquet(run_dir / "equity.parquet")
    image = None
    if not equity.empty:
        horizons = sorted(equity["horizon_minutes"].unique())
        columns = min(2, len(horizons))
        rows = (len(horizons) + columns - 1) // columns
        figure = Figure(figsize=(6 * columns, 4 * rows), layout="constrained")
        axes = figure.subplots(rows, columns, squeeze=False).ravel()
        for axis, horizon in zip(axes, horizons, strict=False):
            for index, (method, track) in enumerate(
                equity.loc[equity["horizon_minutes"].eq(horizon)].groupby("method", sort=True)
            ):
                track = track.sort_values("timestamp")
                axis.plot(
                    track["timestamp"],
                    track["net_equity"],
                    color=f"C{index % 10}",
                    label=f"{method} net",
                )
                axis.plot(
                    track["timestamp"],
                    track["gross_equity"],
                    linestyle="--",
                    color=f"C{index % 10}",
                    alpha=0.65,
                    label=f"{method} gross",
                )
            axis.set(title=f"{horizon} minute horizon", ylabel="Equity (initial = 1)")
            axis.tick_params(axis="x", labelrotation=25)
            axis.legend(fontsize=6, ncol=2)
            axis.grid(alpha=0.2)
        for axis in axes[len(horizons) :]:
            axis.set_visible(False)
        image = _image(figure)
    return {
        "enabled": True,
        "threshold": config["trading"]["threshold"],
        "cost": config["costs"]["bps_per_side"],
        "metrics": _table(metrics),
        "trades": _table(trades.iloc[:30]),
        "count": len(trades),
        "shown": min(30, len(trades)),
        "image": image,
    }


def render_report(run_dir: Path) -> Path:
    """Regenerate a self-contained report from the current complete run-artifact format."""
    run_dir = Path(run_dir)
    config_text = (run_dir / "config.yaml").read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)
    primary_embargo = config["retrieval"]["embargo_sessions"]
    manifest = json.loads((run_dir / "data_manifest.json").read_text(encoding="utf-8"))
    quality = json.loads((run_dir / "data_quality.json").read_text(encoding="utf-8"))
    refit_records = json.loads((run_dir / "refits.json").read_text(encoding="utf-8"))
    aggregate = pd.DataFrame(json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")))
    folds = pd.read_csv(run_dir / "fold_metrics.csv").sort_values(
        ["query_start", "fold", "horizon_minutes", "method"]
    )
    calibration = pd.read_csv(run_dir / "calibration.csv")
    predictions = pd.read_parquet(
        run_dir / "predictions.parquet", filters=[("embargo_sessions", "==", primary_embargo)]
    )
    bootstrap = pd.read_csv(run_dir / "bootstrap.csv")
    sensitivity = pd.read_csv(run_dir / "sensitivity_metrics.csv")
    primary_mask = (
        bootstrap["embargo_sessions"].eq(primary_embargo)
        & bootstrap["reference"].eq("raw_knn")
        & bootstrap["model"].eq("hmm_trajectory")
        & bootstrap["loss"].eq("absolute_error")
    )
    primary = bootstrap.loc[primary_mask].sort_values(["fold", "horizon_minutes"])
    secondary = bootstrap.loc[bootstrap["embargo_sessions"].eq(primary_embargo) & ~primary_mask]
    coverage = (
        predictions.groupby("query_session", sort=True)
        .agg(
            query_timestamps=("query_timestamp", "nunique"),
            first_query=("query_timestamp", "min"),
            last_query=("query_timestamp", "max"),
        )
        .reset_index()
    )
    diagnostics, feature_columns = _diagnostics(run_dir)
    synthetic = bool(manifest["synthetic"] or config["data"]["synthetic"])
    environment = Environment(autoescape=True)
    environment.filters["display_value"] = _display_value
    html = environment.from_string(_TEMPLATE).render(
        run_id=run_dir.name,
        synthetic=synthetic,
        symbol=config["symbol"],
        start=predictions["query_timestamp"].min(),
        end=predictions["query_timestamp"].max(),
        session_count=predictions["query_session"].nunique(),
        query_count=predictions["query_timestamp"].nunique(),
        primary_embargo=primary_embargo,
        config_text=config_text,
        manifest_text=json.dumps(manifest, indent=2),
        quality_text=json.dumps(quality, indent=2),
        session_coverage=_table(coverage),
        aggregate=_table(aggregate),
        folds=_table(folds),
        calibration=_table(calibration),
        primary_bootstrap=_table(primary),
        secondary_bootstrap=_table(secondary),
        all_bootstrap=_table(bootstrap),
        sensitivity=_table(sensitivity),
        embargoes=", ".join(
            str(value) for value in sorted(sensitivity["embargo_sessions"].unique())
        ),
        conclusion=_conclusion(
            primary, config["horizons_minutes"], synthetic, config["walk_forward"]["n_folds"]
        ),
        diagnostics=diagnostics,
        refits=_table(pd.DataFrame(refit_records["refits"])),
        skipped=_table(pd.DataFrame(refit_records["skipped_blocks"])),
        epsilon=LOG_LOSS_EPSILON,
        error_plot=_error_plot(aggregate),
        calibration_plot=_calibration_plot(calibration),
        sensitivity_plot=_sensitivity_plot(sensitivity),
        examples=_audit_examples(run_dir, predictions, primary_embargo, feature_columns),
        trading=_trading(run_dir, config),
    )
    destination = run_dir / "report.html"
    destination.write_text(html, encoding="utf-8")
    return destination
