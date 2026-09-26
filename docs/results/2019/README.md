# Published 2019 evidence

These files were extracted from the completed frozen 2019 evaluation. They contain
aggregate results, an audit summary, a derived figure, and selected run metadata.
No vendor bar rows, individual forecasts, neighbor cohorts, credentials, or local
filesystem paths are included.

| File | Contents |
|---|---|
| [metrics.json](metrics.json) | Primary-embargo aggregate metrics for all six methods |
| [fold_metrics.csv](fold_metrics.csv) | Three chronological fold summaries |
| [bootstrap.csv](bootstrap.csv) | Paired session-bootstrap comparisons, all folds/embargoes |
| [sensitivity_metrics.csv](sensitivity_metrics.csv) | Metrics for embargoes 0, 1, and 5 |
| [calibration.csv](calibration.csv) | Aggregate probability-bin calibration |
| [audit.json](audit.json) | Recorded outcome of the original saved-artifact audit |
| [run-summary.json](run-summary.json) | Sanitized provenance, versions, hashes, and counts |
| [hmm-vs-raw-mae.png](hmm-vs-raw-mae.png) | Primary MAE differences in basis points |

Bootstrap differences are model minus raw kNN; negative favors the model. Fold
zero is the aggregate. Horizon values are minutes. Numeric errors/returns in the
CSV/JSON tables use fractional-return units; the displayed MAE figure and result
tables multiply them by 10,000 to express basis points. Brier scores remain
dimensionless, and squared-error differences are squared-return units.

The audit summary records checks performed against the original private artifacts.
Publishing this summary is not equivalent to publishing enough data to repeat
those checks. Full verification requires your own authorized data snapshot and a
new run, or access to the original inputs and complete run artifacts.

Source hashes describe the implementation used for the historical run, before
publication-only changes to documentation, packaging, and auxiliary tools. They
are provenance identifiers, not a claim that every current file has the same hash.
Vendor revisions and numerical environments may affect fresh reproductions.

See the [findings](../../../experiment-results.md),
[methodology](../../methodology.md), and [reproduction instructions](../../reproduction.md).
