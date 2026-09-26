# Reproducing the frozen experiment

Use the [README](../README.md) for installation, tests, and a direct 2019 download.
Python 3.13.6 with the locked numerical stack was used for the reported run.
Dependencies and platform differences can affect numerical fitting; recorded
package versions are in the [sanitized run summary](results/2019/run-summary.json).

## Inputs and original snapshot

The original source archive was SPY, 2019-01-01 through 2025-12-31, from Alpaca's
historical bars endpoint. Requests specified `feed=sip`, `timeframe=5Min`,
`adjustment=raw`, `asof=-`, ascending timestamps, and pagination. Date bounds are
inclusive exchange-local dates. Input timestamps label starts; vendor `vw` is
preserved as per-bar VWAP. Filtering explicitly retains the XNYS regular-session
grid. No market bars are forward-filled.

The archive contained 136,727 regular-session bars across all 1,760 scheduled
sessions. Missing counts were five bars on 2019-08-12 and two each on
2020-03-09, 2020-03-12, 2020-03-16, and 2020-03-18. Minimum observed coverage
was 93.59%. Only 2019 was evaluated; later years were downloaded but not tested.

Original archive SHA-256:

```text
739899265b38874d86f83c0ec577afbe9f66772a26ab8bfe2ab6ee42e47c8aab
```

The hash identifies one Parquet snapshot, including its serialization. Different
library versions or vendor revisions may produce a different hash. Exact
reproduction requires the original input snapshot and numerical environment;
a fresh download is a new data snapshot, even with the same parameters.

Alpaca access is required only for downloading. Use your own credentials and verify
current access against the vendor's [bars API](https://docs.alpaca.markets/us/reference/stockbarsingle-1)
and [market-data FAQ](https://docs.alpaca.markets/us/docs/market-data-faq).
The downloader sends authentication only to the historical data endpoint, saves
no authentication headers, retries transient errors, and verifies cached pages.
It refuses to overwrite a dataset or its provenance sidecar.

## Original archive preparation route

These commands download and prepare data; they do not run the seven-year experiment.
Use fresh output paths. The direct 2019 download in the README is sufficient to
rerun that period; the route below also creates the exact pilot selection.

```sh
uv run --locked python -m regime_retrieval.data.alpaca --start 2019-01-01 --end 2025-12-31 --output data/raw/SPY_5m_2019_2025_sip_raw.parquet
uv run --locked python scripts/prepare_real_experiment.py
```

Preparation writes two derived Parquet files and their parent hashes, then
generates the three `alpaca_*.yaml` configurations from the frozen baseline.
It refuses to overwrite existing subsets. Model and evaluation settings remain
identical; only input/output paths and the required start-label convention differ.

| Configuration | Input period | Purpose |
|---|---|---|
| `alpaca_pilot_2019.yaml` | January 1–July 19, 2019 | Runtime/storage pilot; ten OOS sessions |
| `alpaca_2019.yaml` | Calendar 2019 | Completed substantive experiment |
| `alpaca_2019_2025.yaml` | Calendar 2019–2025 | Unexecuted proposal retained for provenance |

Run the pilot or 2019 configuration with the normal `regime-retrieval run` command.
To verify that adding later history leaves earlier predictions unchanged:

```sh
uv run --locked python scripts/audit_research.py runs/real-2019/<run_id> --prefix-run runs/real-pilot_2019/<pilot_run_id>
```

This audit script targets the frozen five-minute, four-horizon experiment. Run it
with normal Python, without `-O`, because its audit invariants use assertions.
Fold assignments are deliberately excluded from the prefix comparison: their
boundaries depend on the full planned evaluation period.

## Outputs and storage

A completed run writes a resolved config, input hash and source manifest, data
quality, canonical bars, forecasts, streamed neighbor references, refit/model
diagnostics, aggregate/fold/calibration tables, bootstrap and sensitivity tables,
saved representation examples, and a standalone `report.html`. The separate audit
adds `audit.json`. Enabled trading adds trades, marked equity, and trading metrics.

The HTML report can be regenerated from saved run artifacts without downloading
data or refitting. The full reference audit also checks the source Parquet hash;
saved configs contain absolute paths, so adjust them when moving artifacts.
Generated manifests/reports can contain local paths and should be reviewed before
sharing. The published [aggregate evidence](results/2019/README.md) excludes them.

The 2019 run took 1,321.5 seconds and produced 2.85 GB on the original machine.
The operational pilot took 67.1 seconds. A simple pilot extrapolation for all
seven years suggested 8.62 billion reference rows, about 178 GB, and 21 hours;
these are sizing estimates, not completed measurements. That expansion was stopped.

Tests and the ordinary CLI do not synthesize missing input. To validate your own
dataset, provide its Parquet path and timestamp convention in a copied config.
If Matplotlib cannot write its default cache, set `MPLCONFIGDIR` to a writable
directory such as `.mpl-cache` before running the CLI.
