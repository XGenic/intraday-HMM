# First real-data experiment — 2026-09-26

This plan is recorded before inspecting real out-of-sample forecast results.

- Archive: SPY, 2019-01-01 through 2025-12-31, Alpaca SIP 5-minute bars,
  unadjusted prices/volume, preserved vendor per-bar VWAP, start timestamps.
  Filter the exchange regular-session grid explicitly; never fill missing bars.
- Reproduce the locked environment and existing 149-test suite on this machine.
- Operational pilot: 2019-01-01 through 2019-07-19. Keep all baseline model,
  history, retrieval, embargo, bootstrap, and fold settings unchanged. Use this
  run to measure time, fit acceptance, storage, and audit integrity.
- First substantive evaluation: all 2019 sessions, with the first eligible
  weekly block after 126 training sessions as the start of evaluation. This
  supplies approximately six months of OOS data and three chronological folds.
- Consider expansion across the full archive only after measuring compute and
  storage requirements. A first-year result is limited to that period; it cannot
  establish stability across 2019–2025.
- No parameter selection using OOS outcomes. Keep trading disabled. The main
  comparison is HMM trajectory minus raw kNN paired absolute error, with whole
  session bootstrap intervals and fold signs at all four horizons. Also inspect
  squared error, Brier loss, embargo sensitivity, failures/degeneracy, clock-time
  concentration, and saved-neighbor reconstruction.

Vendor data and full run artifacts remain local. Selected aggregate evidence is
included with the public source release. Run manifests record source-file hashes
and package versions. The later decision to stop is recorded separately in
[experiment-results.md](experiment-results.md); the scientific plan above is unchanged.
