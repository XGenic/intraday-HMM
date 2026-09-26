# Public source boundary

The public repository contains source, tests, frozen configurations, documentation,
an empty credential template, and selected aggregate results. Vendor bars,
credentials, full forecasts/references, reports, source snapshots, local notes,
virtual environments, and caches stay local. The MIT license does not redistribute
vendor data rights.

`.gitignore` protects those local paths during normal Git use. Source distributions
also use an explicit inclusion list in `pyproject.toml`. A repository-shaped ZIP
can be built without Git metadata:

```sh
python scripts/build_public_snapshot.py
```

This writes `dist/intraday-HMM-public.zip` plus a SHA-256 file, using a reviewed
allowlist. It excludes ignored working material even when the workspace itself
contains data and credentials. It does not commit, push, upload, or change repository
visibility. Review the actual archive before distributing it; newly added public
files also need review.

The source export does not rewrite Git history. Before changing an existing
remote to public, inspect its tracked files and history for previously committed
credentials or restricted data. `.gitignore` does not remove tracked files or
historical versions, and deleting old working notes from the current tree leaves
them in earlier commits. If a credential was previously committed, revoke it
before publication.

Published evidence includes aggregate tables and selected metadata only; it cannot
independently prove the per-forecast audit without the private input snapshot and
full run artifacts. The original plan and numerical outcome are preserved without
claiming that the unexecuted multi-year study was completed.
