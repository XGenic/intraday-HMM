# Contributing

The frozen 2019 experiment is complete. Keep its configuration and reported
negative finding intact when fixing implementation or documentation issues.
Treat a new representation, changed hyperparameter, or different evaluation period
as a separate experiment with its own plan recorded before examining results.

Use the dependency versions in `uv.lock` and run:

```sh
uv sync --locked --extra dev --python 3.13
uv run --locked pytest -q
uv run --locked ruff check src tests scripts
uv run --locked ruff format --check src tests scripts
```

Tests must run without network requests, credentials, or downloaded bars. Changes
to preprocessing, filtering, eligibility, or evaluation need regression evidence
that past-only fitting, availability cutoffs, common cohorts, and future-data
independence remain valid. Avoid changing numerical dependencies incidentally.

Do not attach credentials, vendor responses, full run directories, or identifying
local paths to issues or pull requests. Small constructed fixtures and aggregate
diagnostics are preferable. Describe the behavior changed and the checks performed.
