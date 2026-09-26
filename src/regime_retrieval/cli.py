"""Validate local data, run the six-method latent comparison, and render saved reports."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from regime_retrieval.config import load_config, save_config
from regime_retrieval.data.validation import load_and_validate
from regime_retrieval.evaluation.bootstrap import paired_session_bootstrap
from regime_retrieval.evaluation.metrics import metric_tables
from regime_retrieval.evaluation.sensitivity import sensitivity_metrics
from regime_retrieval.evaluation.trading import simulate_trading
from regime_retrieval.evaluation.walk_forward import NoEvaluableQueries, evaluate
from regime_retrieval.reporting.artifacts import NeighborWriter, json_safe, write_json
from regime_retrieval.reporting.report import render_report


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except OSError:
        return None


def run_research(config_path: str | Path) -> Path:
    started = time.perf_counter()
    config = load_config(config_path)
    bars, quality = load_and_validate(config)
    with config.data.path.open("rb") as handle:
        raw_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    provenance = config.data.path.with_suffix(".manifest.json")
    source = None
    if provenance.exists():
        source = json.loads(provenance.read_text(encoding="utf-8"))
        if source.get("output_sha256") != raw_hash:
            raise ValueError("Input dataset checksum differs from its source manifest")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    run_dir = config.report.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    diagnostics_dir = run_dir / "state_diagnostics"
    diagnostics_dir.mkdir()
    save_config(config, run_dir / "config.yaml")
    embargoes = sorted(
        set([config.retrieval.embargo_sessions, *config.sensitivity.embargo_sessions])
    )
    manifest = {
        "run_id": run_id,
        "symbol": config.symbol,
        "synthetic": config.data.synthetic,
        "raw_file": str(config.data.path),
        "raw_sha256": raw_hash,
        "date_start": bars["timestamp"].iloc[0].isoformat(),
        "date_end": bars["timestamp"].iloc[-1].isoformat(),
        "bar_interval": config.bar_interval,
        "input_timestamp_label": config.data.timestamp_label,
        "internal_timestamp_label": "end",
        "row_count": len(bars),
        "session_count": bars["session"].nunique(),
        "feature_schema_version": "1",
        "artifact_schema_version": "2",
        "hmm_inference": "causal_forward_filter",
        "embargoes": embargoes,
        "git_commit": git_commit(),
        "source_sha256": {
            str(path.relative_to(Path(__file__).parent)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(Path(__file__).parent.rglob("*.py"))
        },
        "python_version": sys.version,
        "package_versions": {
            name: version(name)
            for name in (
                "numpy",
                "pandas",
                "scipy",
                "hmmlearn",
                "scikit-learn",
                "exchange-calendars",
            )
        },
        "status": "running",
    }
    write_json(run_dir / "data_manifest.json", manifest)
    if source is not None:
        write_json(run_dir / "source_manifest.json", source)
    write_json(run_dir / "data_quality.json", quality.to_dict())
    bars.to_parquet(run_dir / "bars.parquet", index=False)
    print(f"Run directory: {run_dir}", file=sys.stderr, flush=True)

    def save_diagnostic(diagnostic):
        write_json(diagnostics_dir / f"{diagnostic['refit_id']}.json", diagnostic)
        print(
            f"{diagnostic['refit_id']}: {diagnostic['status']}; "
            f"{diagnostic['training_sessions']} training sessions; "
            f"elapsed {time.perf_counter() - started:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    try:
        with NeighborWriter(run_dir / "neighbors.parquet") as writer:
            result = evaluate(
                bars,
                config,
                neighbor_sink=writer.write,
                embargoes=embargoes,
                diagnostic_sink=save_diagnostic,
            )
        result.predictions.to_parquet(run_dir / "predictions.parquet", index=False)
        result.representation_examples.to_parquet(
            run_dir / "representation_examples.parquet", index=False
        )
        primary = result.predictions.loc[
            result.predictions["embargo_sessions"].eq(config.retrieval.embargo_sessions)
        ]
        aggregate, folds, calibration = metric_tables(primary)
        write_json(run_dir / "metrics.json", aggregate.to_dict(orient="records"))
        folds.to_csv(run_dir / "fold_metrics.csv", index=False)
        calibration.to_csv(run_dir / "calibration.csv", index=False)
        sensitivity_metrics(result.predictions).to_csv(
            run_dir / "sensitivity_metrics.csv", index=False
        )
        paired_session_bootstrap(result.predictions, config.bootstrap, config.random_state).to_csv(
            run_dir / "bootstrap.csv", index=False
        )
        if config.trading.enabled:
            trading_metrics, trades, equity = simulate_trading(primary, bars, config)
            trading_metrics.to_csv(run_dir / "trading_metrics.csv", index=False)
            trades.to_parquet(run_dir / "trades.parquet", index=False)
            equity.to_parquet(run_dir / "equity.parquet", index=False)
        write_json(
            run_dir / "refits.json",
            {"refits": result.refits, "skipped_blocks": result.skipped_blocks},
        )
        manifest.update(
            status="complete",
            prediction_count=len(result.predictions),
            primary_prediction_count=len(primary),
            query_count=primary["query_timestamp"].nunique(),
            neighbor_record_count=writer.row_count,
            feature_columns=result.refits[-1]["feature_columns"],
            accepted_refit_count=sum(refit["status"] == "accepted" for refit in result.refits),
            failed_refit_count=sum(refit["status"] == "failed" for refit in result.refits),
        )
        write_json(run_dir / "data_manifest.json", manifest)
        render_report(run_dir)
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["artifact_bytes"] = sum(
            path.stat().st_size for path in run_dir.rglob("*") if path.is_file()
        )
        write_json(run_dir / "data_manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        write_json(run_dir / "data_manifest.json", manifest)
        if isinstance(exc, NoEvaluableQueries):
            write_json(
                run_dir / "refits.json",
                {"refits": exc.refits, "skipped_blocks": exc.skipped_blocks},
            )
        raise
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-data", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    report = commands.add_parser("report")
    report.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-data":
            _, quality = load_and_validate(load_config(args.config))
            print(json.dumps(json_safe(quality.to_dict()), indent=2, allow_nan=False))
        elif args.command == "run":
            print(run_research(args.config))
        else:
            print(render_report(args.run))
    except (ValueError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        quality = getattr(exc, "report", None)
        if quality is not None:
            data = quality.to_dict() if hasattr(quality, "to_dict") else quality
            print(json.dumps(json_safe(data), indent=2, allow_nan=False), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
