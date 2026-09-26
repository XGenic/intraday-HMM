"""Create predeclared contiguous subsets and configs without changing model defaults."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from regime_retrieval.reporting.artifacts import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "data/raw/SPY_5m_2019_2025_sip_raw.parquet"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    provenance = json.loads(source.with_suffix(".manifest.json").read_text())
    if source_hash != provenance["output_sha256"]:
        raise ValueError("Archive checksum mismatch")
    bars = pd.read_parquet(source)
    baseline = yaml.safe_load((ROOT / "configs/baseline.yaml").read_text())
    for name, end in [("pilot_2019", "2019-07-19"), ("2019", "2019-12-31")]:
        output = ROOT / f"data/processed/SPY_5m_{name}.parquet"
        if output.exists() or output.with_suffix(".manifest.json").exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        stop = pd.Timestamp(end, tz="America/New_York") + pd.DateOffset(days=1)
        selected = bars.loc[bars.timestamp < stop].copy()
        selected.to_parquet(output, index=False)
        write_json(
            output.with_suffix(".manifest.json"),
            {
                "parent_file": str(source),
                "parent_sha256": source_hash,
                "parent_manifest": str(source.with_suffix(".manifest.json")),
                "selection": {"start": "2019-01-01", "end": end},
                "feed": "sip",
                "adjustment": "raw",
                "timestamp_label": "start",
                "vwap_policy": "preserved vendor per-bar VWAP",
                "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "bar_count": len(selected),
            },
        )
        config = json.loads(json.dumps(baseline))
        config["data"]["path"] = f"../data/processed/{output.name}"
        config["data"]["timestamp_label"] = "start"
        config["report"]["output_dir"] = f"../runs/real-{name}"
        path = ROOT / f"configs/alpaca_{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(f"{name}: {len(selected):,} bars; {path}")
    baseline["data"]["path"] = f"../data/raw/{source.name}"
    baseline["data"]["timestamp_label"] = "start"
    baseline["report"]["output_dir"] = "../runs/real-2019-2025"
    (ROOT / "configs/alpaca_2019_2025.yaml").write_text(
        yaml.safe_dump(baseline, sort_keys=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
