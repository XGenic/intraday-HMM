from pathlib import Path

import pytest
from pydantic import ValidationError

from regime_retrieval.config import ResearchConfig, load_config


def test_config_resolves_paths_relative_to_yaml(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("data:\n  path: bars.parquet\nreport:\n  output_dir: output\n")
    config = load_config(path)
    assert config.data.path == tmp_path / "bars.parquet"
    assert config.report.output_dir == tmp_path / "output"


@pytest.mark.parametrize(
    "values",
    [
        {"horizons_minutes": [7]},
        {"horizons_minutes": [15, 5]},
        {"horizons_minutes": [5, 5]},
        {"retrieval": {"k": 0}},
        {"data": {"timestamp_label": "guess"}},
        {"features": {"realized_vol_window": 1}},
        {"hmm": {"n_states": 1}},
        {"hmm": {"n_iter": 1}},
        {"hmm": {"retry_seed_offsets": [1, 2]}},
        {"hmm": {"retry_seed_offsets": [0, 0]}},
        {"sensitivity": {"embargo_sessions": [-1, 0]}},
        {"sensitivity": {"embargo_sessions": [1, 0]}},
        {"bootstrap": {"confidence": 1}},
        {"costs": {"bps_per_side": -1}},
        {"unknown_tuning_parameter": True},
    ],
)
def test_invalid_or_misspelled_configuration_fails(values):
    with pytest.raises(ValidationError):
        ResearchConfig.model_validate(values)


def test_shipped_baseline_is_loadable():
    config = load_config(Path(__file__).parents[1] / "configs/baseline.yaml")
    assert config.data.path.is_absolute()
