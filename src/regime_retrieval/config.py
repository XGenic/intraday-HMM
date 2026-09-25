"""Strict, serializable configuration; relative paths resolve beside the YAML file."""

from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DataConfig(StrictModel):
    provider: Literal["parquet"] = "parquet"
    path: Path = Path("../data/raw/SPY_5m.parquet")
    timestamp_label: Literal["start", "end"] = "end"
    calendar: str = "XNYS"
    min_session_coverage: float = Field(default=0.8, gt=0, le=1)
    synthetic: bool = False


class FeatureConfig(StrictModel):
    log_return: bool = True
    realized_vol_window: int = Field(default=12, ge=2)
    range_fraction: bool = True
    volume_tod_adjusted: bool = True
    vwap_distance: bool = True


class RetrievalConfig(StrictModel):
    k: int = Field(default=50, ge=1)
    distance: Literal["euclidean", "cosine"] = "euclidean"
    max_neighbors_per_session: int = Field(default=3, ge=1)
    embargo_sessions: int = Field(default=1, ge=0)


class WalkForwardConfig(StrictModel):
    min_training_sessions: int = Field(default=126, ge=1)
    refit_frequency: Literal["daily", "weekly", "monthly"] = "weekly"
    n_folds: int = Field(default=3, ge=1)


class HMMConfig(StrictModel):
    n_states: int = Field(default=6, ge=2)
    covariance_type: Literal["full", "diag"] = "full"
    n_iter: int = Field(default=500, ge=2)
    tol: float = Field(default=1e-4, gt=0)
    min_covar: float = Field(default=1e-3, gt=0)
    random_state: int = Field(default=42, ge=0)
    retry_seed_offsets: list[int] = Field(default_factory=lambda: [0, 1, 2], min_length=1)

    @model_validator(mode="after")
    def check_seeds(self) -> "HMMConfig":
        if self.retry_seed_offsets != sorted(set(self.retry_seed_offsets)):
            raise ValueError("retry seed offsets must be unique and increasing")
        if self.retry_seed_offsets[0] != 0:
            raise ValueError("retry seed offsets must start with zero")
        return self


class BootstrapConfig(StrictModel):
    n_resamples: int = Field(default=1000, ge=20)
    confidence: float = Field(default=0.95, gt=0, lt=1)


class SensitivityConfig(StrictModel):
    embargo_sessions: list[int] = Field(default_factory=lambda: [0, 1, 5], min_length=1)

    @model_validator(mode="after")
    def check_embargoes(self) -> "SensitivityConfig":
        if any(value < 0 for value in self.embargo_sessions):
            raise ValueError("sensitivity embargoes must be nonnegative")
        if self.embargo_sessions != sorted(set(self.embargo_sessions)):
            raise ValueError("sensitivity embargoes must be unique and increasing")
        return self


class CostsConfig(StrictModel):
    bps_per_side: float = Field(default=1.0, ge=0, lt=10000)


class TradingConfig(StrictModel):
    enabled: bool = False
    threshold: float = Field(default=0.0, ge=0)


class ReportConfig(StrictModel):
    output_dir: Path = Path("../runs")


class ResearchConfig(StrictModel):
    symbol: str = Field(default="SPY", min_length=1)
    bar_interval: str = Field(default="5m", pattern=r"^[1-9][0-9]*m$")
    regular_hours_only: Literal[True] = True
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    context_bars: int = Field(default=12, ge=1)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    horizons_minutes: list[int] = Field(default_factory=lambda: [5, 15, 30, 60], min_length=1)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    hmm: HMMConfig = Field(default_factory=HMMConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    sensitivity: SensitivityConfig = Field(default_factory=SensitivityConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    random_state: int = Field(default=42, ge=0)

    @property
    def bar_minutes(self) -> int:
        return int(self.bar_interval[:-1])

    @model_validator(mode="after")
    def check_horizons(self) -> "ResearchConfig":
        horizons = self.horizons_minutes
        if any(h <= 0 or h % self.bar_minutes for h in horizons):
            raise ValueError("horizons must be positive multiples of the bar interval")
        if horizons != sorted(set(horizons)):
            raise ValueError("horizons must be unique and increasing")
        return self


def load_config(path: str | Path) -> ResearchConfig:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = ResearchConfig.model_validate(yaml.safe_load(handle))
    for section, name in ((config.data, "path"), (config.report, "output_dir")):
        value = getattr(section, name)
        if not value.is_absolute():
            setattr(section, name, (path.parent / value).resolve())
    return config


def save_config(config: ResearchConfig, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))


def refit_key(timestamp: pd.Timestamp, frequency: str) -> str:
    """Calendar keys are evaluated in exchange-local session dates, not UTC weeks."""
    if frequency == "daily":
        return timestamp.strftime("%Y-%m-%d")
    if frequency == "monthly":
        return timestamp.strftime("%Y-%m")
    year, week, _ = timestamp.isocalendar()
    return f"{year}-W{week:02d}"
