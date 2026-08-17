"""Application configuration loaded from config.yaml and environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


ROOT_DIR = Path(__file__).resolve().parent.parent


class ModelsConfig(BaseModel):
    enabled: bool = True
    megadetector_threshold: float = 0.20
    megadetector_version: str = "MDV6-yolov10-c"
    miewid_model: str = "conservationxlabs/miewid-msv3"
    device: str = "cpu"


class BlankFilterConfig(BaseModel):
    mode: str = "hybrid"  # heuristic | megadetector | hybrid
    confidence_threshold: float = 0.75
    variance_threshold: float = 120.0
    edge_density_threshold: float = 0.015


class MatchingConfig(BaseModel):
    auto_match_threshold: float = 0.85
    review_threshold: float = 0.60
    max_representatives_per_tiger: int = 3


class AlertsConfig(BaseModel):
    core_range_shift_sqkm: float = 17.5
    buffer_range_shift_km: float = 5.0
    absence_days_threshold: int = 30
    buffer_station_proximity_km: float = 2.0


class AnomalyConfig(BaseModel):
    """Visual/behavioural anomaly detection thresholds."""

    enabled: bool = True
    min_frames: int = 2               # an anomaly must recur before an alert fires
    injured_aspect_min: float = 1.15  # body bbox w/h below this reads as crouched/limping
    injured_aspect_max: float = 3.40  # above this reads as an elongated/dragging posture
    blood_pixel_fraction: float = 0.04
    mating_window_hours: float = 6.0
    water_lookback_days: int = 30
    water_sigma: float = 2.0


class ReportConfig(BaseModel):
    output_dir: str = "reports"
    top_alerts: int = 10


class OccupancyConfig(BaseModel):
    home_range_method: str = "convex_hull"
    min_points_for_range: int = 3
    overlap_threshold_pct: float = 10.0


class ReserveConfig(BaseModel):
    name: str = "Pench Tiger Reserve (Nagpur Region)"
    center_lat: float = 21.771
    center_lon: float = 79.323
    state: str = "Maharashtra"
    hackathon: str = "Viksit Nagpur Hackathon"
    core_boundary_file: str | None = None
    buffer_boundary_file: str | None = None


class DemoConfig(BaseModel):
    """Options for datasets without camera-trap metadata (e.g. CVWC2019 Amur Tiger zip)."""
    synthetic_stations: bool = True
    synthetic_station_seed: int = 42


class AppConfig(BaseModel):
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    demo: DemoConfig = Field(default_factory=DemoConfig)
    blank_filter: BlankFilterConfig = Field(default_factory=BlankFilterConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    occupancy: OccupancyConfig = Field(default_factory=OccupancyConfig)
    reserve: ReserveConfig = Field(default_factory=ReserveConfig)


class Settings(BaseSettings):
    database_url: str = f"sqlite:///{ROOT_DIR / 'data' / 'tiger_tracking.db'}"
    friend_model_url: str = "http://localhost:8001/compare"
    friend_model_mock: bool = False
    use_ml_models: bool = True
    data_dir: Path = ROOT_DIR / "data"

    class Config:
        env_file = ROOT_DIR / ".env"
        extra = "ignore"


def load_config(config_path: Path | None = None) -> AppConfig:
    path = config_path or ROOT_DIR / "config.yaml"
    if not path.exists():
        return AppConfig()
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return AppConfig(**raw)


settings = Settings()
app_config = load_config()


def ensure_data_dirs() -> dict[str, Path]:
    base = settings.data_dir
    dirs = {
        "raw": base / "raw",
        "quarantine_blank": base / "quarantine" / "blank",
        "flanks": base / "processed" / "flanks",
        "exports": base / "exports",
        "reports": ROOT_DIR / app_config.report.output_dir,
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
