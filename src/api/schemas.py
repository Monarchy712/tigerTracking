"""Pydantic schemas for API responses."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class RunSummary(BaseModel):
    id: int
    started_at: datetime
    completed_at: datetime | None
    input_dir: str
    total_frames: int
    blank_removed: int
    tiger_detected: int
    new_individuals: int
    auto_matched: int
    pending_review: int
    bytes_quarantined: int
    estimated_time_saved_sec: float
    status: str


class TigerOut(BaseModel):
    id: int
    tiger_code: str
    enrolled_at: datetime
    sighting_count: int = 0


class SightingOut(BaseModel):
    id: int
    tiger_code: str
    station_id: str | None
    captured_at: datetime | None
    latitude: float | None
    longitude: float | None
    match_confidence: float | None
    match_type: str


class ReviewOut(BaseModel):
    id: int
    image_id: int
    candidate_tiger_id: int | None
    candidate_tiger_code: str | None
    confidence: float
    status: str
    created_at: datetime
    flank_path: str | None = None
    image_path: str | None = None


class OccupancyOut(BaseModel):
    tiger_code: str
    centroid_lat: float | None
    centroid_lon: float | None
    area_sq_km: float | None
    capture_count: int
    station_ids: list[str]


class AlertOut(BaseModel):
    id: int
    tiger_code: str | None
    alert_type: str
    severity: str
    confidence: float
    title: str
    description: str
    is_survey_artifact: bool
    created_at: datetime
    acknowledged: bool


class PipelineRunRequest(BaseModel):
    input_dir: str
    station_registry: str | None = None
    recursive: bool = True


class PipelineRunResponse(BaseModel):
    run_id: int
    total_frames: int
    blank_removed: int
    tiger_detected: int
    new_individuals: int
    auto_matched: int
    pending_review: int
    bytes_quarantined: int
    bytes_quarantined_mb: float
    estimated_time_saved_sec: float
    estimated_time_saved_min: float
    occupancy_count: int
    alerts_raised: int
    exports: dict


class IdentifyResponse(BaseModel):
    success: bool
    message: str
    has_tiger: bool = False
    tiger_id: int | None = None
    tiger_code: str | None = None
    confidence: float = 0.0
    action: str = "none"
    flank_path: str | None = None
    matched_against: str | None = None
    last_station_id: str | None = None
    last_captured_at: datetime | None = None
    last_latitude: float | None = None
    last_longitude: float | None = None
    last_zone: str | None = None
    total_sightings: int = 0
    detection_confidence: float = 0.0


class CompareResponse(BaseModel):
    same_tiger: bool
    verdict: str
    confidence: float
    message: str
    flank_a: str | None = None
    flank_b: str | None = None
