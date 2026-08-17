"""FastAPI routes for tiger tracking system."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.api.schemas import (
    AlertOut,
    OccupancyOut,
    PipelineRunRequest,
    PipelineRunResponse,
    ReviewOut,
    RunSummary,
    SightingOut,
    TigerOut,
)
from src.db.models import ImageRecord, MatchReview, get_session, init_db
from src.db.repository import Repository
from src.matching.catalogue import TigerCatalogue
from src.matching.review_queue import ReviewQueue
from src.pipeline.run import TigerTrackingPipeline

router = APIRouter()


def _get_repo() -> Repository:
    init_db()
    return Repository(get_session())


@router.post("/pipeline/run", response_model=PipelineRunResponse)
def run_pipeline(req: PipelineRunRequest):
    input_dir = Path(req.input_dir)
    if not input_dir.exists():
        raise HTTPException(404, f"Input directory not found: {input_dir}")

    pipeline = TigerTrackingPipeline()
    try:
        registry = Path(req.station_registry) if req.station_registry else None
        report = pipeline.run(input_dir, registry, req.recursive)
    finally:
        pipeline.close()

    return PipelineRunResponse(
        run_id=report.run_id,
        total_frames=report.total_frames,
        blank_removed=report.blank_removed,
        tiger_detected=report.tiger_detected,
        new_individuals=report.new_individuals,
        auto_matched=report.auto_matched,
        pending_review=report.pending_review,
        bytes_quarantined=report.bytes_quarantined,
        bytes_quarantined_mb=round(report.bytes_quarantined / (1024 * 1024), 2),
        estimated_time_saved_sec=report.estimated_time_saved_sec,
        estimated_time_saved_min=round(report.estimated_time_saved_sec / 60, 1),
        occupancy_count=report.occupancy_count,
        alerts_raised=report.alerts_raised,
        exports=report.exports,
    )


@router.get("/runs", response_model=list[RunSummary])
def list_runs():
    repo = _get_repo()
    return [
        RunSummary(
            id=r.id,
            started_at=r.started_at,
            completed_at=r.completed_at,
            input_dir=r.input_dir,
            total_frames=r.total_frames,
            blank_removed=r.blank_removed,
            tiger_detected=r.tiger_detected,
            new_individuals=r.new_individuals,
            auto_matched=r.auto_matched,
            pending_review=r.pending_review,
            bytes_quarantined=r.bytes_quarantined,
            estimated_time_saved_sec=r.estimated_time_saved_sec,
            status=r.status,
        )
        for r in repo.get_runs()
    ]


@router.get("/tigers", response_model=list[TigerOut])
def list_tigers():
    repo = _get_repo()
    result = []
    for t in repo.list_tigers():
        sightings = repo.get_sightings_for_tiger(t.id)
        result.append(TigerOut(
            id=t.id,
            tiger_code=t.tiger_code,
            enrolled_at=t.enrolled_at,
            sighting_count=len(sightings),
        ))
    return result


@router.get("/tigers/{tiger_id}/sightings", response_model=list[SightingOut])
def tiger_sightings(tiger_id: int):
    repo = _get_repo()
    tiger = repo.get_tiger(tiger_id)
    if not tiger:
        raise HTTPException(404, "Tiger not found")
    return [
        SightingOut(
            id=s.id,
            tiger_code=tiger.tiger_code,
            station_id=s.station_id,
            captured_at=s.captured_at,
            latitude=s.latitude,
            longitude=s.longitude,
            match_confidence=s.match_confidence,
            match_type=s.match_type,
        )
        for s in repo.get_sightings_for_tiger(tiger_id)
    ]


@router.get("/reviews/pending", response_model=list[ReviewOut])
def pending_reviews():
    repo = _get_repo()
    session = repo.session
    reviews = repo.get_pending_reviews()
    result = []
    for r in reviews:
        img = session.get(ImageRecord, r.image_id)
        candidate = repo.get_tiger(r.candidate_tiger_id) if r.candidate_tiger_id else None
        result.append(ReviewOut(
            id=r.id,
            image_id=r.image_id,
            candidate_tiger_id=r.candidate_tiger_id,
            candidate_tiger_code=candidate.tiger_code if candidate else None,
            confidence=r.confidence,
            status=r.status,
            created_at=r.created_at,
            flank_path=img.flank_path if img else None,
            image_path=img.working_path or img.original_path if img else None,
        ))
    return result


@router.post("/reviews/{review_id}/resolve")
def resolve_review(review_id: int, decision: str, tiger_id: int | None = None):
    repo = _get_repo()
    review = repo.session.get(MatchReview, review_id)
    if not review:
        raise HTTPException(404, "Review not found")

    img = repo.session.get(ImageRecord, review.image_id)
    if not img:
        raise HTTPException(404, "Image not found")

    catalogue = TigerCatalogue(repo)
    queue = ReviewQueue(repo, catalogue)

    if decision == "same":
        if not tiger_id:
            tiger_id = review.candidate_tiger_id
        if not tiger_id:
            raise HTTPException(400, "tiger_id required for 'same' decision")
        queue.resolve_same(
            review_id, tiger_id, img.id, img.flank_path or "",
            img.run_id or 0, img.station_id, img.captured_at,
            img.latitude, img.longitude, review.confidence,
        )
    elif decision == "new":
        new_id = queue.resolve_new(
            review_id, img.id, img.flank_path or "",
            img.run_id or 0, img.station_id, img.captured_at,
            img.latitude, img.longitude,
        )
        return {"status": "resolved", "new_tiger_id": new_id}
    elif decision == "reject":
        queue.resolve_reject(review_id)
    else:
        raise HTTPException(400, f"Unknown decision: {decision}")

    return {"status": "resolved"}


@router.get("/runs/{run_id}/occupancy", response_model=list[OccupancyOut])
def run_occupancy(run_id: int):
    repo = _get_repo()
    snaps = repo.get_occupancy_for_run(run_id)
    result = []
    for s in snaps:
        tiger = repo.get_tiger(s.tiger_id)
        result.append(OccupancyOut(
            tiger_code=tiger.tiger_code if tiger else "?",
            centroid_lat=s.centroid_lat,
            centroid_lon=s.centroid_lon,
            area_sq_km=s.area_sq_km,
            capture_count=s.capture_count,
            station_ids=s.station_ids.split(",") if s.station_ids else [],
        ))
    return result


@router.get("/runs/{run_id}/alerts", response_model=list[AlertOut])
def run_alerts(run_id: int):
    repo = _get_repo()
    alerts = repo.get_alerts(run_id=run_id)
    result = []
    for a in alerts:
        tiger = repo.get_tiger(a.tiger_id) if a.tiger_id else None
        result.append(AlertOut(
            id=a.id,
            tiger_code=tiger.tiger_code if tiger else None,
            alert_type=a.alert_type,
            severity=a.severity,
            confidence=a.confidence,
            title=a.title,
            description=a.description,
            is_survey_artifact=a.is_survey_artifact,
            created_at=a.created_at,
            acknowledged=a.acknowledged,
        ))
    return result


@router.get("/exports/{run_id}/map")
def download_map(run_id: int):
    from src.config import settings
    map_path = settings.data_dir / "exports" / f"run_{run_id}" / "occupancy_map.html"
    if not map_path.exists():
        raise HTTPException(404, "Map not found for this run")
    return FileResponse(map_path, media_type="text/html", filename=f"occupancy_map_run_{run_id}.html")


@router.post("/quarantine/restore/{run_id}")
def restore_quarantined(run_id: int):
    """Restore all blank-quarantined images from a run back to raw directory."""
    import shutil
    from src.config import settings

    repo = _get_repo()
    quarantine_dir = settings.data_dir / "quarantine" / "blank" / f"run_{run_id}"
    raw_dir = settings.data_dir / "raw" / f"restored_run_{run_id}"
    raw_dir.mkdir(parents=True, exist_ok=True)

    restored = 0
    if quarantine_dir.exists():
        for f in quarantine_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, raw_dir / f.name)
                restored += 1

    return {"restored": restored, "destination": str(raw_dir)}
