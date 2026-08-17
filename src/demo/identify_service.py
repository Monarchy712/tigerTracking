"""Single-image tiger identification for judge demo."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.config import app_config, ensure_data_dirs
from src.db.repository import Repository
from src.matching.catalogue import TigerCatalogue
from src.pipeline.detect_crop import detect_and_crop
from src.pipeline.ingest import load_station_registry


@dataclass
class IdentifyResult:
    success: bool
    message: str
    is_blank: bool = False
    has_tiger: bool = False
    tiger_id: int | None = None
    tiger_code: str | None = None
    confidence: float = 0.0
    action: str = "none"  # auto_match | review | enroll | no_tiger
    flank_path: str | None = None
    matched_against: str | None = None
    last_station_id: str | None = None
    last_captured_at: datetime | None = None
    last_latitude: float | None = None
    last_longitude: float | None = None
    last_zone: str | None = None
    total_sightings: int = 0
    detection_confidence: float = 0.0


def _station_for_upload(path: Path, registry: dict) -> tuple[str | None, float | None, float | None, str | None]:
    if not registry or not app_config.demo.synthetic_stations:
        return None, None, None, None
    stations = sorted(registry.keys())
    digest = hashlib.md5(path.name.encode()).hexdigest()
    sid = stations[int(digest, 16) % len(stations)]
    lat, lon, zone = registry[sid]
    return sid, lat, lon, zone


class IdentifyService:
    def __init__(self, repo: Repository):
        self.repo = repo
        self.catalogue = TigerCatalogue(repo)
        self.dirs = ensure_data_dirs()

    def identify_file(
        self,
        image_path: Path,
        station_registry_path: Path | None = None,
        record_sighting: bool = False,
        run_id: int | None = None,
    ) -> IdentifyResult:
        image_path = Path(image_path)
        registry = load_station_registry(station_registry_path) if station_registry_path else {}

        import time

        image_id = int(time.time() * 1000) % 10_000_000
        detection = detect_and_crop(image_path, self.dirs["flanks"], image_id=image_id)
        if not detection.has_tiger or not detection.flank_path:
            return IdentifyResult(
                success=False,
                message="No tiger detected in this image.",
                has_tiger=False,
                detection_confidence=detection.confidence,
            )

        decision = self.catalogue.find_best_match(detection.flank_path)

        tiger_id: int | None = None
        tiger_code: str | None = None
        action = decision.action

        if decision.action == "enroll":
            if record_sighting:
                img = self.repo.add_image(
                    original_path=str(image_path),
                    working_path=str(image_path),
                    flank_path=str(detection.flank_path),
                    status="identified",
                )
                tiger_id = self.catalogue.enroll_tiger(img.id, detection.flank_path)
                station_id, lat, lon, _ = _station_for_upload(image_path, registry)
                if station_id:
                    self.repo.upsert_station(station_id, lat, lon)
                    self.catalogue.record_sighting(
                        tiger_id=tiger_id,
                        image_id=img.id,
                        run_id=run_id,
                        station_id=station_id,
                        captured_at=datetime.utcnow(),
                        latitude=lat,
                        longitude=lon,
                        confidence=decision.confidence,
                        match_type="enroll",
                    )
            else:
                action = "enroll"
        elif decision.tiger_id:
            tiger_id = decision.tiger_id
            tiger = self.repo.get_tiger(tiger_id)
            tiger_code = tiger.tiger_code if tiger else None

            if record_sighting and decision.action == "auto_match":
                station_id, lat, lon, _ = _station_for_upload(image_path, registry)
                img = self.repo.add_image(
                    original_path=str(image_path),
                    working_path=str(image_path),
                    flank_path=str(detection.flank_path),
                    station_id=station_id,
                    latitude=lat,
                    longitude=lon,
                    captured_at=datetime.utcnow(),
                    status="identified",
                )
                if station_id and lat and lon:
                    self.repo.upsert_station(station_id, lat, lon)
                self.catalogue.add_representative_if_needed(
                    tiger_id, img.id, detection.flank_path
                )
                self.catalogue.record_sighting(
                    tiger_id=tiger_id,
                    image_id=img.id,
                    run_id=run_id,
                    station_id=station_id,
                    captured_at=datetime.utcnow(),
                    latitude=lat,
                    longitude=lon,
                    confidence=decision.confidence,
                    match_type="auto",
                )

        last = self.repo.get_last_sighting(tiger_id) if tiger_id else None
        if tiger_id and not tiger_code:
            t = self.repo.get_tiger(tiger_id)
            tiger_code = t.tiger_code if t else None

        sightings_count = len(self.repo.get_sightings_for_tiger(tiger_id)) if tiger_id else 0

        zone = None
        if last and last.station_id:
            st = self.repo.get_station(last.station_id)
            zone = st.zone if st else None

        messages = {
            "auto_match": f"Recognized as {tiger_code} — same individual.",
            "review": f"Likely {tiger_code}, but needs human review.",
            "enroll": "New individual — not in catalogue yet.",
        }

        return IdentifyResult(
            success=True,
            message=messages.get(action, "Identification complete."),
            has_tiger=True,
            tiger_id=tiger_id,
            tiger_code=tiger_code,
            confidence=decision.confidence,
            action=action,
            flank_path=str(detection.flank_path),
            matched_against=decision.compared_against,
            last_station_id=last.station_id if last else None,
            last_captured_at=last.captured_at if last else None,
            last_latitude=last.latitude if last else None,
            last_longitude=last.longitude if last else None,
            last_zone=zone,
            total_sightings=sightings_count,
            detection_confidence=detection.confidence,
        )

    def compare_files(self, path_a: Path, path_b: Path) -> dict:
        from src.ml.model_registry import cosine_similarity, embed_flank_image, ml_available
        from src.matching.friend_model_client import MockFriendModelClient

        dirs = self.dirs["flanks"]
        import time

        uid = int(time.time() * 1000) % 10_000_000
        det_a = detect_and_crop(path_a, dirs, uid)
        det_b = detect_and_crop(path_b, dirs, uid + 1)

        if not det_a.flank_path or not det_b.flank_path:
            missing = []
            if not det_a.flank_path:
                missing.append("Image A")
            if not det_b.flank_path:
                missing.append("Image B")
            return {
                "same_tiger": False,
                "verdict": "error",
                "confidence": 0.0,
                "message": f"Could not extract tiger flank from: {', '.join(missing)}.",
                "flank_a": str(det_a.flank_path) if det_a.flank_path else None,
                "flank_b": str(det_b.flank_path) if det_b.flank_path else None,
            }

        if ml_available():
            emb_a = embed_flank_image(det_a.flank_path)
            emb_b = embed_flank_image(det_b.flank_path)
            sim = cosine_similarity(emb_a, emb_b)
        else:
            sim = MockFriendModelClient().compare(det_a.flank_path, det_b.flank_path).confidence

        cfg = app_config.matching
        if sim >= cfg.auto_match_threshold:
            verdict = "same"
            msg = "Same tiger — stripe patterns match."
        elif sim >= cfg.review_threshold:
            verdict = "uncertain"
            msg = "Possibly the same tiger — ambiguous match."
        else:
            verdict = "different"
            msg = "Different tigers — distinct stripe patterns."

        return {
            "same_tiger": verdict == "same",
            "verdict": verdict,
            "confidence": round(sim, 4),
            "message": msg,
            "flank_a": str(det_a.flank_path),
            "flank_b": str(det_b.flank_path),
        }

    def save_upload(self, filename: str, content: bytes) -> Path:
        upload_dir = self.dirs["raw"] / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(filename).name
        dest = upload_dir / f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{safe_name}"
        dest.write_bytes(content)
        return dest
