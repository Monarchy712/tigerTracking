"""Main pipeline orchestrator — ties all stages together."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.alerts.anomaly import AnomalyEngine, analyse_frame
from src.alerts.deviation import AlertEngine
from src.config import app_config, ensure_data_dirs
from src.db.models import get_session, init_db
from src.db.repository import Repository
from src.integrations.mstripes import export_mstripes_bundle
from src.matching.catalogue import TigerCatalogue
from src.occupancy.home_range import compute_occupancy, compute_overlap_pct
from src.occupancy.map_export import export_csv_report, export_geojson_bundle, generate_occupancy_map
from src.pipeline.blank_filter import classify_blank, estimate_time_saved_sec, quarantine_blank
from src.pipeline.detect_crop import detect_and_crop
from src.pipeline.ingest import ingest_directory, load_station_registry, load_station_tags
from src.ml.model_registry import ml_available, warmup_models


def _assign_synthetic_station(
    path: Path,
    registry_map: dict[str, tuple[float, float, str]],
) -> tuple[str | None, float | None, float | None]:
    """Spread images without metadata across reserve stations for occupancy demo."""
    if not registry_map or not app_config.demo.synthetic_stations:
        return None, None, None
    import hashlib

    stations = sorted(registry_map.keys())
    digest = hashlib.md5(str(path.name).encode()).hexdigest()
    station_id = stations[int(digest, 16) % len(stations)]
    lat, lon, _zone = registry_map[station_id]
    return station_id, lat, lon


@dataclass
class PipelineReport:
    run_id: int
    total_frames: int = 0
    blank_removed: int = 0
    tiger_detected: int = 0
    new_individuals: int = 0
    auto_matched: int = 0
    pending_review: int = 0
    bytes_quarantined: int = 0
    estimated_time_saved_sec: float = 0.0
    occupancy_count: int = 0
    alerts_raised: int = 0
    anomalies_raised: int = 0
    exports: dict = field(default_factory=dict)


class TigerTrackingPipeline:
    def __init__(self):
        init_db()
        self.dirs = ensure_data_dirs()
        self.session = get_session()
        self.repo = Repository(self.session)
        self.catalogue = TigerCatalogue(self.repo)
        self.alert_engine = AlertEngine(self.repo)
        self.anomaly_engine = AnomalyEngine(self.repo)

    def run(
        self,
        input_dir: Path,
        station_registry: Path | None = None,
        recursive: bool = True,
    ) -> PipelineReport:
        input_dir = Path(input_dir)
        run = self.repo.create_run(str(input_dir))
        report = PipelineReport(run_id=run.id)

        if ml_available():
            warmup_models()

        ingested = ingest_directory(input_dir, station_registry, recursive)
        report.total_frames = len(ingested)
        station_registry_map = (
            load_station_registry(station_registry) if station_registry else {}
        )
        # Seed every registered station up front so zone/waterhole/sensitive tags
        # exist for alerting even at stations that recorded nothing this run.
        for sid, attrs in load_station_tags(station_registry).items():
            self.repo.upsert_station(
                sid,
                attrs["latitude"],
                attrs["longitude"],
                name=attrs["name"],
                zone=attrs["zone"],
                is_village_adjacent=attrs["is_village_adjacent"],
                is_sensitive=attrs["is_sensitive"],
                is_waterhole=attrs["is_waterhole"],
                range_name=attrs["range_name"],
                beat=attrs["beat"],
                compartment=attrs["compartment"],
            )

        for item in ingested:
            station_id = item.station_id
            latitude = item.latitude
            longitude = item.longitude
            if station_id and station_id in station_registry_map and (
                latitude is None or longitude is None
            ):
                latitude, longitude, zone = station_registry_map[station_id]
                self.repo.upsert_station(station_id, latitude, longitude, zone=zone)

            if latitude is None or longitude is None:
                syn_id, syn_lat, syn_lon = _assign_synthetic_station(
                    item.path, station_registry_map
                )
                if syn_id and syn_lat and syn_lon:
                    station_id = station_id or syn_id
                    latitude = syn_lat
                    longitude = syn_lon
                    _lat, _lon, zone = station_registry_map[syn_id]
                    self.repo.upsert_station(syn_id, _lat, _lon, zone=zone)

            img_record = self.repo.add_image(
                run_id=run.id,
                original_path=str(item.path),
                working_path=str(item.path),
                station_id=station_id,
                captured_at=item.captured_at,
                latitude=latitude,
                longitude=longitude,
                file_size_bytes=item.file_size_bytes,
                status="processing",
            )

            if station_id and latitude and longitude:
                self.repo.upsert_station(
                    station_id,
                    latitude,
                    longitude,
                )

            blank_result = classify_blank(item.path)
            if blank_result.is_blank:
                quarantine_path = quarantine_blank(
                    item.path,
                    self.dirs["quarantine_blank"],
                    run.id,
                )
                report.blank_removed += 1
                report.bytes_quarantined += item.file_size_bytes
                self.repo.update_image(
                    img_record,
                    is_blank=True,
                    blank_confidence=blank_result.confidence,
                    quarantine_path=str(quarantine_path),
                    working_path=None,
                    status="quarantined",
                )
                continue

            self.repo.update_image(
                img_record,
                is_blank=False,
                blank_confidence=blank_result.confidence,
                status="retained",
            )

            detection = detect_and_crop(item.path, self.dirs["flanks"], img_record.id)
            if not detection.has_tiger or not detection.flank_path:
                self.repo.update_image(
                    img_record,
                    has_tiger=False,
                    tiger_confidence=detection.confidence,
                    status="no_tiger",
                )
                continue

            report.tiger_detected += 1
            frame_anomaly = analyse_frame(
                detection.flank_path, detection.bbox, detection.animal_count
            )
            self.repo.update_image(
                img_record,
                has_tiger=True,
                tiger_confidence=detection.confidence,
                flank_path=str(detection.flank_path),
                status="matched",
                bbox_json=json.dumps(list(detection.bbox)) if detection.bbox else None,
                animal_count=detection.animal_count,
                anomaly_json=frame_anomaly.to_json(),
            )

            decision = self.catalogue.find_best_match(detection.flank_path)

            if decision.action == "auto_match" and decision.tiger_id:
                report.auto_matched += 1
                self.catalogue.add_representative_if_needed(
                    decision.tiger_id, img_record.id, detection.flank_path
                )
                self.catalogue.record_sighting(
                    tiger_id=decision.tiger_id,
                    image_id=img_record.id,
                    run_id=run.id,
                    station_id=station_id,
                    captured_at=item.captured_at,
                    latitude=latitude,
                    longitude=longitude,
                    confidence=decision.confidence,
                    match_type="auto",
                )
            elif decision.action == "review":
                report.pending_review += 1
                self.repo.add_review(
                    img_record.id,
                    decision.tiger_id,
                    decision.confidence,
                )
            else:
                report.new_individuals += 1
                tiger_id = self.catalogue.enroll_tiger(img_record.id, detection.flank_path)
                self.catalogue.record_sighting(
                    tiger_id=tiger_id,
                    image_id=img_record.id,
                    run_id=run.id,
                    station_id=station_id,
                    captured_at=item.captured_at,
                    latitude=latitude,
                    longitude=longitude,
                    confidence=decision.confidence,
                    match_type="enroll",
                )

        report.estimated_time_saved_sec = estimate_time_saved_sec(report.blank_removed)

        occupancy_snapshots = self._compute_occupancy(run.id)
        report.occupancy_count = len(occupancy_snapshots)

        alerts = self.alert_engine.run(run.id, occupancy_snapshots)
        anomalies = self.anomaly_engine.run(run.id)
        report.anomalies_raised = len(anomalies)
        report.alerts_raised = len(alerts) + len(anomalies)

        report.exports = self._export_results(run.id, occupancy_snapshots)

        self.repo.complete_run(
            run,
            total_frames=report.total_frames,
            blank_removed=report.blank_removed,
            tiger_detected=report.tiger_detected,
            new_individuals=report.new_individuals,
            auto_matched=report.auto_matched,
            pending_review=report.pending_review,
            bytes_quarantined=report.bytes_quarantined,
            estimated_time_saved_sec=report.estimated_time_saved_sec,
        )

        return report

    def _compute_occupancy(self, run_id: int) -> list:
        snapshots = []
        tigers = self.repo.list_tigers()
        cfg = app_config.occupancy

        for tiger in tigers:
            sightings = self.repo.get_sightings_for_tiger(tiger.id)
            result = compute_occupancy(
                sightings,
                min_points=cfg.min_points_for_range,
                method=cfg.home_range_method,
            )
            if not result:
                continue

            snap = self.repo.add_occupancy_snapshot(
                run_id=run_id,
                tiger_id=tiger.id,
                centroid_lat=result.centroid_lat,
                centroid_lon=result.centroid_lon,
                area_sq_km=result.area_sq_km,
                capture_count=result.capture_count,
                station_ids=",".join(result.station_ids),
                home_range_geojson=result.home_range_geojson,
            )
            snapshots.append(snap)
        return snapshots

    def _export_results(self, run_id: int, snapshots: list) -> dict:
        exports_dir = self.dirs["exports"] / f"run_{run_id}"
        exports_dir.mkdir(parents=True, exist_ok=True)

        occupancy_data = []
        for snap in snapshots:
            tiger = self.repo.get_tiger(snap.tiger_id)
            occupancy_data.append({
                "tiger_id": snap.tiger_id,
                "tiger_code": tiger.tiger_code if tiger else "?",
                "centroid_lat": snap.centroid_lat,
                "centroid_lon": snap.centroid_lon,
                "area_sq_km": snap.area_sq_km,
                "capture_count": snap.capture_count,
                "station_ids": snap.station_ids.split(",") if snap.station_ids else [],
                "home_range_geojson": snap.home_range_geojson,
            })

        overlap_pairs = []
        for i, a in enumerate(occupancy_data):
            for b in occupancy_data[i + 1:]:
                gj_a = json.loads(a["home_range_geojson"]) if a.get("home_range_geojson") else None
                gj_b = json.loads(b["home_range_geojson"]) if b.get("home_range_geojson") else None
                overlap = compute_overlap_pct(gj_a, gj_b)
                if overlap > 0:
                    overlap_pairs.append({
                        "tiger_a": a["tiger_code"],
                        "tiger_b": b["tiger_code"],
                        "tiger_a_lat": a["centroid_lat"],
                        "tiger_a_lon": a["centroid_lon"],
                        "tiger_b_lat": b["centroid_lat"],
                        "tiger_b_lon": b["centroid_lon"],
                        "overlap_pct": overlap,
                    })

        map_path = generate_occupancy_map(
            occupancy_data,
            overlap_pairs,
            exports_dir / "occupancy_map.html",
            stations=self.repo.get_stations(),
        )
        geojson_path = export_geojson_bundle(occupancy_data, exports_dir / "home_ranges.geojson")
        csv_path = export_csv_report(occupancy_data, exports_dir / "occupancy_report.csv")

        overlap_path = exports_dir / "territorial_overlaps.json"
        with open(overlap_path, "w") as f:
            json.dump(overlap_pairs, f, indent=2)

        mstripes = export_mstripes_bundle(self.repo, run_id, exports_dir / "mstripes")

        return {
            "map": str(map_path),
            "geojson": str(geojson_path),
            "csv": str(csv_path),
            "overlaps": str(overlap_path),
            "mstripes": str(mstripes.directory),
        }

    def close(self):
        self.session.close()
