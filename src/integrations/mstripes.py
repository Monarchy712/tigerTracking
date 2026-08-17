"""M-STrIPES-aligned exports for the forest department.

M-STrIPES (Monitoring System for Tigers - Intensive Protection and Ecological
Status, NTCA/WII) has no published import schema, so nothing here claims to be
an officially certified file. What these exporters produce is the *Phase IV
camera-trap data layout* the ecological module is built around:

  1. a camera deployment sheet (one row per trap, with effort dates),
  2. a capture sheet (one row per identified capture),
  3. the capture-history and trap-effort matrices that density estimation
     (SECR / CAPTURE / MARK) consumes,
  4. patrol waypoints as GPX, loadable on a ranger handset.

Column names live in `DEPLOYMENT_COLUMNS` / `CAPTURE_COLUMNS` so they can be
remapped in one place once a real departmental sheet is available.

Fields the pipeline cannot derive are written as `UNKNOWN_VALUE` rather than
guessed — see `README_MSTRIPES.txt` in the export directory. Caveats are kept
in that sidecar and in `manifest.json` so the CSVs stay machine-importable.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from xml.sax.saxutils import escape

from src.config import app_config

SPECIES_NAME = "Panthera tigris"
UNKNOWN_VALUE = "U"
NOT_RECORDED = ""

DEPLOYMENT_COLUMNS = [
    "TrapID",
    "TrapName",
    "Range",
    "Beat",
    "Compartment",
    "Latitude",
    "Longitude",
    "Zone",
    "DatePlaced",
    "DateRetrieved",
    "TrapNights",
    "CameraMake",
    "IsWaterhole",
    "IsSensitive",
]

CAPTURE_COLUMNS = [
    "TrapID",
    "PhotoID",
    "CaptureDateTime",
    "Species",
    "IndividualID",
    "Flank",
    "Sex",
    "AgeClass",
    "AnimalCount",
    "Latitude",
    "Longitude",
    "MatchConfidence",
    "MatchType",
    "PhotoPath",
]

README_TEXT = """M-STrIPES-aligned export
========================

Produced by the Tiger Tracking pipeline for {reserve}.
Run ID: {run_id}
Generated: {generated}

WHAT THIS IS
------------
These files follow the Phase IV camera-trap data layout used by the M-STrIPES
ecological module. They are NOT an officially certified M-STrIPES import
bundle - no public import schema exists. Column names can be remapped in
src/integrations/mstripes.py (DEPLOYMENT_COLUMNS / CAPTURE_COLUMNS) once a real
departmental sheet is available.

FILES
-----
camera_deployment.csv  One row per camera trap station, with effort dates.
capture_records.csv    One row per identified tiger capture.
capture_history.csv    Individuals x sampling occasions, 0/1. Input format for
                       SECR / CAPTURE / MARK density estimation.
trap_effort.csv        Traps x sampling occasions, 1 = camera active that day.
population_summary.csv Minimum count, trap nights, recapture rate and a
                       Chapman-corrected Lincoln-Petersen estimate.
alerts.gpx             Deviation and anomaly alerts as GPX waypoints, plus
                       waterhole and village-interface stations.
manifest.json          File inventory, counts and the caveats below.

DEFINITIONS USED
----------------
Sampling occasion   One calendar day, from the first to the last capture date
                    in the database (inclusive).
Trap active         A station is counted as active on a day if any frame from
                    that station carries that date - including frames that were
                    quarantined as blank, since a blank frame still proves the
                    camera was running.
Trap night          One active station-day.
Capture history     Scoped to the whole database, not a single run: capture-
                    recapture analysis needs the full survey block. The capture,
                    deployment and effort sheets share that scope; only
                    alerts.gpx is filtered to this run.

FIELDS NOT DERIVED BY THE PIPELINE
----------------------------------
Written as "{unknown}" (unknown) or left blank rather than guessed:

  Flank (L/R)   The detector crops a flank but does not label which side.
                Stripe matching is side-sensitive, so a guessed value would
                corrupt any downstream individual analysis.
  Sex           Not inferrable from the current model.
  AgeClass      Not inferrable from the current model.
  CameraMake    Not recorded in the station registry.
  Range/Beat/   Read from the station registry CSV when those columns are
  Compartment   present; blank otherwise.

Any density figure derived from this bundle inherits those gaps and should be
treated as provisional until a reviewer fills them in.
"""


@dataclass
class MStripesExport:
    """Result of one bundle export."""

    directory: Path
    files: dict[str, Path] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)


# --- helpers ---------------------------------------------------------------


def _as_date(value: datetime | None) -> date | None:
    return value.date() if isinstance(value, datetime) else None


def _occasion_days(bounds: tuple[datetime | None, datetime | None]) -> list[date]:
    """Every calendar day between the first and last capture, inclusive."""
    start, end = _as_date(bounds[0]), _as_date(bounds[1])
    if not start or not end or end < start:
        return []
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _station_active_days(repo) -> dict[str, set[date]]:
    """Days each station produced a frame — blanks included, they prove uptime."""
    active: dict[str, set[date]] = {}
    for station_id, captured_at in repo.station_frame_days():
        day = _as_date(captured_at)
        if not station_id or not day:
            continue
        active.setdefault(station_id, set()).add(day)
    return active


def _truthy_str(value) -> str:
    return "true" if bool(value) else "false"


# --- A. deployment + capture sheets ---------------------------------------


def export_camera_deployment(repo, output_path: Path) -> Path:
    """Camera trap deployment sheet — one row per station."""
    active = _station_active_days(repo)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DEPLOYMENT_COLUMNS)
        writer.writeheader()
        for station in sorted(repo.get_stations(), key=lambda s: s.station_id):
            days = sorted(active.get(station.station_id, set()))
            writer.writerow({
                "TrapID": station.station_id,
                "TrapName": station.name or NOT_RECORDED,
                "Range": getattr(station, "range_name", None) or NOT_RECORDED,
                "Beat": getattr(station, "beat", None) or NOT_RECORDED,
                "Compartment": getattr(station, "compartment", None) or NOT_RECORDED,
                "Latitude": station.latitude,
                "Longitude": station.longitude,
                "Zone": station.zone,
                "DatePlaced": days[0].isoformat() if days else NOT_RECORDED,
                "DateRetrieved": days[-1].isoformat() if days else NOT_RECORDED,
                "TrapNights": len(days),
                "CameraMake": NOT_RECORDED,
                "IsWaterhole": _truthy_str(station.is_waterhole),
                "IsSensitive": _truthy_str(station.is_sensitive),
            })
    return output_path


def export_capture_records(repo, output_path: Path, run_id: int | None = None) -> Path:
    """Capture sheet — one row per identified tiger capture."""
    sightings = (
        repo.get_sightings_for_run(run_id) if run_id is not None else repo.get_all_sightings()
    )
    tiger_codes = {t.id: t.tiger_code for t in repo.list_tigers()}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CAPTURE_COLUMNS)
        writer.writeheader()
        for s in sorted(sightings, key=lambda x: (x.captured_at or datetime.min, x.id)):
            image = repo.get_image(s.image_id)
            writer.writerow({
                "TrapID": s.station_id or NOT_RECORDED,
                "PhotoID": s.image_id,
                "CaptureDateTime": s.captured_at.isoformat(sep=" ") if s.captured_at else NOT_RECORDED,
                "Species": SPECIES_NAME,
                "IndividualID": tiger_codes.get(s.tiger_id, f"#{s.tiger_id}"),
                "Flank": UNKNOWN_VALUE,
                "Sex": UNKNOWN_VALUE,
                "AgeClass": UNKNOWN_VALUE,
                "AnimalCount": getattr(image, "animal_count", 0) or 1,
                "Latitude": s.latitude if s.latitude is not None else NOT_RECORDED,
                "Longitude": s.longitude if s.longitude is not None else NOT_RECORDED,
                "MatchConfidence": round(s.match_confidence, 4) if s.match_confidence else NOT_RECORDED,
                "MatchType": s.match_type,
                "PhotoPath": image.original_path if image else NOT_RECORDED,
            })
    return output_path


# --- B. capture history + effort matrices ---------------------------------


def build_capture_history(repo) -> tuple[list[date], dict[str, set[date]]]:
    """Per-individual capture days across the whole survey block."""
    occasions = _occasion_days(repo.sighting_date_bounds())
    tiger_codes = {t.id: t.tiger_code for t in repo.list_tigers()}

    history: dict[str, set[date]] = {}
    for s in repo.get_all_sightings():
        day = _as_date(s.captured_at)
        if not day:
            continue
        code = tiger_codes.get(s.tiger_id, f"#{s.tiger_id}")
        history.setdefault(code, set()).add(day)
    return occasions, history


def export_capture_history(repo, output_path: Path) -> Path:
    """Individuals x occasions detection matrix (0/1) for SECR / MARK."""
    occasions, history = build_capture_history(repo)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["IndividualID"] + [d.isoformat() for d in occasions])
        for code in sorted(history):
            days = history[code]
            writer.writerow([code] + [1 if d in days else 0 for d in occasions])
    return output_path


def export_trap_effort(repo, output_path: Path) -> Path:
    """Traps x occasions effort matrix (1 = camera active that day)."""
    occasions = _occasion_days(repo.sighting_date_bounds())
    active = _station_active_days(repo)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["TrapID", "Latitude", "Longitude"] + [d.isoformat() for d in occasions])
        for station in sorted(repo.get_stations(), key=lambda s: s.station_id):
            days = active.get(station.station_id, set())
            writer.writerow(
                [station.station_id, station.latitude, station.longitude]
                + [1 if d in days else 0 for d in occasions]
            )
    return output_path


def _chapman_estimate(n1: int, n2: int, m2: int) -> float | None:
    """Chapman-corrected Lincoln-Petersen — less biased than n1*n2/m2 for small m2."""
    if n1 == 0 or n2 == 0:
        return None
    return ((n1 + 1) * (n2 + 1) / (m2 + 1)) - 1


def compute_population_summary(repo) -> dict:
    """Minimum count, effort and a two-half capture-recapture estimate."""
    occasions, history = build_capture_history(repo)
    active = _station_active_days(repo)
    trap_nights = sum(len(days) for days in active.values())
    captures = sum(len(days) for days in history.values())

    summary = {
        "individuals_min_count": len(history),
        "sampling_occasions": len(occasions),
        "first_occasion": occasions[0].isoformat() if occasions else "",
        "last_occasion": occasions[-1].isoformat() if occasions else "",
        "active_traps": len([s for s, d in active.items() if d]),
        "trap_nights": trap_nights,
        "capture_events": captures,
        "captures_per_100_trap_nights": (
            round(100.0 * captures / trap_nights, 2) if trap_nights else 0.0
        ),
        "estimator": "Chapman-corrected Lincoln-Petersen, survey split into two equal halves",
    }

    if len(occasions) >= 2 and history:
        midpoint = occasions[len(occasions) // 2]
        first_half = {c for c, days in history.items() if any(d < midpoint for d in days)}
        second_half = {c for c, days in history.items() if any(d >= midpoint for d in days)}
        recaptured = first_half & second_half
        estimate = _chapman_estimate(len(first_half), len(second_half), len(recaptured))
        summary.update({
            "period_1_individuals": len(first_half),
            "period_2_individuals": len(second_half),
            "recaptured_individuals": len(recaptured),
            "recapture_rate": (
                round(len(recaptured) / len(first_half), 3) if first_half else 0.0
            ),
            "population_estimate": round(estimate, 1) if estimate is not None else "",
        })
    return summary


def export_population_summary(repo, output_path: Path) -> Path:
    summary = compute_population_summary(repo)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])
    return output_path


# --- C. patrol waypoints ---------------------------------------------------


def _waypoint(lat: float, lon: float, name: str, desc: str, symbol: str, when: datetime | None) -> str:
    time_tag = f"\n    <time>{when.isoformat()}Z</time>" if when else ""
    return (
        f'  <wpt lat="{lat:.6f}" lon="{lon:.6f}">\n'
        f"    <name>{escape(name)}</name>\n"
        f"    <desc>{escape(desc)}</desc>\n"
        f"    <sym>{escape(symbol)}</sym>\n"
        f"    <type>{escape(symbol)}</type>{time_tag}\n"
        f"  </wpt>"
    )


def _alert_location(repo, alert) -> tuple[float, float] | None:
    """Alert coordinates — evidence centroid first, else the tiger's last fix."""
    if alert.evidence_json:
        try:
            evidence = json.loads(alert.evidence_json)
        except (json.JSONDecodeError, TypeError):
            evidence = {}
        centroid = evidence.get("new_centroid") or evidence.get("centroid")
        if isinstance(centroid, (list, tuple)) and len(centroid) == 2:
            if centroid[0] is not None and centroid[1] is not None:
                return float(centroid[0]), float(centroid[1])
        station_id = evidence.get("station_id")
        if station_id:
            station = repo.get_station(station_id)
            if station:
                return station.latitude, station.longitude

    if alert.tiger_id:
        last = repo.get_last_sighting(alert.tiger_id)
        if last and last.latitude is not None and last.longitude is not None:
            return last.latitude, last.longitude
    return None


def export_patrol_gpx(repo, output_path: Path, run_id: int | None = None) -> Path:
    """Alerts and priority stations as GPX waypoints for a ranger handset."""
    waypoints: list[str] = []

    for alert in repo.get_alerts(run_id=run_id):
        location = _alert_location(repo, alert)
        if not location:
            continue
        lat, lon = location
        waypoints.append(
            _waypoint(
                lat,
                lon,
                f"[{alert.severity.upper()}] {alert.title}",
                f"{alert.alert_type} | confidence {alert.confidence:.2f}\n{alert.description}",
                "Danger Area" if alert.severity == "high" else "Flag",
                alert.created_at,
            )
        )

    for station in repo.get_stations():
        if station.is_waterhole:
            waypoints.append(
                _waypoint(
                    station.latitude,
                    station.longitude,
                    f"Waterhole {station.station_id}",
                    f"{station.name or station.station_id} — perennial water source",
                    "Water Source",
                    None,
                )
            )
        if station.is_sensitive:
            waypoints.append(
                _waypoint(
                    station.latitude,
                    station.longitude,
                    f"Village interface {station.station_id}",
                    f"{station.name or station.station_id} — livestock/village interface",
                    "Residence",
                    None,
                )
            )

    reserve = app_config.reserve.name
    body = "\n".join(waypoints)
    gpx = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx version="1.1" creator="Tiger Tracking System" '
        'xmlns="http://www.topografix.com/GPX/1/1">\n'
        "  <metadata>\n"
        f"    <name>{escape(reserve)} — patrol waypoints</name>\n"
        f"    <time>{datetime.utcnow().isoformat()}Z</time>\n"
        "  </metadata>\n"
        f"{body}\n"
        "</gpx>\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(gpx)
    return output_path


# --- bundle ----------------------------------------------------------------


def export_mstripes_bundle(repo, run_id: int, output_dir: Path) -> MStripesExport:
    """Write the full M-STrIPES-aligned bundle for one run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "deployment": export_camera_deployment(repo, output_dir / "camera_deployment.csv"),
        "captures": export_capture_records(repo, output_dir / "capture_records.csv"),
        "capture_history": export_capture_history(repo, output_dir / "capture_history.csv"),
        "trap_effort": export_trap_effort(repo, output_dir / "trap_effort.csv"),
        "population_summary": export_population_summary(repo, output_dir / "population_summary.csv"),
        "gpx": export_patrol_gpx(repo, output_dir / "alerts.gpx", run_id=run_id),
    }

    generated = datetime.utcnow().isoformat(sep=" ", timespec="seconds")
    readme = output_dir / "README_MSTRIPES.txt"
    readme.write_text(
        README_TEXT.format(
            reserve=app_config.reserve.name,
            run_id=run_id,
            generated=generated,
            unknown=UNKNOWN_VALUE,
        )
    )
    files["readme"] = readme

    stats = compute_population_summary(repo)
    manifest = output_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "reserve": app_config.reserve.name,
                "run_id": run_id,
                "generated_utc": generated,
                "schema": "phase-iv-aligned",
                "certified_mstripes_import": False,
                "files": {k: v.name for k, v in files.items()},
                "summary": stats,
                "unresolved_fields": ["Flank", "Sex", "AgeClass", "CameraMake"],
            },
            indent=2,
        )
    )
    files["manifest"] = manifest

    return MStripesExport(directory=output_dir, files=files, stats=stats)
