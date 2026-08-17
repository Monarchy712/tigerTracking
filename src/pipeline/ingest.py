"""Image ingestion: walk directories, extract metadata."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class IngestedImage:
    path: Path
    station_id: str | None
    captured_at: datetime | None
    latitude: float | None
    longitude: float | None
    file_size_bytes: int


def _parse_gps(info: dict) -> tuple[float | None, float | None]:
    def to_degrees(value) -> float:
        d, m, s = value
        return float(d) + float(m) / 60.0 + float(s) / 3600.0

    gps_info = {}
    for key, val in info.items():
        decoded = GPSTAGS.get(key, key)
        gps_info[decoded] = val

    lat = gps_info.get("GPSLatitude")
    lon = gps_info.get("GPSLongitude")
    if not lat or not lon:
        return None, None

    lat_val = to_degrees(lat)
    lon_val = to_degrees(lon)
    if gps_info.get("GPSLatitudeRef") == "S":
        lat_val = -lat_val
    if gps_info.get("GPSLongitudeRef") == "W":
        lon_val = -lon_val
    return lat_val, lon_val


def _extract_exif(path: Path) -> tuple[datetime | None, float | None, float | None]:
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return None, None, None

            captured_at = None
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "DateTimeOriginal" and isinstance(value, str):
                    try:
                        captured_at = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                    except ValueError:
                        pass

            gps_ifd = exif.get_ifd(0x8825) if hasattr(exif, "get_ifd") else None
            if gps_ifd:
                lat, lon = _parse_gps(gps_ifd)
                return captured_at, lat, lon
            return captured_at, None, None
    except Exception:
        return None, None, None


def _normalize_station_id(raw: str | None) -> str | None:
    """Extract canonical station id like CAM01 from noisy filename tokens."""
    if not raw:
        return None
    cam_match = re.search(r"(CAM\d+)", raw, re.I)
    if cam_match:
        return cam_match.group(1).upper()
    return raw.upper()


def _match_known_station(name: str, known_ids) -> str | None:
    """Find a registered station id inside a filename, longest id first.

    Reserves use their own prefixes (CAM01, PEN03, ...), so matching against the
    registry is more reliable than any single hard-coded pattern.
    """
    for sid in sorted(known_ids, key=len, reverse=True):
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(sid)}(?![A-Za-z0-9])", name, re.I):
            return sid
    return None


def _parse_filename_metadata(
    path: Path, known_station_ids=None
) -> tuple[str | None, datetime | None]:
    """Parse station and timestamp from common camera-trap naming patterns."""
    name = path.stem
    station_id = None
    captured_at = None

    if known_station_ids:
        station_id = _match_known_station(name, known_station_ids)

    if not station_id:
        station_match = re.search(r"(?:station|cam|st)[_-]?(CAM\d+)", name, re.I)
        if station_match:
            station_id = station_match.group(1).upper()
        else:
            cam_only = re.search(r"(CAM\d+)", name, re.I)
            if cam_only:
                station_id = cam_only.group(1).upper()

    ts_match = re.search(r"(\d{8})[_-]?(\d{6})?", name)
    if ts_match:
        date_part = ts_match.group(1)
        time_part = ts_match.group(2) or "000000"
        try:
            captured_at = datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")
        except ValueError:
            pass

    return station_id, captured_at


def load_station_registry(registry_path: Path | None) -> dict[str, tuple[float, float, str]]:
    """Load station_id -> (lat, lon, zone) from CSV."""
    if not registry_path or not registry_path.exists():
        return {}

    registry: dict[str, tuple[float, float, str]] = {}
    with open(registry_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("station_id") or row.get("id")
            if not sid:
                continue
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            zone = row.get("zone", "core")
            registry[sid] = (lat, lon, zone)
    return registry


_TRUTHY = {"1", "true", "yes", "y", "t"}


def load_station_tags(registry_path: Path | None) -> dict[str, dict]:
    """Load the full station row, including optional `is_sensitive` / `is_waterhole` tags.

    `load_station_registry` stays coordinate-only for existing callers; the tags
    drive village-proximity and waterhole-stress alerting.
    """
    if not registry_path or not registry_path.exists():
        return {}

    tags: dict[str, dict] = {}
    with open(registry_path, newline="") as f:
        for row in csv.DictReader(f):
            sid = row.get("station_id") or row.get("id")
            if not sid:
                continue
            zone = row.get("zone", "core")
            tags[sid] = {
                "name": row.get("name") or None,
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "zone": zone,
                "is_village_adjacent": zone == "village_adjacent",
                "is_sensitive": (row.get("is_sensitive") or "").strip().lower() in _TRUTHY
                or zone == "village_adjacent",
                "is_waterhole": (row.get("is_waterhole") or "").strip().lower() in _TRUTHY,
                # Optional administrative units — only the M-STrIPES export needs them.
                "range_name": (row.get("range_name") or row.get("range") or "").strip() or None,
                "beat": (row.get("beat") or "").strip() or None,
                "compartment": (row.get("compartment") or "").strip() or None,
            }
    return tags


def discover_images(input_dir: Path, recursive: bool = True) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    paths: list[Path] = []
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    for p in sorted(iterator):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p)
    return paths


def ingest_image(path: Path, station_registry: dict | None = None) -> IngestedImage:
    station_registry = station_registry or {}
    captured_at, lat, lon = _extract_exif(path)
    station_id, fn_time = _parse_filename_metadata(path, station_registry.keys())

    if fn_time and not captured_at:
        captured_at = fn_time

    if station_id and station_id not in station_registry:
        station_id = _normalize_station_id(station_id)

    if station_id and station_id in station_registry and (lat is None or lon is None):
        lat, lon, _ = station_registry[station_id]

    return IngestedImage(
        path=path,
        station_id=station_id,
        captured_at=captured_at,
        latitude=lat,
        longitude=lon,
        file_size_bytes=path.stat().st_size,
    )


def ingest_directory(
    input_dir: Path,
    station_registry_path: Path | None = None,
    recursive: bool = True,
) -> list[IngestedImage]:
    registry = load_station_registry(station_registry_path)
    return [ingest_image(p, registry) for p in discover_images(input_dir, recursive)]
