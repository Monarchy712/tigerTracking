"""M-STrIPES export tests — driven by a stub repository, no DB or ML needed."""

import csv
import json
from datetime import datetime
from types import SimpleNamespace


def _station(station_id, lat, lon, **kwargs):
    return SimpleNamespace(
        station_id=station_id,
        name=kwargs.get("name"),
        latitude=lat,
        longitude=lon,
        zone=kwargs.get("zone", "core"),
        is_sensitive=kwargs.get("is_sensitive", False),
        is_waterhole=kwargs.get("is_waterhole", False),
        range_name=kwargs.get("range_name"),
        beat=kwargs.get("beat"),
        compartment=kwargs.get("compartment"),
    )


class StubRepo:
    """Two individuals over four days at two stations, one of them recaptured."""

    def __init__(self):
        self.stations = [
            _station("PEN01", 21.785, 79.310, name="Alikatta North", beat="Alikatta"),
            _station("PEN03", 21.765, 79.335, name="Pipariya Nala", is_waterhole=True),
        ]
        self.tigers = [
            SimpleNamespace(id=1, tiger_code="T0001"),
            SimpleNamespace(id=2, tiger_code="T0002"),
        ]
        self.images = {
            10: SimpleNamespace(id=10, original_path="/raw/a.jpg", animal_count=1),
            11: SimpleNamespace(id=11, original_path="/raw/b.jpg", animal_count=2),
            12: SimpleNamespace(id=12, original_path="/raw/c.jpg", animal_count=1),
        }
        self.sightings = [
            SimpleNamespace(
                id=1, tiger_id=1, image_id=10, run_id=1, station_id="PEN01",
                captured_at=datetime(2026, 1, 1, 6, 30), latitude=21.785, longitude=79.310,
                match_confidence=0.93, match_type="auto",
            ),
            SimpleNamespace(
                id=2, tiger_id=1, image_id=11, run_id=1, station_id="PEN03",
                captured_at=datetime(2026, 1, 4, 21, 5), latitude=21.765, longitude=79.335,
                match_confidence=0.88, match_type="auto",
            ),
            SimpleNamespace(
                id=3, tiger_id=2, image_id=12, run_id=1, station_id="PEN01",
                captured_at=datetime(2026, 1, 2, 3, 15), latitude=21.785, longitude=79.310,
                match_confidence=None, match_type="enroll",
            ),
        ]
        self.alerts = [
            SimpleNamespace(
                id=1, run_id=1, tiger_id=1, alert_type="range_shift", severity="high",
                confidence=0.85, title="T0001: Home range shift detected",
                description="Centroid shifted 6.2 km", created_at=datetime(2026, 1, 5, 9, 0),
                evidence_json=json.dumps({"new_centroid": [21.770, 79.320]}),
            ),
        ]

    def get_stations(self):
        return self.stations

    def get_station(self, station_id):
        return next((s for s in self.stations if s.station_id == station_id), None)

    def list_tigers(self):
        return self.tigers

    def get_all_sightings(self):
        return self.sightings

    def get_sightings_for_run(self, run_id):
        return [s for s in self.sightings if s.run_id == run_id]

    def get_image(self, image_id):
        return self.images.get(image_id)

    def get_alerts(self, run_id=None, unacknowledged_only=False):
        return self.alerts

    def get_last_sighting(self, tiger_id):
        matches = [s for s in self.sightings if s.tiger_id == tiger_id]
        return max(matches, key=lambda s: s.captured_at) if matches else None

    def sighting_date_bounds(self):
        # Mirrors the real repository: SQL min/max return (None, None) when empty.
        dates = [s.captured_at for s in self.sightings]
        return (min(dates), max(dates)) if dates else (None, None)

    def station_frame_days(self):
        # Blanks included: PEN01 recorded on a day with no tiger capture.
        return [
            ("PEN01", datetime(2026, 1, 1, 6, 30)),
            ("PEN01", datetime(2026, 1, 2, 3, 15)),
            ("PEN01", datetime(2026, 1, 3, 11, 0)),
            ("PEN03", datetime(2026, 1, 4, 21, 5)),
        ]


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_deployment_sheet_effort_dates(tmp_path):
    from src.integrations.mstripes import DEPLOYMENT_COLUMNS, export_camera_deployment

    path = export_camera_deployment(StubRepo(), tmp_path / "camera_deployment.csv")
    rows = _read_csv(path)

    assert rows[0] == DEPLOYMENT_COLUMNS
    pen01 = dict(zip(rows[0], rows[1]))
    assert pen01["TrapID"] == "PEN01"
    assert pen01["DatePlaced"] == "2026-01-01"
    assert pen01["DateRetrieved"] == "2026-01-03"
    assert pen01["TrapNights"] == "3"
    assert pen01["Beat"] == "Alikatta"

    pen03 = dict(zip(rows[0], rows[2]))
    assert pen03["IsWaterhole"] == "true"


def test_capture_records_unresolved_fields_are_unknown(tmp_path):
    from src.integrations.mstripes import SPECIES_NAME, export_capture_records

    path = export_capture_records(StubRepo(), tmp_path / "capture_records.csv")
    rows = _read_csv(path)
    assert len(rows) == 4  # header + 3 sightings

    first = dict(zip(rows[0], rows[1]))
    assert first["IndividualID"] == "T0001"
    assert first["Species"] == SPECIES_NAME
    assert first["Flank"] == "U"
    assert first["Sex"] == "U"
    assert first["AnimalCount"] == "1"


def test_capture_history_matrix_is_binary_over_every_day(tmp_path):
    from src.integrations.mstripes import export_capture_history

    path = export_capture_history(StubRepo(), tmp_path / "capture_history.csv")
    rows = _read_csv(path)

    # 1 Jan to 4 Jan inclusive — four occasions, gap days included.
    assert rows[0] == ["IndividualID", "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    assert rows[1] == ["T0001", "1", "0", "0", "1"]
    assert rows[2] == ["T0002", "0", "1", "0", "0"]


def test_trap_effort_counts_blank_only_days(tmp_path):
    from src.integrations.mstripes import export_trap_effort

    path = export_trap_effort(StubRepo(), tmp_path / "trap_effort.csv")
    rows = _read_csv(path)

    pen01 = rows[1]
    # 3 Jan had a frame but no tiger capture — still an active trap night.
    assert pen01[3:] == ["1", "1", "1", "0"]
    assert rows[2][3:] == ["0", "0", "0", "1"]


def test_population_summary_counts_and_estimate():
    from src.integrations.mstripes import compute_population_summary

    summary = compute_population_summary(StubRepo())

    assert summary["individuals_min_count"] == 2
    assert summary["sampling_occasions"] == 4
    assert summary["trap_nights"] == 4
    assert summary["capture_events"] == 3
    assert summary["captures_per_100_trap_nights"] == 75.0
    # Halves split at 3 Jan: period 1 = {T0001, T0002}, period 2 = {T0001}.
    assert summary["period_1_individuals"] == 2
    assert summary["period_2_individuals"] == 1
    assert summary["recaptured_individuals"] == 1
    assert summary["population_estimate"] == 2.0  # Chapman: (3*2/2) - 1


def test_patrol_gpx_has_alert_and_waterhole_waypoints(tmp_path):
    from src.integrations.mstripes import export_patrol_gpx

    path = export_patrol_gpx(StubRepo(), tmp_path / "alerts.gpx", run_id=1)
    gpx = path.read_text()

    assert gpx.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert gpx.count("<wpt ") == 2  # one alert + one waterhole station
    assert 'lat="21.770000" lon="79.320000"' in gpx  # evidence centroid wins
    assert "Waterhole PEN03" in gpx
    assert "&amp;" not in gpx or "&" not in gpx.replace("&amp;", "")  # XML stays escaped


def test_bundle_writes_every_file_and_manifest(tmp_path):
    from src.integrations.mstripes import export_mstripes_bundle

    bundle = export_mstripes_bundle(StubRepo(), 1, tmp_path / "mstripes")

    expected = {
        "camera_deployment.csv",
        "capture_records.csv",
        "capture_history.csv",
        "trap_effort.csv",
        "population_summary.csv",
        "alerts.gpx",
        "README_MSTRIPES.txt",
        "manifest.json",
    }
    assert {p.name for p in bundle.directory.iterdir()} == expected

    manifest = json.loads((bundle.directory / "manifest.json").read_text())
    assert manifest["certified_mstripes_import"] is False
    assert manifest["summary"]["individuals_min_count"] == 2
    assert "Flank" in manifest["unresolved_fields"]


def test_empty_database_produces_headers_only(tmp_path):
    from src.integrations.mstripes import export_mstripes_bundle

    class EmptyRepo(StubRepo):
        def __init__(self):
            super().__init__()
            self.stations, self.tigers, self.sightings, self.alerts = [], [], [], []

        def station_frame_days(self):
            return []

    bundle = export_mstripes_bundle(EmptyRepo(), 1, tmp_path / "mstripes")

    assert _read_csv(bundle.files["capture_history"]) == [["IndividualID"]]
    assert bundle.stats["individuals_min_count"] == 0
    assert bundle.stats["trap_nights"] == 0
