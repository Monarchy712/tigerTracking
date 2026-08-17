"""Deviation and trend alerting engine."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta

from src.config import app_config
from src.db.repository import Repository
from src.occupancy.home_range import _haversine_km


class AlertEngine:
    def __init__(self, repo: Repository):
        self.repo = repo
        self.cfg = app_config.alerts

    def run(self, run_id: int, current_occupancy: list) -> list:
        """Compare current run against historical data and raise alerts."""
        alerts = []
        tigers = self.repo.list_tigers()
        tiger_map = {t.id: t for t in tigers}

        for occ in current_occupancy:
            tiger = tiger_map.get(occ.tiger_id)
            if not tiger:
                continue

            prev = self.repo.get_latest_occupancy(occ.tiger_id, before_run_id=run_id)
            if prev and prev.centroid_lat and prev.centroid_lon and occ.centroid_lat and occ.centroid_lon:
                shift_km = _haversine_km(
                    prev.centroid_lat, prev.centroid_lon,
                    occ.centroid_lat, occ.centroid_lon,
                )
                area_delta = abs((occ.area_sq_km or 0) - (prev.area_sq_km or 0))

                zone = self._dominant_zone(occ.station_ids or "")
                threshold = self.cfg.buffer_range_shift_km if zone == "buffer" else self.cfg.core_range_shift_sqkm

                if shift_km > threshold or area_delta > threshold:
                    is_artifact = self._likely_survey_artifact(occ, prev)
                    alert = self.repo.add_alert(
                        run_id=run_id,
                        tiger_id=occ.tiger_id,
                        alert_type="range_shift",
                        severity="high" if shift_km > threshold * 1.5 else "medium",
                        confidence=0.85 if not is_artifact else 0.45,
                        title=f"{tiger.tiger_code}: Home range shift detected",
                        description=(
                            f"Centroid shifted {shift_km:.1f} km (threshold: {threshold} km). "
                            f"Area changed by {area_delta:.1f} sq km."
                        ),
                        evidence_json=self.repo.dump_evidence({
                            "shift_km": shift_km,
                            "area_delta_sq_km": area_delta,
                            "prev_centroid": [prev.centroid_lat, prev.centroid_lon],
                            "new_centroid": [occ.centroid_lat, occ.centroid_lon],
                            "threshold": threshold,
                            "zone": zone,
                        }),
                        is_survey_artifact=is_artifact,
                    )
                    alerts.append(alert)

            alerts.extend(self._check_new_station(run_id, tiger, occ))
            alerts.extend(self._check_buffer_proximity(run_id, tiger, occ))

        alerts.extend(self._check_prolonged_absence(run_id, tigers))
        return alerts

    def _dominant_zone(self, station_ids_str: str) -> str:
        if not station_ids_str:
            return "core"
        ids = [s.strip() for s in station_ids_str.split(",") if s.strip()]
        zones = []
        for sid in ids:
            st = self.repo.get_station(sid)
            if st:
                zones.append(st.zone)
        if "buffer" in zones:
            return "buffer"
        if "village_adjacent" in zones:
            return "village_adjacent"
        return "core"

    def _likely_survey_artifact(self, current, previous) -> bool:
        """Flag low-capture runs as potential survey artefacts."""
        curr_count = current.capture_count or 0
        prev_count = previous.capture_count or 0
        if curr_count < 3 and prev_count >= 5:
            return True
        if curr_count < 2:
            return True
        return False

    def _check_new_station(self, run_id: int, tiger, occ) -> list:
        alerts = []
        historical = self.repo.stations_used_by_tiger(tiger.id)
        current_stations = set()
        if occ.station_ids:
            current_stations = {s.strip() for s in occ.station_ids.split(",") if s.strip()}

        new_stations = current_stations - historical
        for sid in new_stations:
            st = self.repo.get_station(sid)
            zone = st.zone if st else "unknown"
            alert = self.repo.add_alert(
                run_id=run_id,
                tiger_id=tiger.id,
                alert_type="new_station",
                severity="medium",
                confidence=0.9,
                title=f"{tiger.tiger_code}: First capture at station {sid}",
                description=f"Individual detected for the first time at station {sid} ({zone} zone).",
                evidence_json=self.repo.dump_evidence({
                    "station_id": sid,
                    "zone": zone,
                    "historical_stations": list(historical),
                }),
                is_survey_artifact=False,
            )
            alerts.append(alert)
        return alerts

    def _check_buffer_proximity(self, run_id: int, tiger, occ) -> list:
        alerts = []
        if not occ.station_ids:
            return alerts

        for sid in occ.station_ids.split(","):
            sid = sid.strip()
            if not sid:
                continue
            st = self.repo.get_station(sid)
            if not st:
                continue
            if st.zone in ("buffer", "village_adjacent") or st.is_village_adjacent:
                alert = self.repo.add_alert(
                    run_id=run_id,
                    tiger_id=tiger.id,
                    alert_type="buffer_movement",
                    severity="high" if st.is_village_adjacent else "medium",
                    confidence=0.88,
                    title=f"{tiger.tiger_code}: Movement into {st.zone} zone",
                    description=(
                        f"Tiger captured at station {sid} in {st.zone} zone"
                        + (" (village-adjacent)" if st.is_village_adjacent else "")
                        + "."
                    ),
                    evidence_json=self.repo.dump_evidence({
                        "station_id": sid,
                        "zone": st.zone,
                        "is_village_adjacent": st.is_village_adjacent,
                        "coordinates": [st.latitude, st.longitude],
                    }),
                    is_survey_artifact=False,
                )
                alerts.append(alert)
        return alerts

    def _check_prolonged_absence(self, run_id: int, tigers) -> list:
        alerts = []
        threshold = timedelta(days=self.cfg.absence_days_threshold)
        now = datetime.utcnow()

        for tiger in tigers:
            last_seen = self.repo.last_sighting_date(tiger.id)
            if not last_seen:
                continue
            sightings = self.repo.get_sightings_for_tiger(tiger.id)
            if len(sightings) < 3:
                continue

            gap = now - last_seen
            if gap > threshold:
                is_artifact = len(sightings) < 5
                alert = self.repo.add_alert(
                    run_id=run_id,
                    tiger_id=tiger.id,
                    alert_type="prolonged_absence",
                    severity="high",
                    confidence=0.75 if not is_artifact else 0.4,
                    title=f"{tiger.tiger_code}: Prolonged absence ({gap.days} days)",
                    description=(
                        f"Previously regular individual not seen for {gap.days} days "
                        f"(threshold: {self.cfg.absence_days_threshold} days)."
                    ),
                    evidence_json=self.repo.dump_evidence({
                        "last_seen": str(last_seen),
                        "gap_days": gap.days,
                        "total_sightings": len(sightings),
                    }),
                    is_survey_artifact=is_artifact,
                )
                alerts.append(alert)
        return alerts
