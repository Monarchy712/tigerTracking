"""Visual and behavioural anomaly detection on retained camera-trap frames.

Three deliberately cheap detectors run alongside re-identification:

* ``anomaly:injured`` — body-box aspect ratio outside the normal standing range
  (a limping or crouched tiger photographs with unusual proportions), optionally
  reinforced by a visible-blood colour heuristic on the flank crop.
* ``anomaly:mating`` — two different individuals at one station inside a short
  window, or two animals detected in a single frame.
* ``anomaly:water`` — a waterhole station whose recent capture rate exceeds its
  own historical rate by 2 sigma, which reads as drought / thirst stress.

None of these are diagnoses. They are triage signals that tell a ranger which
frames deserve a human eye first, so every alert carries its evidence.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from src.config import app_config
from src.db.repository import Repository


@dataclass
class FrameAnomaly:
    """Per-frame signals, persisted on ``images.anomaly_json``."""

    aspect_ratio: float | None = None
    aspect_anomalous: bool = False
    blood_fraction: float = 0.0
    blood_flagged: bool = False
    animal_count: int = 0
    injured_score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def injured_suspect(self) -> bool:
        return self.injured_score >= 0.5

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def _blood_pixel_fraction(flank_path: Path) -> float:
    """Fraction of flank-crop pixels in the dark-red range of a fresh wound."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return 0.0

    img = cv2.imread(str(flank_path))
    if img is None or img.size == 0:
        return 0.0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Red wraps around the hue circle, so both ends are needed. Saturation and
    # value bounds keep tiger orange (bright, high-value) out of the mask.
    lower = cv2.inRange(hsv, np.array([0, 90, 40]), np.array([10, 255, 160]))
    upper = cv2.inRange(hsv, np.array([170, 90, 40]), np.array([180, 255, 160]))
    mask = cv2.bitwise_or(lower, upper)
    return float(mask.astype(bool).sum()) / float(mask.size)


def analyse_frame(
    flank_path: Path | None,
    bbox: tuple[int, int, int, int] | None,
    animal_count: int = 1,
) -> FrameAnomaly:
    """Score one retained frame. Cheap enough to run on every detection."""
    cfg = app_config.anomaly
    result = FrameAnomaly(animal_count=animal_count)
    if not cfg.enabled:
        return result

    if bbox:
        _x, _y, w, h = bbox
        if h > 0:
            ratio = w / h
            result.aspect_ratio = round(ratio, 3)
            result.aspect_anomalous = not (
                cfg.injured_aspect_min <= ratio <= cfg.injured_aspect_max
            )
            if result.aspect_anomalous:
                result.reasons.append(f"body aspect ratio {ratio:.2f} outside normal range")

    if flank_path and Path(flank_path).exists():
        result.blood_fraction = round(_blood_pixel_fraction(Path(flank_path)), 4)
        result.blood_flagged = result.blood_fraction >= cfg.blood_pixel_fraction
        if result.blood_flagged:
            result.reasons.append(
                f"{result.blood_fraction:.1%} of flank crop in blood-red colour range"
            )

    result.injured_score = round(
        0.55 * result.aspect_anomalous + 0.45 * result.blood_flagged, 3
    )
    return result


class AnomalyEngine:
    """Aggregates per-frame signals for a run into ``anomaly:*`` alerts."""

    def __init__(self, repo: Repository):
        self.repo = repo
        self.cfg = app_config.anomaly

    def run(self, run_id: int) -> list:
        if not self.cfg.enabled:
            return []
        alerts: list = []
        alerts.extend(self._check_injured(run_id))
        alerts.extend(self._check_mating(run_id))
        alerts.extend(self._check_water_stress(run_id))
        return alerts

    # --- injured ---------------------------------------------------------

    def _check_injured(self, run_id: int) -> list:
        """Fire once per tiger whose frames repeatedly look injured."""
        alerts = []
        sightings = self.repo.get_sightings_for_run(run_id)

        per_tiger: dict[int, list[tuple[int, FrameAnomaly]]] = {}
        for sighting in sightings:
            image = self.repo.get_image(sighting.image_id)
            if not image or not image.anomaly_json:
                continue
            try:
                signals = FrameAnomaly(**json.loads(image.anomaly_json))
            except (json.JSONDecodeError, TypeError):
                continue
            if signals.injured_suspect:
                per_tiger.setdefault(sighting.tiger_id, []).append((image.id, signals))

        for tiger_id, hits in per_tiger.items():
            if len(hits) < self.cfg.min_frames:
                continue  # single-frame signals are almost always pose noise
            tiger = self.repo.get_tiger(tiger_id)
            if not tiger:
                continue

            mean_score = statistics.fmean(s.injured_score for _img, s in hits)
            reasons = sorted({r for _img, s in hits for r in s.reasons})
            alerts.append(
                self.repo.add_alert(
                    run_id=run_id,
                    tiger_id=tiger_id,
                    alert_type="anomaly:injured",
                    severity="high" if mean_score >= 0.8 else "medium",
                    confidence=round(min(0.9, 0.45 + 0.1 * len(hits) + mean_score * 0.3), 2),
                    title=f"{tiger.tiger_code}: possible injury — needs visual check",
                    description=(
                        f"{len(hits)} frames in this run show an abnormal body posture or "
                        f"blood-coloured pixels on the flank. Verify before dispatching a patrol."
                    ),
                    evidence_json=self.repo.dump_evidence({
                        "frame_count": len(hits),
                        "image_ids": [img_id for img_id, _s in hits],
                        "mean_injured_score": round(mean_score, 3),
                        "reasons": reasons,
                    }),
                    is_survey_artifact=False,
                )
            )
        return alerts

    # --- mating ----------------------------------------------------------

    def _check_mating(self, run_id: int) -> list:
        """Two individuals sharing a station in a short window, or one frame with two animals."""
        alerts = []
        window = timedelta(hours=self.cfg.mating_window_hours)
        sightings = [s for s in self.repo.get_sightings_for_run(run_id) if s.captured_at]

        by_station: dict[str, list] = {}
        for sighting in sightings:
            if sighting.station_id:
                by_station.setdefault(sighting.station_id, []).append(sighting)

        reported: set[tuple[str, int, int]] = set()
        for station_id, group in by_station.items():
            group.sort(key=lambda s: s.captured_at)
            for i, first in enumerate(group):
                nearby = [
                    s for s in group[i + 1:]
                    if s.captured_at - first.captured_at <= window
                ]
                others = {s.tiger_id for s in nearby if s.tiger_id != first.tiger_id}
                # A courting pair keeps the site to itself. Three or more
                # individuals in one window is a waterhole or a kill, not mating,
                # so the pairwise alerts there would be noise.
                if len(others) != 1:
                    continue
                second = next(s for s in nearby if s.tiger_id in others)
                pair = (station_id, *sorted((first.tiger_id, second.tiger_id)))
                if pair in reported:
                    continue
                reported.add(pair)
                alerts.append(self._mating_alert(run_id, first, second, station_id))

        for sighting in sightings:
            image = self.repo.get_image(sighting.image_id)
            if image and (image.animal_count or 0) >= 2:
                tiger = self.repo.get_tiger(sighting.tiger_id)
                alerts.append(
                    self.repo.add_alert(
                        run_id=run_id,
                        tiger_id=sighting.tiger_id,
                        alert_type="anomaly:mating",
                        severity="low",
                        confidence=0.6,
                        title=(
                            f"{tiger.tiger_code if tiger else 'Unknown'}: "
                            f"two animals in one frame at {sighting.station_id or 'unknown station'}"
                        ),
                        description=(
                            "The detector found two animals in a single frame — consistent with "
                            "a mating pair or a female with a sub-adult cub."
                        ),
                        evidence_json=self.repo.dump_evidence({
                            "image_id": image.id,
                            "animal_count": image.animal_count,
                            "station_id": sighting.station_id,
                            "captured_at": sighting.captured_at,
                        }),
                        is_survey_artifact=False,
                    )
                )
        return alerts

    def _mating_alert(self, run_id: int, first, second, station_id: str):
        tiger_a = self.repo.get_tiger(first.tiger_id)
        tiger_b = self.repo.get_tiger(second.tiger_id)
        gap_hours = (second.captured_at - first.captured_at).total_seconds() / 3600.0
        return self.repo.add_alert(
            run_id=run_id,
            tiger_id=first.tiger_id,
            alert_type="anomaly:mating",
            severity="low",
            confidence=round(max(0.5, 0.85 - gap_hours / 24.0), 2),
            title=(
                f"{tiger_a.tiger_code if tiger_a else '?'} and "
                f"{tiger_b.tiger_code if tiger_b else '?'}: co-occurrence at {station_id}"
            ),
            description=(
                f"Two individuals were captured at station {station_id} "
                f"{gap_hours:.1f} hours apart — possible mating or territorial encounter."
            ),
            evidence_json=self.repo.dump_evidence({
                "station_id": station_id,
                "tiger_ids": [first.tiger_id, second.tiger_id],
                "tiger_codes": [
                    tiger_a.tiger_code if tiger_a else None,
                    tiger_b.tiger_code if tiger_b else None,
                ],
                "gap_hours": round(gap_hours, 2),
                "window_hours": self.cfg.mating_window_hours,
            }),
            is_survey_artifact=False,
        )

    # --- water stress ----------------------------------------------------

    def _check_water_stress(self, run_id: int) -> list:
        """Waterhole stations whose recent capture rate is 2 sigma above their own baseline."""
        alerts = []
        lookback = timedelta(days=self.cfg.water_lookback_days)
        waterholes = [st for st in self.repo.get_stations() if st.is_waterhole]

        for station in waterholes:
            history = self.repo.sightings_at_station(station.station_id)
            dated = [s for s in history if s.captured_at]
            if len(dated) < self.cfg.min_frames:
                continue

            latest = max(s.captured_at for s in dated)
            earliest = min(s.captured_at for s in dated)
            # A station with only a few months of history cannot support a full
            # 30-day comparison window, so the window shrinks to a third of the
            # available span and the alert reports the window it actually used.
            window = min(lookback, (latest - earliest) / 3)
            if window < timedelta(days=1):
                continue
            window_days = window.total_seconds() / 86400

            recent = [s for s in dated if s.captured_at > latest - window]
            baseline = self._baseline_windows(dated, latest, window)
            if len(baseline) < 3:
                continue  # not enough history for a meaningful sigma

            mean = statistics.fmean(baseline)
            sigma = statistics.pstdev(baseline)
            if sigma <= 0:
                continue

            excess = (len(recent) - mean) / sigma
            if excess < self.cfg.water_sigma:
                continue

            alerts.append(
                self.repo.add_alert(
                    run_id=run_id,
                    tiger_id=None,
                    alert_type="anomaly:water",
                    severity="medium",
                    confidence=round(min(0.9, 0.5 + excess / 10.0), 2),
                    title=f"Station {station.station_id}: unusual waterhole activity",
                    description=(
                        f"{len(recent)} captures in the last {window_days:.0f} days "
                        f"against a rolling baseline of {mean:.1f} ({excess:.1f} sigma above "
                        "normal). Elevated waterhole use can indicate drought stress."
                    ),
                    evidence_json=self.repo.dump_evidence({
                        "station_id": station.station_id,
                        "station_name": station.name,
                        "recent_captures": len(recent),
                        "baseline_mean": round(mean, 2),
                        "baseline_sigma": round(sigma, 2),
                        "sigma_above": round(excess, 2),
                        "window_days": round(window_days, 1),
                        "baseline_samples": len(baseline),
                    }),
                    is_survey_artifact=False,
                )
            )
        return alerts

    @staticmethod
    def _baseline_windows(sightings: list, latest: datetime, window: timedelta) -> list[int]:
        """Rolling capture counts over the history preceding the recent window.

        Windows slide by a quarter of their length rather than tiling end to end,
        so a station yields a usable baseline after a few months of history
        instead of needing four full survey windows.
        """
        earliest = min(s.captured_at for s in sightings)
        step = window / 4
        counts: list[int] = []
        end = latest - window
        while end - window >= earliest and len(counts) < 24:
            start = end - window
            counts.append(sum(1 for s in sightings if start < s.captured_at <= end))
            end -= step
        return counts
