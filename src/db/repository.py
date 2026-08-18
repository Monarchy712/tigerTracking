"""Database access helpers."""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from src.db.models import (
    Alert,
    ImageRecord,
    MatchReview,
    OccupancySnapshot,
    ProcessingRun,
    Sighting,
    Station,
    Tiger,
    TigerRepresentative,
)


class Repository:
    def __init__(self, session: Session):
        self.session = session

    def create_run(self, input_dir: str) -> ProcessingRun:
        run = ProcessingRun(input_dir=input_dir, status="running")
        self.session.add(run)
        self.session.commit()
        self.session.refresh(run)
        return run

    def complete_run(self, run: ProcessingRun, **stats) -> ProcessingRun:
        for key, value in stats.items():
            if hasattr(run, key):
                setattr(run, key, value)
        run.completed_at = datetime.utcnow()
        run.status = "completed"
        self.session.commit()
        self.session.refresh(run)
        return run

    def add_image(self, **kwargs) -> ImageRecord:
        img = ImageRecord(**kwargs)
        self.session.add(img)
        self.session.commit()
        self.session.refresh(img)
        return img

    def update_image(self, img: ImageRecord, **kwargs) -> ImageRecord:
        for key, value in kwargs.items():
            setattr(img, key, value)
        self.session.commit()
        self.session.refresh(img)
        return img

    def next_tiger_code(self) -> str:
        count = self.session.query(func.count(Tiger.id)).scalar() or 0
        return f"T{count + 1:04d}"

    def create_tiger(self, notes: str | None = None) -> Tiger:
        tiger = Tiger(tiger_code=self.next_tiger_code(), notes=notes)
        self.session.add(tiger)
        self.session.commit()
        self.session.refresh(tiger)
        return tiger

    def add_representative(
        self,
        tiger_id: int,
        image_id: int,
        flank_path: str,
        embedding: list[float] | None = None,
    ) -> TigerRepresentative:
        rep = TigerRepresentative(
            tiger_id=tiger_id,
            image_id=image_id,
            flank_path=flank_path,
            embedding_json=json.dumps(embedding) if embedding is not None else None,
        )
        self.session.add(rep)
        self.session.commit()
        self.session.refresh(rep)
        return rep

    def update_representative_embedding(self, rep_id: int, embedding: list[float]) -> None:
        rep = self.session.get(TigerRepresentative, rep_id)
        if rep:
            rep.embedding_json = json.dumps(embedding)
            self.session.commit()

    def get_representatives(self, tiger_id: int | None = None) -> list[TigerRepresentative]:
        q = self.session.query(TigerRepresentative)
        if tiger_id is not None:
            q = q.filter(TigerRepresentative.tiger_id == tiger_id)
        return q.all()

    def count_representatives(self, tiger_id: int) -> int:
        return (
            self.session.query(func.count(TigerRepresentative.id))
            .filter(TigerRepresentative.tiger_id == tiger_id)
            .scalar()
            or 0
        )

    def add_sighting(self, **kwargs) -> Sighting:
        sighting = Sighting(**kwargs)
        self.session.add(sighting)
        self.session.commit()
        self.session.refresh(sighting)
        return sighting

    def add_review(self, image_id: int, candidate_tiger_id: int | None, confidence: float) -> MatchReview:
        review = MatchReview(
            image_id=image_id,
            candidate_tiger_id=candidate_tiger_id,
            confidence=confidence,
        )
        self.session.add(review)
        self.session.commit()
        self.session.refresh(review)
        return review

    def get_pending_reviews(self) -> list[MatchReview]:
        return (
            self.session.query(MatchReview)
            .filter(MatchReview.status == "pending")
            .order_by(desc(MatchReview.created_at))
            .all()
        )

    def resolve_review(
        self,
        review_id: int,
        decision: str,
        tiger_id: int | None = None,
        notes: str | None = None,
    ) -> MatchReview:
        review = self.session.get(MatchReview, review_id)
        if not review:
            raise ValueError(f"Review {review_id} not found")
        review.status = "resolved"
        review.reviewer_decision = decision
        review.resolved_at = datetime.utcnow()
        review.notes = notes
        if decision == "same" and tiger_id:
            review.candidate_tiger_id = tiger_id
        elif decision == "new":
            review.candidate_tiger_id = None
        self.session.commit()
        self.session.refresh(review)
        return review

    def list_tigers(self) -> list[Tiger]:
        return self.session.query(Tiger).filter(Tiger.is_active.is_(True)).order_by(Tiger.tiger_code).all()

    def get_tiger(self, tiger_id: int) -> Tiger | None:
        return self.session.get(Tiger, tiger_id)

    def get_tiger_by_code(self, code: str) -> Tiger | None:
        return self.session.query(Tiger).filter(Tiger.tiger_code == code).first()

    def get_sightings_for_tiger(self, tiger_id: int) -> list[Sighting]:
        return (
            self.session.query(Sighting)
            .filter(Sighting.tiger_id == tiger_id)
            .order_by(Sighting.captured_at)
            .all()
        )

    def get_all_sightings(self) -> list[Sighting]:
        return self.session.query(Sighting).all()

    def add_occupancy_snapshot(self, **kwargs) -> OccupancySnapshot:
        snap = OccupancySnapshot(**kwargs)
        self.session.add(snap)
        self.session.commit()
        self.session.refresh(snap)
        return snap

    def get_latest_occupancy(self, tiger_id: int, before_run_id: int | None = None) -> OccupancySnapshot | None:
        q = self.session.query(OccupancySnapshot).filter(OccupancySnapshot.tiger_id == tiger_id)
        if before_run_id is not None:
            q = q.filter(OccupancySnapshot.run_id < before_run_id)
        return q.order_by(desc(OccupancySnapshot.run_id)).first()

    def get_occupancy_for_run(self, run_id: int) -> list[OccupancySnapshot]:
        return self.session.query(OccupancySnapshot).filter(OccupancySnapshot.run_id == run_id).all()

    def add_alert(self, **kwargs) -> Alert:
        alert = Alert(**kwargs)
        self.session.add(alert)
        self.session.commit()
        self.session.refresh(alert)
        return alert

    def get_alerts(self, run_id: int | None = None, unacknowledged_only: bool = False) -> list[Alert]:
        q = self.session.query(Alert)
        if run_id is not None:
            q = q.filter(Alert.run_id == run_id)
        if unacknowledged_only:
            q = q.filter(Alert.acknowledged.is_(False))
        return q.order_by(desc(Alert.created_at)).all()

    def upsert_station(self, station_id: str, latitude: float, longitude: float, **kwargs) -> Station:
        station = self.session.query(Station).filter(Station.station_id == station_id).first()
        if station:
            station.latitude = latitude
            station.longitude = longitude
            for k, v in kwargs.items():
                setattr(station, k, v)
        else:
            station = Station(station_id=station_id, latitude=latitude, longitude=longitude, **kwargs)
            self.session.add(station)
        self.session.commit()
        self.session.refresh(station)
        return station

    def get_stations(self) -> list[Station]:
        return self.session.query(Station).all()

    def get_station(self, station_id: str) -> Station | None:
        return self.session.query(Station).filter(Station.station_id == station_id).first()

    def get_runs(self, limit: int = 20) -> list[ProcessingRun]:
        return (
            self.session.query(ProcessingRun)
            .order_by(desc(ProcessingRun.started_at))
            .limit(limit)
            .all()
        )

    def get_run(self, run_id: int) -> ProcessingRun | None:
        return self.session.get(ProcessingRun, run_id)

    def stations_used_by_tiger(
        self,
        tiger_id: int,
        run_id: int | None = None,
        before_run_id: int | None = None,
    ) -> set[str]:
        """Stations an individual has used — optionally within, or before, a run.

        `run_id` restricts to a single run; `before_run_id` gives the prior
        history, which is what "first capture at this station" must compare
        against (all-time history already contains the current run's captures).
        """
        q = self.session.query(Sighting.station_id).filter(
            Sighting.tiger_id == tiger_id, Sighting.station_id.isnot(None)
        )
        if run_id is not None:
            q = q.filter(Sighting.run_id == run_id)
        if before_run_id is not None:
            q = q.filter(Sighting.run_id < before_run_id)
        return {r[0] for r in q.distinct().all() if r[0]}

    def last_sighting_date(self, tiger_id: int) -> datetime | None:
        return (
            self.session.query(func.max(Sighting.captured_at))
            .filter(Sighting.tiger_id == tiger_id)
            .scalar()
        )

    def get_last_sighting(self, tiger_id: int) -> Sighting | None:
        return (
            self.session.query(Sighting)
            .filter(Sighting.tiger_id == tiger_id)
            .order_by(desc(Sighting.captured_at), desc(Sighting.id))
            .first()
        )

    # --- Queries backing the dashboard map, profiles and reports ---

    def get_sightings_in_range(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Sighting]:
        """Sightings with GPS inside a date window (used by the map scrubber)."""
        q = self.session.query(Sighting).filter(
            Sighting.latitude.isnot(None), Sighting.longitude.isnot(None)
        )
        if start is not None:
            q = q.filter(Sighting.captured_at >= start)
        if end is not None:
            q = q.filter(Sighting.captured_at <= end)
        return q.order_by(Sighting.captured_at).all()

    def sighting_date_bounds(self) -> tuple[datetime | None, datetime | None]:
        row = self.session.query(
            func.min(Sighting.captured_at), func.max(Sighting.captured_at)
        ).one()
        return row[0], row[1]

    def get_alerts_for_tiger(self, tiger_id: int, since: datetime | None = None) -> list[Alert]:
        q = self.session.query(Alert).filter(Alert.tiger_id == tiger_id)
        if since is not None:
            q = q.filter(Alert.created_at >= since)
        return q.order_by(desc(Alert.created_at)).all()

    def get_images_for_tiger(self, tiger_id: int) -> list[ImageRecord]:
        """Every frame a tiger was identified in, newest first."""
        return (
            self.session.query(ImageRecord)
            .join(Sighting, Sighting.image_id == ImageRecord.id)
            .filter(Sighting.tiger_id == tiger_id)
            .order_by(desc(Sighting.captured_at))
            .all()
        )

    def get_sightings_for_run(self, run_id: int) -> list[Sighting]:
        return self.session.query(Sighting).filter(Sighting.run_id == run_id).all()

    def get_images_for_run(self, run_id: int) -> list[ImageRecord]:
        return self.session.query(ImageRecord).filter(ImageRecord.run_id == run_id).all()

    def get_image(self, image_id: int) -> ImageRecord | None:
        return self.session.get(ImageRecord, image_id)

    def station_frame_days(self) -> list[tuple[str, datetime]]:
        """(station_id, captured_at) for every frame that reached the pipeline.

        Blanks are included on purpose: a blank frame still proves the camera
        was running that day, which is what trap-night effort counts.
        """
        return (
            self.session.query(ImageRecord.station_id, ImageRecord.captured_at)
            .filter(ImageRecord.station_id.isnot(None), ImageRecord.captured_at.isnot(None))
            .distinct()
            .all()
        )

    def sightings_at_station(
        self,
        station_id: str,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[Sighting]:
        q = self.session.query(Sighting).filter(Sighting.station_id == station_id)
        if since is not None:
            q = q.filter(Sighting.captured_at >= since)
        if until is not None:
            q = q.filter(Sighting.captured_at <= until)
        return q.order_by(Sighting.captured_at).all()

    def count_totals(self) -> dict:
        """Headline counters for the dashboard and report cover page."""
        return {
            "tigers": self.session.query(func.count(Tiger.id))
            .filter(Tiger.is_active.is_(True))
            .scalar()
            or 0,
            "sightings": self.session.query(func.count(Sighting.id)).scalar() or 0,
            "images": self.session.query(func.count(ImageRecord.id)).scalar() or 0,
            "blanks": self.session.query(func.count(ImageRecord.id))
            .filter(ImageRecord.is_blank.is_(True))
            .scalar()
            or 0,
            "alerts": self.session.query(func.count(Alert.id)).scalar() or 0,
            "pending_reviews": self.session.query(func.count(MatchReview.id))
            .filter(MatchReview.status == "pending")
            .scalar()
            or 0,
        }

    def dump_evidence(self, data: dict) -> str:
        return json.dumps(data, default=str)
