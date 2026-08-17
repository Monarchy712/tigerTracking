"""Human review queue for ambiguous tiger matches."""

from __future__ import annotations

from pathlib import Path

from src.db.models import MatchReview
from src.db.repository import Repository
from src.matching.catalogue import TigerCatalogue


class ReviewQueue:
    def __init__(self, repo: Repository, catalogue: TigerCatalogue):
        self.repo = repo
        self.catalogue = catalogue

    def pending(self) -> list[MatchReview]:
        return self.repo.get_pending_reviews()

    def resolve_same(
        self,
        review_id: int,
        tiger_id: int,
        image_id: int,
        flank_path: str,
        run_id: int,
        station_id: str | None,
        captured_at,
        latitude: float | None,
        longitude: float | None,
        confidence: float,
    ) -> None:
        self.repo.resolve_review(review_id, decision="same", tiger_id=tiger_id)
        self.catalogue.add_representative_if_needed(tiger_id, image_id, Path(flank_path))
        self.catalogue.record_sighting(
            tiger_id=tiger_id,
            image_id=image_id,
            run_id=run_id,
            station_id=station_id,
            captured_at=captured_at,
            latitude=latitude,
            longitude=longitude,
            confidence=confidence,
            match_type="human_review",
        )

    def resolve_new(
        self,
        review_id: int,
        image_id: int,
        flank_path: str,
        run_id: int,
        station_id: str | None,
        captured_at,
        latitude: float | None,
        longitude: float | None,
    ) -> int:
        self.repo.resolve_review(review_id, decision="new")
        tiger_id = self.catalogue.enroll_tiger(image_id, Path(flank_path))
        self.catalogue.record_sighting(
            tiger_id=tiger_id,
            image_id=image_id,
            run_id=run_id,
            station_id=station_id,
            captured_at=captured_at,
            latitude=latitude,
            longitude=longitude,
            confidence=1.0,
            match_type="human_enroll",
        )
        return tiger_id

    def resolve_reject(self, review_id: int, notes: str | None = None) -> None:
        self.repo.resolve_review(review_id, decision="reject", notes=notes)
