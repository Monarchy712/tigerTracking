"""Tiger catalogue matching using MiewID embeddings (with mock fallback)."""

from __future__ import annotations

from pathlib import Path

from src.config import app_config
from src.db.repository import Repository
from src.matching.embedding_matcher import EmbeddingMatcher, MatchDecision


class TigerCatalogue:
    def __init__(self, repo: Repository, matcher: EmbeddingMatcher | None = None):
        self.repo = repo
        self.matcher = matcher or EmbeddingMatcher(repo)
        self.cfg = app_config.matching

    def find_best_match(self, flank_path: Path) -> MatchDecision:
        return self.matcher.find_best_match(flank_path)

    def enroll_tiger(self, image_id: int, flank_path: Path) -> int:
        embedding = self.matcher.compute_embedding(flank_path)
        tiger = self.repo.create_tiger()
        self.repo.add_representative(tiger.id, image_id, str(flank_path), embedding=embedding)
        return tiger.id

    def add_representative_if_needed(self, tiger_id: int, image_id: int, flank_path: Path) -> None:
        count = self.repo.count_representatives(tiger_id)
        if count < self.cfg.max_representatives_per_tiger:
            embedding = self.matcher.compute_embedding(flank_path)
            self.repo.add_representative(tiger_id, image_id, str(flank_path), embedding=embedding)

    def record_sighting(
        self,
        tiger_id: int,
        image_id: int,
        run_id: int,
        station_id: str | None,
        captured_at,
        latitude: float | None,
        longitude: float | None,
        confidence: float,
        match_type: str,
    ) -> None:
        self.repo.add_sighting(
            tiger_id=tiger_id,
            image_id=image_id,
            run_id=run_id,
            station_id=station_id,
            captured_at=captured_at,
            latitude=latitude,
            longitude=longitude,
            match_confidence=confidence,
            match_type=match_type,
        )
