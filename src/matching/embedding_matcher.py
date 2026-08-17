"""MiewID embedding-based catalogue matching."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.config import app_config
from src.db.repository import Repository
from src.matching.friend_model_client import CompareResult, MockFriendModelClient
from src.ml.model_registry import cosine_similarity, embed_flank_image, ml_available


@dataclass
class MatchDecision:
    action: str  # auto_match | review | enroll
    tiger_id: int | None
    confidence: float
    compared_against: str | None
    compare_result: CompareResult | None = None


class EmbeddingMatcher:
    def __init__(self, repo: Repository):
        self.repo = repo
        self.cfg = app_config.matching

    def compute_embedding(self, flank_path: Path) -> list[float] | None:
        if not ml_available():
            return None
        return embed_flank_image(flank_path)

    def find_best_match(self, flank_path: Path) -> MatchDecision:
        if not ml_available():
            return self._find_best_match_mock(flank_path)

        query_emb = self.compute_embedding(flank_path)
        if query_emb is None:
            return self._find_best_match_mock(flank_path)

        representatives = self.repo.get_representatives()

        if not representatives:
            return MatchDecision(
                action="enroll",
                tiger_id=None,
                confidence=0.0,
                compared_against=None,
                compare_result=None,
            )

        best_conf = 0.0
        best_tiger_id: int | None = None
        best_path: str | None = None

        for rep in representatives:
            rep_path = Path(rep.flank_path)
            if not rep_path.exists():
                continue

            if rep.embedding_json:
                rep_emb = json.loads(rep.embedding_json)
                sim = cosine_similarity(query_emb, rep_emb)
            else:
                rep_emb = self.compute_embedding(rep_path)
                if rep_emb is None:
                    continue
                self.repo.update_representative_embedding(rep.id, rep_emb)
                sim = cosine_similarity(query_emb, rep_emb)

            if sim > best_conf:
                best_conf = sim
                best_tiger_id = rep.tiger_id
                best_path = str(rep_path)

        return self._decision_from_confidence(best_conf, best_tiger_id, best_path)

    def _find_best_match_mock(self, flank_path: Path) -> MatchDecision:
        """Fallback to pairwise mock client when ML deps unavailable."""
        client = MockFriendModelClient()
        representatives = self.repo.get_representatives()
        if not representatives:
            return MatchDecision("enroll", None, 0.0, None, None)

        best_conf = 0.0
        best_tiger_id: int | None = None
        best_result: CompareResult | None = None
        best_path: str | None = None

        for rep in representatives:
            rep_path = Path(rep.flank_path)
            if not rep_path.exists():
                continue
            result = client.compare(flank_path, rep_path)
            if result.confidence > best_conf:
                best_conf = result.confidence
                best_tiger_id = rep.tiger_id
                best_result = result
                best_path = str(rep_path)

        decision = self._decision_from_confidence(best_conf, best_tiger_id, best_path)
        decision.compare_result = best_result
        return decision

    def _decision_from_confidence(
        self,
        confidence: float,
        tiger_id: int | None,
        compared_against: str | None,
    ) -> MatchDecision:
        if confidence >= self.cfg.auto_match_threshold:
            action = "auto_match"
        elif confidence >= self.cfg.review_threshold:
            action = "review"
        else:
            action = "enroll"
            tiger_id = None

        return MatchDecision(
            action=action,
            tiger_id=tiger_id,
            confidence=confidence,
            compared_against=compared_against,
            compare_result=None,
        )
