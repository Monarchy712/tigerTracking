"""Client adapter for friend's tiger comparison model."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from src.config import settings


@dataclass
class CompareResult:
    same_tiger: bool
    confidence: float
    model_version: str


class FriendModelClient(ABC):
    @abstractmethod
    def compare(self, image_a: Path, image_b: Path) -> CompareResult:
        ...


class HttpFriendModelClient(FriendModelClient):
    """Calls friend's model served as HTTP POST /compare."""

    def __init__(self, url: str | None = None):
        self.url = url or settings.friend_model_url

    def compare(self, image_a: Path, image_b: Path) -> CompareResult:
        with open(image_a, "rb") as fa, open(image_b, "rb") as fb:
            files = {
                "image_a": (image_a.name, fa, "image/jpeg"),
                "image_b": (image_b.name, fb, "image/jpeg"),
            }
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(self.url, files=files)
                resp.raise_for_status()
                data = resp.json()

        return CompareResult(
            same_tiger=bool(data.get("same_tiger", False)),
            confidence=float(data.get("confidence", 0.0)),
            model_version=str(data.get("model_version", "http")),
        )


class MockFriendModelClient(FriendModelClient):
    """
    Deterministic mock for demo/dev when friend's model isn't running.
    Uses perceptual hash similarity as a stand-in for stripe pattern matching.
    Replace with HttpFriendModelClient in production.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model_version = "mock-v1"

    def _file_hash(self, path: Path) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _similarity(self, path_a: Path, path_b: Path) -> float:
        """Compute pseudo-similarity from file content hashes + names."""
        ha = self._file_hash(path_a)
        hb = self._file_hash(path_b)

        if ha == hb:
            return 0.99

        name_a = path_a.stem.lower()
        name_b = path_b.stem.lower()

        # Match tigers sharing the same prefix (e.g. tiger_alpha_...)
        prefix_a = name_a.split("_station_")[0] if "_station_" in name_a else name_a.split("_")[0]
        prefix_b = name_b.split("_station_")[0] if "_station_" in name_b else name_b.split("_")[0]
        if prefix_a == prefix_b and prefix_a.startswith("tiger"):
            return 0.92

        shared_prefix = 0
        for ca, cb in zip(name_a, name_b):
            if ca == cb:
                shared_prefix += 1
            else:
                break

        hash_sim = sum(a == b for a, b in zip(ha, hb)) / len(ha)
        prefix_bonus = min(0.3, shared_prefix * 0.05)
        noise = (int(ha[:4], 16) ^ int(hb[:4], 16)) / 65535.0 * 0.15
        return float(np.clip(hash_sim * 0.5 + prefix_bonus + 0.2 - noise, 0.0, 0.99))

    def compare(self, image_a: Path, image_b: Path) -> CompareResult:
        sim = self._similarity(image_a, image_b)
        return CompareResult(
            same_tiger=sim >= 0.75,
            confidence=sim,
            model_version=self.model_version,
        )


def get_friend_model_client() -> FriendModelClient:
    if settings.friend_model_mock:
        return MockFriendModelClient()
    return HttpFriendModelClient()
