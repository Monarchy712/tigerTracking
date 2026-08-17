"""Blank image detection with quarantine-based safe removal."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.config import app_config
from src.ml.megadetector_utils import max_detection_confidence, parse_megadetector_result
from src.ml.model_registry import ml_available, run_megadetector


@dataclass
class BlankFilterResult:
    is_blank: bool
    confidence: float
    variance: float
    edge_density: float
    reason: str


def _load_grayscale(path: Path, max_dim: int = 512) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def classify_blank_heuristic(path: Path) -> BlankFilterResult:
    """Classify image as blank using variance + edge density heuristics."""
    cfg = app_config.blank_filter
    gray = _load_grayscale(path)
    if gray is None:
        return BlankFilterResult(
            is_blank=True,
            confidence=0.99,
            variance=0.0,
            edge_density=0.0,
            reason="unreadable_image",
        )

    variance = float(np.var(gray))
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / edges.size

    blank_signals = []
    if variance < cfg.variance_threshold:
        blank_signals.append(("low_variance", 0.4))
    if edge_density < cfg.edge_density_threshold:
        blank_signals.append(("low_edges", 0.4))

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / (hist.sum() + 1e-6)
    entropy = -float(np.sum(hist_norm * np.log2(hist_norm + 1e-10)))
    if entropy < 3.0:
        blank_signals.append(("low_entropy", 0.2))

    if not blank_signals:
        return BlankFilterResult(
            is_blank=False,
            confidence=0.85,
            variance=variance,
            edge_density=edge_density,
            reason="subject_likely_present",
        )

    confidence = min(0.99, sum(w for _, w in blank_signals))
    is_blank = confidence >= cfg.confidence_threshold
    reason = "+".join(s[0] for s in blank_signals)

    return BlankFilterResult(
        is_blank=is_blank,
        confidence=confidence,
        variance=variance,
        edge_density=edge_density,
        reason=reason,
    )


def classify_blank_megadetector(path: Path) -> BlankFilterResult:
    """Classify blank frames using MegaDetector — no detection above threshold => blank."""
    threshold = app_config.models.megadetector_threshold
    result = run_megadetector(path)
    detections = parse_megadetector_result(result, threshold=threshold)
    max_conf = max_detection_confidence(detections)

    is_blank = max_conf < threshold
    confidence = (1.0 - max_conf) if is_blank else max_conf

    return BlankFilterResult(
        is_blank=is_blank,
        confidence=confidence,
        variance=0.0,
        edge_density=0.0,
        reason="no_detection" if is_blank else "subject_detected",
    )


def classify_blank(path: Path) -> BlankFilterResult:
    """Dispatch blank classification based on configured mode."""
    mode = app_config.blank_filter.mode
    use_ml = ml_available()

    if mode == "heuristic" or not use_ml:
        return classify_blank_heuristic(path)

    if mode == "megadetector":
        return classify_blank_megadetector(path)

    # hybrid: fast heuristic pre-screen, MegaDetector confirmation on uncertain frames
    heuristic = classify_blank_heuristic(path)
    if heuristic.is_blank and heuristic.confidence >= 0.90:
        return heuristic
    if not heuristic.is_blank and heuristic.confidence >= 0.90:
        return heuristic

    return classify_blank_megadetector(path)


def quarantine_blank(source: Path, quarantine_dir: Path, run_id: int) -> Path:
    """Move blank image to quarantine (reversible staged delete)."""
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    dest = quarantine_dir / f"run_{run_id}" / source.name
    dest.parent.mkdir(parents=True, exist_ok=True)

    if source.exists():
        shutil.move(str(source), str(dest))
    return dest


def estimate_time_saved_sec(num_blank: int, avg_processing_sec: float = 2.5) -> float:
    """Estimate downstream processing time saved by removing blanks."""
    return num_blank * avg_processing_sec
