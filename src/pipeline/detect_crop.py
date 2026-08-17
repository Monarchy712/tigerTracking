"""Tiger detection and flank region extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.config import app_config
from src.ml.megadetector_utils import best_animal_detection, parse_megadetector_result
from src.ml.model_registry import ml_available, run_megadetector


@dataclass
class DetectionResult:
    has_tiger: bool
    confidence: float
    bbox: tuple[int, int, int, int] | None
    flank_bbox: tuple[int, int, int, int] | None
    flank_path: Path | None
    reason: str


def _detect_subject_bbox_heuristic(gray: np.ndarray) -> tuple[tuple[int, int, int, int] | None, float]:
    """Heuristic subject detection via edge clustering."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    closed = cv2.dilate(closed, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0

    h, w = gray.shape
    min_area = (h * w) * 0.05
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid:
        return None, 0.0

    largest = max(valid, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)
    area_ratio = (bw * bh) / (h * w)
    confidence = min(0.95, 0.5 + area_ratio * 2)
    return (x, y, bw, bh), confidence


def _extract_flank_bbox(bbox: tuple[int, int, int, int], img_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Extract flank region from body bbox — central-right portion typical for stripe ID."""
    x, y, w, h = bbox
    fh = int(h * 0.55)
    fy = y + int(h * 0.25)
    fw = int(w * 0.45)
    fx = x + int(w * 0.35)
    ih, iw = img_shape
    fx = max(0, min(fx, iw - 1))
    fy = max(0, min(fy, ih - 1))
    fw = min(fw, iw - fx)
    fh = min(fh, ih - fy)
    return fx, fy, fw, fh


def _crop_and_save_flank(img: np.ndarray, flank_bbox: tuple[int, int, int, int], output_dir: Path, image_id: int) -> Path | None:
    fx, fy, fw, fh = flank_bbox
    if fw < 20 or fh < 20:
        return None

    flank = img[fy : fy + fh, fx : fx + fw]
    output_dir.mkdir(parents=True, exist_ok=True)
    flank_path = output_dir / f"flank_{image_id}.jpg"
    cv2.imwrite(str(flank_path), flank)
    return flank_path


def detect_and_crop(path: Path, output_dir: Path, image_id: int) -> DetectionResult:
    """Detect tiger/subject and crop flank region for stripe matching."""
    img = cv2.imread(str(path))
    if img is None:
        return DetectionResult(
            has_tiger=False,
            confidence=0.0,
            bbox=None,
            flank_bbox=None,
            flank_path=None,
            reason="unreadable",
        )

    threshold = app_config.models.megadetector_threshold

    if ml_available():
        md_result = run_megadetector(path)
        detections = parse_megadetector_result(md_result, threshold=threshold)
        best = best_animal_detection(detections)

        if best is None:
            return DetectionResult(
                has_tiger=False,
                confidence=0.0,
                bbox=None,
                flank_bbox=None,
                flank_path=None,
                reason="no_animal_detected",
            )

        bbox = best.bbox
        confidence = best.confidence
        reason = "animal_detected"
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bbox, confidence = _detect_subject_bbox_heuristic(gray)
        if bbox is None or confidence < 0.55:
            return DetectionResult(
                has_tiger=False,
                confidence=confidence,
                bbox=bbox,
                flank_bbox=None,
                flank_path=None,
                reason="no_subject_detected",
            )
        reason = "subject_detected_heuristic"

    flank_bbox = _extract_flank_bbox(bbox, img.shape[:2])
    flank_path = _crop_and_save_flank(img, flank_bbox, output_dir, image_id)

    if flank_path is None:
        return DetectionResult(
            has_tiger=False,
            confidence=confidence,
            bbox=bbox,
            flank_bbox=flank_bbox,
            flank_path=None,
            reason="flank_too_small",
        )

    return DetectionResult(
        has_tiger=True,
        confidence=confidence,
        bbox=bbox,
        flank_bbox=flank_bbox,
        flank_path=flank_path,
        reason=reason,
    )
