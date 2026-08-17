"""Normalize MegaDetector result formats from PyTorch-Wildlife."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParsedDetection:
    category: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, w, h


def _category_name(class_id: int) -> str:
    mapping = {0: "animal", 1: "person", 2: "vehicle"}
    return mapping.get(class_id, "animal")


def parse_megadetector_result(result, threshold: float = 0.0) -> list[ParsedDetection]:
    """Parse PyTorch-Wildlife / MegaDetector outputs into a uniform list."""
    if result is None:
        return []

    parsed: list[ParsedDetection] = []

    if isinstance(result, dict) and "detections" in result:
        detections = result["detections"]

        if hasattr(detections, "xyxy") and hasattr(detections, "confidence"):
            class_ids = getattr(detections, "class_id", None)
            for idx, (xyxy, conf) in enumerate(zip(detections.xyxy, detections.confidence)):
                conf_val = float(conf)
                if conf_val < threshold:
                    continue
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                class_id = int(class_ids[idx]) if class_ids is not None else 0
                parsed.append(
                    ParsedDetection(
                        category=_category_name(class_id),
                        confidence=conf_val,
                        bbox=(x1, y1, max(1, x2 - x1), max(1, y2 - y1)),
                    )
                )
            return parsed

        if isinstance(detections, list):
            for det in detections:
                conf_val = float(det.get("confidence", det.get("conf", 0.0)))
                if conf_val < threshold:
                    continue
                bbox = det.get("bbox", det.get("box", [0, 0, 0, 0]))
                if len(bbox) == 4:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    parsed.append(
                        ParsedDetection(
                            category=str(det.get("category", "animal")),
                            confidence=conf_val,
                            bbox=(x1, y1, max(1, x2 - x1), max(1, y2 - y1)),
                        )
                    )
            return parsed

    if isinstance(result, list):
        for record in result:
            if not isinstance(record, dict):
                continue
            for det in record.get("detections", []):
                conf_val = float(det.get("confidence", 0.0))
                if conf_val < threshold:
                    continue
                bbox = det.get("bbox", [0, 0, 0, 0])
                x1, y1, x2, y2 = [int(v) for v in bbox]
                parsed.append(
                    ParsedDetection(
                        category=str(det.get("category", "animal")),
                        confidence=conf_val,
                        bbox=(x1, y1, max(1, x2 - x1), max(1, y2 - y1)),
                    )
                )

    return parsed


def best_animal_detection(detections: list[ParsedDetection]) -> ParsedDetection | None:
    animals = [d for d in detections if d.category == "animal"]
    if not animals:
        return None
    return max(animals, key=lambda d: d.confidence)


def max_detection_confidence(detections: list[ParsedDetection]) -> float:
    if not detections:
        return 0.0
    return max(d.confidence for d in detections)
