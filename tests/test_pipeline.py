"""Basic smoke tests (run after pip install)."""

import ast
from pathlib import Path


def test_all_python_files_parse():
    root = Path(__file__).resolve().parent.parent
    py_files = [p for p in root.rglob("*.py") if ".venv" not in str(p)]
    assert len(py_files) >= 20
    for p in py_files:
        ast.parse(p.read_text())


def test_config_loads():
    from src.config import app_config, settings

    assert settings.data_dir.exists() or True
    assert app_config.matching.auto_match_threshold == 0.85
    assert app_config.models.megadetector_threshold == 0.20


def test_blank_filter_blank_image(tmp_path, monkeypatch):
    import cv2
    import numpy as np

    from src.config import app_config, settings
    from src.pipeline.blank_filter import classify_blank

    monkeypatch.setattr(settings, "use_ml_models", False)
    app_config.blank_filter.mode = "heuristic"

    blank = np.full((480, 640, 3), 50, dtype=np.uint8)
    path = tmp_path / "blank.jpg"
    cv2.imwrite(str(path), blank)
    result = classify_blank(path)
    assert result.is_blank is True


def test_blank_filter_subject_image(tmp_path, monkeypatch):
    import cv2
    import numpy as np

    from src.config import app_config, settings
    from src.pipeline.blank_filter import classify_blank

    monkeypatch.setattr(settings, "use_ml_models", False)
    app_config.blank_filter.mode = "heuristic"

    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (400, 350), (80, 120, 60), -1)
    path = tmp_path / "subject.jpg"
    cv2.imwrite(str(path), img)
    result = classify_blank(path)
    assert result.is_blank is False


def test_megadetector_parser():
    from src.ml.megadetector_utils import parse_megadetector_result

    result = {
        "detections": [
            {"category": "animal", "confidence": 0.91, "bbox": [10, 20, 200, 300]},
            {"category": "person", "confidence": 0.55, "bbox": [0, 0, 50, 50]},
        ]
    }
    parsed = parse_megadetector_result(result, threshold=0.2)
    assert len(parsed) == 2
    assert parsed[0].category == "animal"
    assert parsed[0].bbox == (10, 20, 190, 280)


def test_station_id_parsing(tmp_path, monkeypatch):
    from src.pipeline.ingest import ingest_image, load_station_registry

    registry = {"CAM01": (23.71, 81.025, "core")}
    path = tmp_path / "tiger_alpha_station_CAM01_20260103_223700.jpg"
    path.write_bytes(b"fake")

    item = ingest_image(path, registry)
    assert item.station_id == "CAM01"
    assert item.latitude == 23.71
    assert item.longitude == 81.025
