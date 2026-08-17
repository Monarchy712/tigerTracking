#!/usr/bin/env python3
"""Smoke test for ML pipeline stages."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    from src.config import app_config
    from src.ml.model_registry import embed_flank_image, ml_available, warmup_models
    from src.pipeline.blank_filter import classify_blank
    from src.pipeline.detect_crop import detect_and_crop

    demo_dir = Path("data/raw/demo")
    if not demo_dir.exists():
        print("No demo data found. Run: python scripts/seed_demo_data.py")
        sys.exit(1)

    sample = next(demo_dir.glob("*.jpg"))
    print(f"ML enabled: {ml_available()}")
    print(f"Blank filter mode: {app_config.blank_filter.mode}")

    if ml_available():
        warmup_models()

    blank = classify_blank(sample)
    print(f"Blank result: is_blank={blank.is_blank}, confidence={blank.confidence:.2f}, reason={blank.reason}")

    detect = detect_and_crop(sample, Path("data/processed/flanks"), 9999)
    print(
        f"Detect result: has_tiger={detect.has_tiger}, confidence={detect.confidence:.2f}, reason={detect.reason}"
    )

    if detect.flank_path and ml_available():
        emb = embed_flank_image(detect.flank_path)
        print(f"Embedding dimensions: {len(emb)}")

    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
