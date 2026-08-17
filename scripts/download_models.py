#!/usr/bin/env python3
"""One-time download of MegaDetector and MiewID model weights."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _ensure_deps() -> None:
    """Install PyTorchWildlife optional deps that are not declared in its METADATA."""
    extras = [
        "setuptools>=65,<81",
        "soundfile>=0.12",
        "librosa>=0.10",
        "transformers>=4.40,<5",
    ]
    print("Checking ML dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *extras])


def main() -> None:
    _ensure_deps()

    from src.config import app_config

    print("Downloading MegaDetector V6...")
    from PytorchWildlife.models import detection as pw_detection

    pw_detection.MegaDetectorV6(
        device=app_config.models.device,
        version=app_config.models.megadetector_version,
    )
    print(f"MegaDetector ready ({app_config.models.megadetector_version}).")

    print("Downloading MiewID-msv3...")
    from transformers import AutoModel

    AutoModel.from_pretrained(app_config.models.miewid_model, trust_remote_code=True)
    print("MiewID ready.")

    print("\nAll models downloaded successfully.")


if __name__ == "__main__":
    main()
