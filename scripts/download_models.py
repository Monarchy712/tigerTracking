#!/usr/bin/env python3
"""One-time download of MegaDetector and MiewID model weights."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    print("Downloading MegaDetector V6...")
    from PytorchWildlife.models import detection as pw_detection

    pw_detection.MegaDetectorV6()
    print("MegaDetector ready.")

    print("Downloading MiewID-msv3...")
    from transformers import AutoModel

    AutoModel.from_pretrained("conservationxlabs/miewid-msv3", trust_remote_code=True)
    print("MiewID ready.")

    print("\nAll models downloaded successfully.")


if __name__ == "__main__":
    main()
