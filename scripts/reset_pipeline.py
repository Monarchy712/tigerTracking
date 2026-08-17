#!/usr/bin/env python3
"""Reset database and quarantine data for a clean pipeline rerun."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings


def main() -> None:
    db_path = settings.data_dir / "tiger_tracking.db"
    quarantine = settings.data_dir / "quarantine"
    flanks = settings.data_dir / "processed" / "flanks"
    exports = settings.data_dir / "exports"

    if db_path.exists():
        db_path.unlink()
        print(f"Removed {db_path}")

    for folder in (quarantine, flanks, exports):
        if folder.exists():
            shutil.rmtree(folder)
            print(f"Removed {folder}")

    print("Reset complete. Re-seed and rerun the pipeline:")
    print("  python scripts/seed_demo_data.py")
    print("  python scripts/run_pipeline.py -i data/raw/demo -s data/stations.csv")


if __name__ == "__main__":
    main()
