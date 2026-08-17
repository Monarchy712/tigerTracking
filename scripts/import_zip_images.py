#!/usr/bin/env python3
"""
Import tiger images from a ZIP archive for pipeline processing.

Usage:
  # Inspect only (no extract)
  python scripts/import_zip_images.py --zip /path/to/tigers.zip --inspect

  # Extract all images
  python scripts/import_zip_images.py --zip /path/to/tigers.zip

  # Extract first 200 images (recommended for hackathon demo on CPU)
  python scripts/import_zip_images.py --zip /path/to/tigers.zip --limit 200

  # Then run pipeline
  python scripts/run_pipeline.py -i data/raw/imported -s data/stations_pench.csv
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTENSIONS


def inspect_zip(zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if not m.endswith("/") and _is_image(m)]

    top_folders = Counter()
    for m in members:
        parts = Path(m).parts
        if len(parts) > 1:
            top_folders[parts[0]] += 1
        else:
            top_folders["(root)"] += 1

    return {
        "total_images": len(members),
        "top_folders": top_folders.most_common(10),
        "sample_paths": members[:5],
    }


def extract_zip(zip_path: Path, output_dir: Path, limit: int | None = None) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if not m.endswith("/") and _is_image(m)]
        if limit is not None:
            members = members[:limit]

        for member in members:
            src = Path(member)
            # Flatten nested paths to avoid overly deep dirs; keep subfolder prefix for ID hints
            if len(src.parts) > 1:
                dest_name = "__".join(src.parts)
            else:
                dest_name = src.name

            dest = output_dir / dest_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src_file, open(dest, "wb") as out_file:
                shutil.copyfileobj(src_file, out_file)
            extracted += 1

    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description="Import tiger images from ZIP")
    parser.add_argument("--zip", required=True, help="Path to ZIP file")
    parser.add_argument(
        "--output",
        default="data/raw/imported",
        help="Directory to extract images into",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max images to extract (use 100-300 for quick demo on CPU)",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Only inspect ZIP structure, do not extract",
    )
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        print(f"Error: ZIP not found: {zip_path}")
        print("\nCopy your file into the project first, e.g.:")
        print("  cp /path/to/tigers.zip data/tigers.zip")
        sys.exit(1)

    info = inspect_zip(zip_path)
    print(f"ZIP: {zip_path}")
    print(f"Total images: {info['total_images']}")
    print("\nTop folders (may indicate individual tiger IDs):")
    for folder, count in info["top_folders"]:
        print(f"  {folder}: {count} images")
    print("\nSample paths:")
    for p in info["sample_paths"]:
        print(f"  {p}")

    if args.inspect:
        return

    output_dir = Path(args.output)
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"\nWarning: {output_dir} is not empty. Files may be overwritten.")

    print(f"\nExtracting to {output_dir}...")
    count = extract_zip(zip_path, output_dir, limit=args.limit)
    print(f"Extracted {count} images.")

    print("\nNext steps:")
    print(f"  python scripts/run_pipeline.py -i {output_dir} -s data/stations_pench.csv")
    if info["total_images"] > 500:
        print("\nTip: 5000 images on CPU can take many hours.")
        print("For the hackathon demo, use --limit 200 first, then scale up.")


if __name__ == "__main__":
    main()
