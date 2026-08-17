#!/usr/bin/env python3
"""
Extract images from a tiger PDF and prepare them for pipeline ingestion.

Usage:
  python scripts/import_pdf_tigers.py --pdf /path/to/tigers.pdf --output data/raw/imported

The PDF is useful when it contains camera-trap or flank photos of individual tigers.
Each extracted page/image can then be processed by the main pipeline to build the catalogue.

If the PDF is a text report (e.g. All-India tiger census with 5155 total count),
it cannot be used for stripe matching — use it only as background context in your pitch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def extract_with_pymupdf(pdf_path: Path, output_dir: Path) -> int:
    import fitz  # PyMuPDF

    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    count = 0
    for page_idx in range(len(doc)):
        page = doc.load_page(page_idx)
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base = doc.extract_image(xref)
            ext = base.get("ext", "png")
            out = output_dir / f"pdf_p{page_idx:04d}_img{img_idx:02d}.{ext}"
            out.write_bytes(base["image"])
            count += 1
    doc.close()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract tiger images from PDF")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--output", default="data/raw/imported", help="Output directory")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    output_dir = Path(args.output)

    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    try:
        import fitz  # noqa: F401
    except ImportError:
        print("Installing PyMuPDF...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pymupdf"])
        import fitz  # noqa: F401

    count = extract_with_pymupdf(pdf_path, output_dir)
    print(f"Extracted {count} images to {output_dir}")
    if count == 0:
        print(
            "No embedded images found. The PDF may be text-only (census report).\n"
            "For identification you need camera-trap JPG/PNG folders, not a text PDF."
        )
    else:
        print("\nNext step:")
        print(f"  python scripts/run_pipeline.py -i {output_dir} -s data/stations_pench.csv")


if __name__ == "__main__":
    main()
