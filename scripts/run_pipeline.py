#!/usr/bin/env python3
"""CLI entry point for running the tiger tracking pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.run import TigerTrackingPipeline


def main():
    parser = argparse.ArgumentParser(description="Tiger Tracking Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Raw image directory")
    parser.add_argument("--stations", "-s", help="Station registry CSV path")
    parser.add_argument("--no-recursive", action="store_true", help="Don't scan subdirectories")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    station_registry = Path(args.stations) if args.stations else None

    print(f"Starting pipeline on {input_dir}...")
    pipeline = TigerTrackingPipeline()
    try:
        report = pipeline.run(input_dir, station_registry, recursive=not args.no_recursive)
    finally:
        pipeline.close()

    print("\n=== Pipeline Complete ===")
    print(f"Run ID:              {report.run_id}")
    print(f"Total frames:        {report.total_frames}")
    print(f"Blank removed:       {report.blank_removed}")
    print(f"Space saved:         {report.bytes_quarantined / (1024*1024):.2f} MB")
    print(f"Time saved:          {report.estimated_time_saved_sec / 60:.1f} min")
    print(f"Tigers detected:     {report.tiger_detected}")
    print(f"Auto-matched:        {report.auto_matched}")
    print(f"New individuals:     {report.new_individuals}")
    print(f"Pending review:      {report.pending_review}")
    print(f"Occupancy computed:  {report.occupancy_count}")
    print(f"Alerts raised:       {report.alerts_raised}")
    print(f"\nExports:")
    print(json.dumps(report.exports, indent=2))


if __name__ == "__main__":
    main()
