#!/usr/bin/env python3
"""Generate a two-survey sample dataset that exercises every alert path.

Unlike ``seed_demo_data.py`` (a handful of generic frames), this builds a full
demo story: ten stations across a reserve, six individuals, and a second survey
month whose deviations trigger each detector on cue.

    python scripts/generate_sample_data.py

Then, from the repo root:

    python scripts/run_pipeline.py -i data/raw/sample_run1 -s data/sample_stations.csv
    python scripts/run_pipeline.py -i data/raw/sample_run2 -s data/sample_stations.csv

Scripted deviations in survey 2:

  * ``tiger_delta``   relocates ~18 km south      -> range_shift
  * ``tiger_bravo``   appears at a brand new cam  -> new_station
  * ``tiger_charlie`` stops appearing entirely    -> prolonged_absence
  * ``tiger_echo``    walks a village-edge camera -> buffer_movement
  * ``tiger_alpha`` + ``tiger_foxtrot`` share a station within hours
                                                  -> anomaly:mating
  * the waterhole camera sees a capture surge     -> anomaly:water
  * ``tiger_foxtrot`` photographs with an abnormal
    posture and a wound-coloured flank            -> anomaly:injured
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent

# Pench Tiger Reserve, Nagpur region — the reserve this project is built around.
# (station_id, name, lat, lon, zone, sensitive, waterhole)
STATIONS = [
    ("PEN01", "Alikatta North",      21.785, 79.310, "core",             False, False),
    ("PEN02", "Chhindimatta",        21.778, 79.325, "core",             False, False),
    ("PEN03", "Pipariya Nala",       21.765, 79.335, "core",             False, True),
    ("PEN04", "Bodhanala Ridge",     21.758, 79.318, "core",             False, False),
    ("PEN05", "Karmajhiri Buffer",   21.752, 79.340, "buffer",           False, False),
    ("PEN06", "Turia Gate",          21.745, 79.305, "buffer",           True,  False),
    ("PEN07", "Sillari Village Edge",21.738, 79.328, "village_adjacent", True,  False),
    ("PEN08", "Totladoh Backwater",  21.792, 79.345, "core",             False, False),
    ("PEN09", "Rukhad Corridor",     21.800, 79.300, "core",             False, False),
    # ~18 km south of the core cluster: the destination for the range-shift story.
    ("PEN10", "Khawasa South Beat",  21.610, 79.330, "buffer",           False, False),
]

TIGERS = [
    # (prefix, base coat colour, stripe colour, home stations)
    ("tiger_alpha",   (46, 96, 150), (18, 34, 52), ["PEN01", "PEN02", "PEN09"]),
    ("tiger_bravo",   (52, 104, 158), (22, 38, 58), ["PEN02", "PEN03"]),
    ("tiger_charlie", (40, 88, 142), (16, 30, 48), ["PEN03", "PEN04"]),
    ("tiger_delta",   (56, 110, 164), (24, 42, 62), ["PEN04", "PEN05"]),
    ("tiger_echo",    (44, 92, 146), (20, 36, 54), ["PEN05", "PEN08"]),
    ("tiger_foxtrot", (50, 100, 154), (21, 39, 59), ["PEN08", "PEN09"]),
]

STATION_BY_ID = {s[0]: s for s in STATIONS}
WIDTH, HEIGHT = 640, 480


# --- frame rendering -----------------------------------------------------


def _forest_background(rng: random.Random) -> np.ndarray:
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for y in range(HEIGHT):
        shade = int(30 + 45 * (y / HEIGHT))
        img[y, :] = [shade // 3, shade // 2 + 18, shade // 4 + 12]

    for _ in range(14):  # trunks and undergrowth give the frame real edge content
        x = rng.randint(0, WIDTH - 1)
        w = rng.randint(6, 20)
        cv2.rectangle(img, (x, 0), (x + w, HEIGHT), (24, 40, 30), -1)
    for _ in range(70):
        cx, cy = rng.randint(0, WIDTH - 1), rng.randint(HEIGHT // 2, HEIGHT - 1)
        cv2.circle(img, (cx, cy), rng.randint(3, 11), (18, 58, 26), -1)

    noise = np.random.randint(0, 14, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    return cv2.add(img, noise)


def _draw_tiger(
    img: np.ndarray,
    coat,
    stripe,
    rng: random.Random,
    injured: bool = False,
    second_animal: bool = False,
) -> None:
    """Draw a stylised tiger. Injured frames get a flattened body and a wound."""
    body_w = int(WIDTH * (0.62 if injured else 0.42))
    body_h = int(HEIGHT * (0.16 if injured else 0.30))
    bx = rng.randint(40, max(41, WIDTH - body_w - 40))
    by = rng.randint(HEIGHT // 3, max(HEIGHT // 3 + 1, HEIGHT - body_h - 40))

    cv2.ellipse(img, (bx + body_w // 2, by + body_h // 2),
                (body_w // 2, body_h // 2), 0, 0, 360, coat, -1)
    for _ in range(rng.randint(12, 18)):
        sx = bx + rng.randint(8, max(9, body_w - 12))
        sy = by + rng.randint(4, max(5, body_h - 6))
        cv2.line(img, (sx, sy), (sx + rng.randint(4, 14), sy + rng.randint(-6, 6)), stripe, 3)

    head_x = min(WIDTH - 30, bx + body_w + 12)
    head_y = by + body_h // 3
    cv2.circle(img, (head_x, head_y), 26, coat, -1)
    cv2.circle(img, (head_x + 9, head_y - 6), 4, (10, 10, 10), -1)

    if injured:
        # A dark-red wound over the flank region the crop is taken from, large
        # enough to clear the blood-fraction threshold in the anomaly detector.
        fx = bx + int(body_w * 0.35)
        fy = by + int(body_h * 0.25)
        fw = int(body_w * 0.45)
        fh = int(body_h * 0.55)
        cv2.ellipse(img, (fx + fw // 2, fy + fh // 2),
                    (max(8, fw // 2), max(6, fh // 2)), 0, 0, 360, (28, 22, 120), -1)

    if second_animal:
        ox = max(10, bx - int(body_w * 0.75))
        oy = min(HEIGHT - 60, by + 26)
        cv2.ellipse(img, (ox + body_w // 3, oy + body_h // 2),
                    (body_w // 3, body_h // 2), 0, 0, 360, coat, -1)
        cv2.circle(img, (min(WIDTH - 20, ox + int(body_w * 0.72)), oy + body_h // 3), 20, coat, -1)


def _blank_frame(rng: random.Random) -> np.ndarray:
    """An empty night frame: low variance, low edge density, low entropy."""
    base = rng.randint(38, 54)
    img = np.full((HEIGHT, WIDTH, 3), base, dtype=np.uint8)
    return cv2.add(img, np.random.randint(0, 7, (HEIGHT, WIDTH, 3), dtype=np.uint8))


# --- capture scheduling --------------------------------------------------


def _filename(prefix: str, station_id: str, when: datetime, index: int) -> str:
    # "<tiger>_station_<STATION>_<YYYYMMDD>_<HHMMSS>_<n>" — the "_station_" marker
    # is what the offline mock matcher keys on to group frames per individual.
    return f"{prefix}_station_{station_id}_{when:%Y%m%d}_{when:%H%M%S}_{index:03d}.jpg"


def _write_capture(
    out_dir: Path,
    prefix: str,
    coat,
    stripe,
    station_id: str,
    when: datetime,
    index: int,
    rng: random.Random,
    injured: bool = False,
    second_animal: bool = False,
) -> None:
    img = _forest_background(rng)
    _draw_tiger(img, coat, stripe, rng, injured=injured, second_animal=second_animal)
    cv2.imwrite(str(out_dir / _filename(prefix, station_id, when, index)), img)


def _write_blank(out_dir: Path, station_id: str, when: datetime, index: int, rng: random.Random) -> None:
    name = f"blank_station_{station_id}_{when:%Y%m%d}_{when:%H%M%S}_{index:03d}.jpg"
    cv2.imwrite(str(out_dir / name), _blank_frame(rng))


def _survey_one(out_dir: Path, anchor: datetime, rng: random.Random) -> int:
    """Baseline month: every individual works its own home stations."""
    count = 0
    for prefix, coat, stripe, stations in TIGERS:
        for day in range(0, 15):
            for station_id in stations:
                if rng.random() > 0.55:
                    continue
                when = anchor + timedelta(days=day, hours=rng.randint(4, 21),
                                          minutes=rng.randint(0, 59))
                _write_capture(out_dir, prefix, coat, stripe, station_id, when, count, rng)
                count += 1

    for i in range(28):  # blank frames for the filter to strip
        station_id = rng.choice(STATIONS)[0]
        when = anchor + timedelta(days=rng.randint(0, 14), hours=rng.randint(0, 23))
        _write_blank(out_dir, station_id, when, 900 + i, rng)
        count += 1
    return count


def _survey_two(out_dir: Path, anchor: datetime, rng: random.Random) -> int:
    """Second month, with each deviation scripted so the alerts fire on demand."""
    coats = {t[0]: (t[1], t[2]) for t in TIGERS}
    count = 0

    for prefix, coat, stripe, stations in TIGERS:
        if prefix == "tiger_charlie":
            continue  # absent this survey -> prolonged_absence
        if prefix == "tiger_delta":
            stations = ["PEN10"]  # relocated ~18 km south -> range_shift
        if prefix == "tiger_bravo":
            stations = stations + ["PEN06"]  # never used before -> new_station
        if prefix == "tiger_echo":
            stations = stations + ["PEN07"]  # village edge -> buffer_movement

        for day in range(0, 14):
            for station_id in stations:
                if rng.random() > 0.6:
                    continue
                when = anchor + timedelta(days=day, hours=rng.randint(4, 21),
                                          minutes=rng.randint(0, 59))
                # foxtrot is the injured individual: abnormal posture + wound colour
                injured = prefix == "tiger_foxtrot" and rng.random() < 0.7
                _write_capture(out_dir, prefix, coat, stripe, station_id, when, count, rng,
                               injured=injured)
                count += 1

    # Co-occurrence: alpha and foxtrot at one station three hours apart.
    meeting = anchor + timedelta(days=6, hours=19)
    alpha_coat, alpha_stripe = coats["tiger_alpha"]
    fox_coat, fox_stripe = coats["tiger_foxtrot"]
    _write_capture(out_dir, "tiger_alpha", alpha_coat, alpha_stripe, "PEN09", meeting, count, rng)
    count += 1
    _write_capture(out_dir, "tiger_foxtrot", fox_coat, fox_stripe, "PEN09",
                   meeting + timedelta(hours=3), count, rng, second_animal=True)
    count += 1

    # Waterhole surge at PEN03: a burst well above its own baseline rate.
    for i in range(16):
        prefix, coat, stripe, _ = TIGERS[i % len(TIGERS)]
        if prefix == "tiger_charlie":
            prefix, coat, stripe, _ = TIGERS[0]
        when = anchor + timedelta(days=8 + i % 5, hours=11 + i % 6, minutes=rng.randint(0, 59))
        _write_capture(out_dir, prefix, coat, stripe, "PEN03", when, count, rng)
        count += 1

    for i in range(22):
        station_id = rng.choice(STATIONS)[0]
        when = anchor + timedelta(days=rng.randint(0, 13), hours=rng.randint(0, 23))
        _write_blank(out_dir, station_id, when, 900 + i, rng)
        count += 1
    return count


def write_station_registry(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["station_id", "name", "latitude", "longitude", "zone", "is_sensitive", "is_waterhole"]
        )
        for sid, name, lat, lon, zone, sensitive, waterhole in STATIONS:
            writer.writerow([sid, name, lat, lon, zone, str(sensitive).lower(), str(waterhole).lower()])
    return path


def generate(output_root: Path, registry_path: Path, seed: int = 7, clean: bool = True) -> dict:
    rng = random.Random(seed)
    np.random.seed(seed)

    # Timestamps are anchored to today so the absence detector, which compares
    # against the current date, has something realistic to measure.
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    run1_anchor = today - timedelta(days=75)
    run2_anchor = today - timedelta(days=20)

    run1_dir = output_root / "sample_run1"
    run2_dir = output_root / "sample_run2"
    for d in (run1_dir, run2_dir):
        if clean and d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    counts = {
        "run1": _survey_one(run1_dir, run1_anchor, rng),
        "run2": _survey_two(run2_dir, run2_anchor, rng),
    }
    write_station_registry(registry_path)

    return {
        "run1_dir": run1_dir,
        "run2_dir": run2_dir,
        "registry": registry_path,
        "run1_frames": counts["run1"],
        "run2_frames": counts["run2"],
        "run1_window": (run1_anchor, run1_anchor + timedelta(days=15)),
        "run2_window": (run2_anchor, run2_anchor + timedelta(days=14)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=ROOT / "data" / "raw",
                        help="Directory to write sample_run1/ and sample_run2/ into")
    parser.add_argument("-s", "--stations", type=Path,
                        default=ROOT / "data" / "sample_stations.csv",
                        help="Where to write the station registry CSV")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    result = generate(args.output, args.stations, seed=args.seed)

    print(f"Survey 1: {result['run1_frames']:>4} frames -> {result['run1_dir']}")
    print(f"Survey 2: {result['run2_frames']:>4} frames -> {result['run2_dir']}")
    print(f"Stations: {len(STATIONS)} -> {result['registry']}")
    print("\nRun the pipeline twice, in order:")
    print(f"  python scripts/run_pipeline.py -i {result['run1_dir']} -s {result['registry']}")
    print(f"  python scripts/run_pipeline.py -i {result['run2_dir']} -s {result['registry']}")


if __name__ == "__main__":
    main()
