#!/usr/bin/env python3
"""Generate synthetic demo camera-trap images for hackathon demo."""

from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

STATIONS = [
    ("CAM01", 23.710, 81.025, "core"),
    ("CAM02", 23.705, 81.040, "core"),
    ("CAM03", 23.690, 81.050, "core"),
    ("CAM04", 23.680, 81.030, "buffer"),
    ("CAM05", 23.675, 81.015, "buffer"),
    ("CAM06", 23.668, 81.005, "village_adjacent"),
]

TIGER_PROFILES = [
    {"prefix": "tiger_alpha", "color": (40, 80, 120), "stripe_color": (20, 40, 60)},
    {"prefix": "tiger_beta", "color": (50, 90, 130), "stripe_color": (25, 45, 65)},
    {"prefix": "tiger_gamma", "color": (45, 85, 125), "stripe_color": (22, 42, 62)},
]


def _draw_tiger_pattern(img: np.ndarray, color, stripe_color, seed: int):
    rng = random.Random(seed)
    h, w = img.shape[:2]
    body_x, body_y = w // 4, h // 3
    body_w, body_h = w // 2, h // 3
    cv2.ellipse(img, (body_x + body_w // 2, body_y + body_h // 2), (body_w // 2, body_h // 2), 0, 0, 360, color, -1)

    for i in range(8):
        sx = body_x + rng.randint(10, body_w - 20)
        sy = body_y + rng.randint(5, body_h - 10)
        cv2.line(img, (sx, sy), (sx + rng.randint(5, 15), sy + rng.randint(-5, 5)), stripe_color, 2)

    head_x = body_x + body_w + 10
    head_y = body_y + body_h // 4
    cv2.circle(img, (head_x, head_y), 25, color, -1)
    cv2.circle(img, (head_x + 10, head_y - 5), 4, (0, 0, 0), -1)


def _blank_frame(w=640, h=480) -> np.ndarray:
    img = np.full((h, w, 3), 45, dtype=np.uint8)
    noise = np.random.randint(0, 8, (h, w, 3), dtype=np.uint8)
    return cv2.add(img, noise)


def _forest_bg(w=640, h=480) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        shade = int(30 + 40 * (y / h))
        img[y, :] = [shade // 3, shade // 2 + 20, shade // 4 + 10]
    return img


def generate_demo_data(output_dir: Path, num_days: int = 14):
    output_dir.mkdir(parents=True, exist_ok=True)
    base_date = datetime(2026, 1, 1)

    station_csv = output_dir.parent / "stations.csv"
    with open(station_csv, "w") as f:
        f.write("station_id,latitude,longitude,zone\n")
        for sid, lat, lon, zone in STATIONS:
            f.write(f"{sid},{lat},{lon},{zone}\n")

    count = 0
    for day in range(num_days):
        date = base_date + timedelta(days=day)
        for station_id, lat, lon, zone in STATIONS:
            num_frames = random.randint(2, 5)
            for frame in range(num_frames):
                ts = date.replace(hour=random.randint(0, 23), minute=random.randint(0, 59))
                is_blank = random.random() < 0.25

                if is_blank:
                    img = _blank_frame()
                    fname = f"blank_station_{station_id}_{ts.strftime('%Y%m%d_%H%M%S')}.jpg"
                else:
                    profile = random.choice(TIGER_PROFILES)
                    img = _forest_bg()
                    seed = hash(f"{profile['prefix']}_{station_id}_{day}") % 10000
                    _draw_tiger_pattern(img, profile["color"], profile["stripe_color"], seed)
                    fname = f"{profile['prefix']}_station_{station_id}_{ts.strftime('%Y%m%d_%H%M%S')}.jpg"

                path = output_dir / fname
                cv2.imwrite(str(path), img)
                count += 1

    print(f"Generated {count} demo images in {output_dir}")
    print(f"Station registry: {station_csv}")
    return output_dir, station_csv


if __name__ == "__main__":
    from src.config import settings

    demo_dir = settings.data_dir / "raw" / "demo"
    generate_demo_data(demo_dir)
