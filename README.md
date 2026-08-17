# Tiger Tracking System

End-to-end camera trap pipeline for tiger reserves: blank image filtering, individual identification via stripe pattern matching, occupancy mapping, and behavioural deviation alerting.

Uses **MegaDetector V6** (detection + blank filtering) and **MiewID-msv3** (individual tiger re-identification via stripe embeddings).

## Features

| Requirement | Implementation |
|---|---|
| **i. Blank filtering** | MegaDetector V6 (hybrid with heuristic pre-screen); quarantine folder (reversible delete); reports frames removed, MB saved, time saved |
| **ii. Individual ID** | MegaDetector bbox → flank crop → MiewID embedding → auto-match / human review / new enrollment → SQLite database |
| **iii. Occupancy mapping** | Convex hull home range per tiger; centroid + area; Folium map + GeoJSON + CSV export; territorial overlap detection |
| **iv. Deviation alerts** | Range shifts, new stations, buffer/village movement, prolonged absence; survey artefact flagging; confidence scores |

## Quick Start

```bash
# 1. Setup (creates venv, installs deps, downloads models, runs demo)
bash setup.sh

# Or manual steps:
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py      # ~250 MB, one-time
python scripts/seed_demo_data.py
python scripts/run_pipeline.py -i data/raw/demo -s data/stations.csv

# 2. Launch dashboard
streamlit run dashboard.py

# 3. Launch API
uvicorn src.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

## ML Models (no training required)

| Stage | Model | Weights |
|-------|-------|---------|
| Blank filter + detection | MegaDetector V6 | Auto-download via `PytorchWildlife` |
| Individual ID | MiewID-msv3 | [Hugging Face](https://huggingface.co/conservationxlabs/miewid-msv3) |

Download weights once:

```bash
python scripts/download_models.py
```

Verify:

```bash
python scripts/smoke_test.py
```

### Fallback mode

If ML dependencies are missing, the pipeline automatically falls back to:
- Heuristic blank filter + detection
- Mock pairwise matching (demo data with `tiger_alpha_*` prefixes)

Set in `.env`:

```env
USE_ML_MODELS=true
FRIEND_MODEL_MOCK=false
```

## Architecture

```
Raw Images → Blank Filter (MegaDetector) → Quarantine (reversible)
                ↓ retained
           Tiger Detect (MegaDetector) + Flank Crop
                ↓
           MiewID Embedding ← catalogue representatives
                ↓
     ┌──────────┼──────────┐
  auto-match  review    enroll
     ↓           ↓         ↓
  SQLite DB (tigers, sightings, images, stations)
     ↓
  Occupancy Engine → Map / GeoJSON / CSV
     ↓
  Alert Engine → deviation detection
```

### Confidence thresholds (config.yaml)

- `≥ 0.85` → auto-match to existing individual
- `0.60 – 0.85` → human review queue
- `< 0.60` → enroll as new individual

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/pipeline/run` | Run full pipeline on image directory |
| GET | `/api/v1/tigers` | List all known individuals |
| GET | `/api/v1/tigers/{id}/sightings` | Sightings for one tiger |
| GET | `/api/v1/reviews/pending` | Ambiguous matches needing review |
| POST | `/api/v1/reviews/{id}/resolve` | Human reviewer decision |
| GET | `/api/v1/runs/{id}/occupancy` | Occupancy data for a run |
| GET | `/api/v1/runs/{id}/alerts` | Alerts for a run |
| GET | `/api/v1/exports/{id}/map` | Download occupancy map HTML |
| GET | `/api/v1/exports/{id}/mstripes` | Download M-STrIPES-aligned bundle (zip) |
| POST | `/api/v1/quarantine/restore/{id}` | Restore quarantined blanks |

## Configuration

Edit `config.yaml`:

```yaml
models:
  enabled: true
  megadetector_threshold: 0.20
  miewid_model: "conservationxlabs/miewid-msv3"
  device: "cpu"

blank_filter:
  mode: hybrid   # heuristic | megadetector | hybrid

matching:
  auto_match_threshold: 0.85
  review_threshold: 0.60

alerts:
  core_range_shift_sqkm: 17.5
  buffer_range_shift_km: 5.0
  absence_days_threshold: 30
```

## Station Registry

CSV format for camera trap stations:

```csv
station_id,latitude,longitude,zone
CAM01,23.710,81.025,core
CAM04,23.680,81.030,buffer
CAM06,23.668,81.005,village_adjacent
```

Zones: `core`, `buffer`, `village_adjacent`

## Project Structure

```
src/
├── integrations/      # M-STrIPES-aligned forest department export
├── ml/                # MegaDetector + MiewID model registry
├── pipeline/          # ingest, blank filter, detect/crop, orchestrator
├── matching/          # MiewID embedding matcher, catalogue, review queue
├── occupancy/         # home range, map export
├── alerts/            # deviation detection
├── db/                # SQLAlchemy models + repository
└── api/               # FastAPI routes
scripts/
├── download_models.py # one-time weight download
├── smoke_test.py      # verify ML stages
├── run_pipeline.py    # CLI
└── seed_demo_data.py  # demo image generator
dashboard.py           # Streamlit UI
```

## Database

SQLite at `data/tiger_tracking.db`. Queryable tables:

- `tigers` — individual catalogue
- `tiger_representatives` — flank crops + MiewID embeddings
- `sightings` — tiger ↔ image ↔ station ↔ GPS ↔ timestamp
- `images` — all processed frames with metadata
- `occupancy_snapshots` — per-run home range per tiger
- `alerts` — deviation alerts with evidence
- `match_reviews` — pending human decisions

## Exports (per run)

Generated in `data/exports/run_{id}/`:

- `occupancy_map.html` — interactive map for forest department
- `home_ranges.geojson` — GIS-compatible home ranges
- `occupancy_report.csv` — tabular summary
- `territorial_overlaps.json` — overlap between individuals
- `mstripes/` — M-STrIPES-aligned bundle (see below)

## M-STrIPES-aligned export

`data/exports/run_{id}/mstripes/` holds the camera-trap data in the Phase IV
layout the M-STrIPES ecological module is built around. It is **not** an
officially certified M-STrIPES import bundle — no public import schema exists.
Column names live in `DEPLOYMENT_COLUMNS` / `CAPTURE_COLUMNS` in
`src/integrations/mstripes.py` and can be remapped in one place once a real
departmental sheet is available.

| File | Contents |
|------|----------|
| `camera_deployment.csv` | One row per trap: coordinates, zone, effort dates, trap nights |
| `capture_records.csv` | One row per identified capture: trap, timestamp, individual ID, confidence |
| `capture_history.csv` | Individuals × sampling occasions (0/1) — SECR / CAPTURE / MARK input |
| `trap_effort.csv` | Traps × occasions (1 = camera active) |
| `population_summary.csv` | Minimum count, trap nights, recapture rate, Chapman-corrected Lincoln-Petersen estimate |
| `alerts.gpx` | Alerts, waterholes and village-interface stations as GPX waypoints for a ranger handset |
| `README_MSTRIPES.txt` / `manifest.json` | Definitions used, and the fields the pipeline cannot derive |

Definitions: a **sampling occasion** is one calendar day between the first and
last capture; a trap counts as **active** on a day if any frame from it carries
that date, blanks included, since a blank frame still proves the camera ran.
Capture history spans the whole database, not one run — capture-recapture needs
the full survey block.

`Flank`, `Sex`, `AgeClass` and `CameraMake` are exported as `U`/blank rather
than guessed. Optional `range_name`, `beat` and `compartment` columns in the
station registry CSV are carried through when present.

## License

MIT — Hackathon project
