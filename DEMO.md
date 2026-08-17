# Viksit Nagpur Hackathon — Demo Guide

## Judge presentation UI (use this for judges)

```bash
source .venv/bin/activate
pip install streamlit-folium   # if not installed

# Build catalogue once from your Amur tiger zip
python scripts/import_zip_images.py --zip data/tigers.zip --limit 300
python scripts/run_pipeline.py -i data/raw/imported -s data/stations_pench.csv

# Launch clean judge UI
streamlit run present.py
```

| Tab | What to show judges |
|-----|---------------------|
| **Identify Tiger** | Upload image → tiger ID (T0003) → **last camera device** (PEN04) → map pin |
| **Compare Two Images** | Two photos → same tiger or different (live stripe match) |
| **Sighting History** | Pick any tiger → full camera sighting log |

---

## Problem statement → Our system

| # | Requirement | Where it lives | Demo action |
|---|-------------|----------------|-------------|
| **i** | Blank filtering + quarantine + stats | `src/pipeline/blank_filter.py` | Show quarantine folder + pipeline stats |
| **ii** | Tiger ID via stripes + DB | MiewID + `data/tiger_tracking.db` | Show tiger catalogue + review queue |
| **iii** | Occupancy map + exports | `data/exports/run_*/` | Open `occupancy_map.html` |
| **iv** | Deviation alerts | Dashboard → Alerts tab | Run pipeline twice, show alerts |
| **+** | Forest dept handover | `src/integrations/mstripes.py` | Dashboard → Forest Dept Export → build bundle, open `alerts.gpx` on a phone |

---

## 5-minute judge demo script

### 1. Intro (30 sec)
> "We built an automated camera-trap intelligence system for **Pench Tiger Reserve** near Nagpur. It clears blank frames, identifies individual tigers by stripe pattern, maps home ranges, and alerts rangers when behaviour changes."

### 2. Live identification demo (2 min)
```bash
streamlit run present.py
```
Tab **Identify Tiger** → upload two different images of same tiger → show same ID + last camera device.

Tab **Compare Two Images** → upload pair → show "Same tiger" with confidence.

### 2b. Batch pipeline (optional, 1 min)
```bash
streamlit run dashboard.py
```
Tab **Run Pipeline** → click **Run Pipeline** → show metrics:
- Blank removed, MB saved, time saved
- Auto-matched vs new individuals vs pending review

### 3. Individual identification (1 min)
Tab **Tigers** → show catalogue (`T0001`, `T0002`, …) with sightings linked to station, GPS, timestamp.

Tab **Review Queue** → resolve one ambiguous match live (Same Tiger / New Individual).

### 4. Occupancy map (1 min)
Tab **Maps & Exports** → show home range polygons, camera stations (core/buffer/village), territorial overlaps.

Or open: `data/exports/run_1/occupancy_map.html`

### 5. Alerts (1 min)
Tab **Alerts** → show range shift, new station, buffer movement, prolonged absence with confidence + survey-artefact flag.

### 6. Tech stack (30 sec)
- **MegaDetector V6** — blank filter + animal detection (Microsoft, pretrained)
- **MiewID-msv3** — stripe-pattern fingerprint (Wild Me / Hugging Face, pretrained)
- **No model training** — inference only

---

## Using your tiger image ZIP (~5000 images)

This is **exactly what the pipeline needs** — real photos for stripe-pattern identification.

### Step 1 — Copy ZIP into project
```bash
cp /path/to/your/tigers.zip ~/Desktop/Hackathons/tigerTracking/data/tigers.zip
```

### Step 2 — Inspect structure (folders may = individual tigers)
```bash
python scripts/import_zip_images.py --zip data/tigers.zip --inspect
```

### Step 3 — Extract (start small for demo)
```bash
# Quick demo batch (~15-30 min on CPU)
python scripts/import_zip_images.py --zip data/tigers.zip --limit 200

# Full dataset (can take hours on CPU — run overnight)
python scripts/import_zip_images.py --zip data/tigers.zip
```

### Step 4 — Run pipeline
```bash
python scripts/reset_pipeline.py          # optional clean start
python scripts/run_pipeline.py -i data/raw/imported -s data/stations_pench.csv
streamlit run dashboard.py
```

| ZIP structure | What happens |
|---------------|--------------|
| **Folder per tiger** (e.g. `tiger_001/img1.jpg`) | MiewID groups by stripe pattern; folder names help you validate |
| **Flat folder** of camera-trap shots | System auto-enrolls new IDs and grows catalogue |
| **~5000 images in `Amur Tigers/test/`** | CVWC2019 Amur Tiger benchmark — **excellent for ID** (MiewID trained on this species). No GPS in filenames → we auto-assign Pench stations for map demo. |

**Your ZIP structure:**
```
Amur Tigers/test/000000.jpg
Amur Tigers/test/000004.jpg
...
```
All 5156 images in one folder — individual IDs are **not** in folder names; the pipeline discovers them via stripe matching.

**Recommended commands for your dataset:**
```bash
python scripts/import_zip_images.py --zip data/tigers.zip --limit 300
python scripts/reset_pipeline.py
python scripts/run_pipeline.py -i data/raw/imported -s data/stations_pench.csv
python scripts/run_pipeline.py -i data/raw/imported -s data/stations_pench.csv   # 2nd run → alerts
```

---

## What we need from you to improve the demo

1. **Copy the ZIP** into `tigerTracking/data/tigers.zip`
2. Run `--inspect` and share the output (folder structure matters)
3. Real **camera station GPS** for Pench/Tadoba/Nagzira if available
4. Which **reserve** you're presenting for (Pench is default)

---

## Clean demo reset

```bash
source .venv/bin/activate
python scripts/reset_pipeline.py
python scripts/seed_demo_data.py
python scripts/run_pipeline.py -i data/raw/demo -s data/stations_pench.csv
python scripts/run_pipeline.py -i data/raw/demo -s data/stations_pench.csv   # 2nd run → alerts
streamlit run dashboard.py
```

---

## Honest limits (if judges ask)

- Detects **animal**, not species — deer could pass (SpeciesNet can be added)
- MiewID trained mainly on Amur tigers — works on Indian tigers but real photos test best
- Occupancy uses **camera station locations**, not GPS collars
- Demo data is synthetic — replace with real trap images for production credibility
