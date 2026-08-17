# Project Synopsis — Camera-Trap Tiger Intelligence System

**Event:** Viksit Nagpur Hackathon — Introductory Session  
**Date:** 17 August 2026  
**Reserve focus:** Pench Tiger Reserve (Nagpur region)  
**Team:** *[Add team name and member names]*

---

## One-line pitch

An automated pipeline that turns thousands of camera-trap photos into **actionable wildlife intelligence** — filtering blanks, identifying individual tigers by stripe pattern, mapping home ranges, and alerting rangers when behaviour deviates from history.

---

## Problem

Forest departments deploy hundreds of camera traps across reserves like Pench. Each trap generates thousands of images, most of which are empty (no animal). Rangers must manually sort frames, try to recognize the same tiger across different cameras and dates, and infer movement patterns — a slow, error-prone process that delays conservation decisions.

The hackathon problem statement asks for a system that addresses four needs:

| # | Need | Why it matters |
|---|------|----------------|
| **i** | Blank image filtering (reversible) | Saves storage, review time, and analyst fatigue |
| **ii** | Individual tiger identification | Enables long-term monitoring of specific animals |
| **iii** | Occupancy / home-range mapping | Shows where each tiger moves across the reserve |
| **iv** | Deviation alerts | Flags unusual movement before human–wildlife conflict escalates |

---

## Proposed solution

We are building an **end-to-end camera-trap intelligence platform** tailored for Indian tiger reserves:

```
Raw trap images
    → Blank filter (quarantine, reversible)
    → Tiger detection + flank crop
    → Stripe-pattern fingerprint (re-ID)
    → Persistent catalogue (SQLite DB)
    → Occupancy maps + GeoJSON/CSV exports
    → Behavioural deviation alerts
```

### Key design choices

- **No model training** — uses pretrained conservation AI models only (inference on laptop/CPU).
- **Human in the loop** — ambiguous matches go to a review queue; nothing is silently mis-labelled.
- **Reversible quarantine** — blank frames are moved aside, not permanently deleted.
- **Export-ready outputs** — interactive maps, GeoJSON, and CSV for forest GIS workflows.
- **Live demo UI** — judges and rangers can upload a photo and get an instant ID + last camera sighting.

---

## Technology stack

| Stage | Model / tool | Role |
|-------|--------------|------|
| Blank filter + detection | **MegaDetector V6** (Microsoft / PytorchWildlife) | Detect animals; discard empty frames |
| Individual ID | **MiewID-msv3** (Conservation X Labs, Hugging Face) | Stripe-pattern embeddings for same/different tiger |
| Backend | Python, FastAPI, SQLAlchemy, SQLite | Pipeline orchestration + REST API |
| Frontend | Streamlit (`present.py` for judges, `dashboard.py` for ops) | Live identification, maps, alerts |
| Maps | Folium + GeoJSON export | Home ranges and camera stations on reserve map |

---

## Current status (working prototype)

| Component | Status |
|-----------|--------|
| Blank filtering + quarantine | ✅ Implemented |
| Tiger detection + flank crop | ✅ Implemented |
| Individual ID + catalogue DB | ✅ Implemented |
| Human review queue | ✅ Implemented |
| Occupancy mapping + exports | ✅ Implemented |
| Deviation alerts | ✅ Implemented |
| Judge demo UI (identify / compare / history) | ✅ Implemented |
| Tested on ~5,000 Amur tiger images (CVWC2019 benchmark) | ✅ Pipeline runs on CPU |

**Demo command:** `streamlit run present.py` — upload a photo → tiger ID → last camera device → map pin.

---

## Expected impact

- **For rangers:** Hours of manual sorting reduced to automated batch processing with clear stats (frames removed, MB saved, time saved).
- **For researchers:** Persistent individual IDs enable population monitoring, territory overlap analysis, and longitudinal studies.
- **For reserve management:** Early alerts on range shifts, buffer-zone movement, or prolonged absence support proactive conflict prevention.
- **For Nagpur / Pench:** A locally runnable system that can be pointed at real trap data once camera station GPS is provided — no cloud dependency required for core inference.

---

## What we plan next

1. Run full pipeline on real Pench/Tadoba/Nagzira trap batches (with actual station coordinates).
2. Tune match thresholds on Indian tiger photos and validate ID accuracy with field experts.
3. Add species classification (e.g. SpeciesNet) to reduce false positives from deer/leopard.
4. Package for forest-department handoff: one-click pipeline run + export bundle.

---

## What we need from organizers / mentors

- Confirmation of target reserve(s) and access to sample camera-trap metadata (station ID, GPS, timestamp).
- Guidance on evaluation criteria for the formal hackathon phase.
- Any existing Pench camera-station registry or boundary shapefiles.

---

## Contact

**Team lead:** *[Name, phone, email]*  
**Repository:** `tigerTracking/` (local — bring laptop with demo running)

---

*This document is prepared for the introductory session (Seminar Hall, 5 PM). It describes our proposed direction; the formal hackathon evaluation will follow separately.*
