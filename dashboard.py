"""Streamlit dashboard for human review and system monitoring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings, app_config
from src.db.models import ImageRecord, MatchReview, get_session, init_db
from src.db.repository import Repository
from src.matching.catalogue import TigerCatalogue
from src.matching.review_queue import ReviewQueue
from src.pipeline.run import TigerTrackingPipeline

st.set_page_config(page_title="Pench Tiger Tracking | Viksit Nagpur", page_icon="🐯", layout="wide")
init_db()

reserve = app_config.reserve


def get_repo():
    return Repository(get_session())


st.title("🐯 Pench Tiger Reserve — Camera Trap Intelligence")
st.caption(f"{reserve.hackathon} · {reserve.name} · Individual ID · Occupancy · Alerts")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Run Pipeline", "Tigers", "Review Queue", "Alerts", "Maps & Exports",
])

with tab1:
    st.header("Process Camera Trap Images")
    input_dir = st.text_input("Input directory", value=str(settings.data_dir / "raw" / "demo"))
    station_csv = st.text_input("Station registry CSV", value=str(settings.data_dir / "stations.csv"))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate Demo Data", type="secondary"):
            from scripts.seed_demo_data import generate_demo_data
            generate_demo_data(Path(input_dir))
            st.success("Demo data generated!")

    with col2:
        if st.button("Run Pipeline", type="primary"):
            with st.spinner("Processing..."):
                pipeline = TigerTrackingPipeline()
                try:
                    registry = Path(station_csv) if Path(station_csv).exists() else None
                    report = pipeline.run(Path(input_dir), registry)
                finally:
                    pipeline.close()

            st.success("Pipeline complete!")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Frames", report.total_frames)
            c2.metric("Blank Removed", report.blank_removed)
            c3.metric("Tigers Detected", report.tiger_detected)
            c4.metric("Alerts", report.alerts_raised)

            st.subheader("Blank Filter Report")
            st.write(f"**Space saved:** {report.bytes_quarantined / (1024*1024):.2f} MB")
            st.write(f"**Time saved:** {report.estimated_time_saved_sec / 60:.1f} minutes")

            st.subheader("Matching Results")
            m1, m2, m3 = st.columns(3)
            m1.metric("Auto-matched", report.auto_matched)
            m2.metric("New Individuals", report.new_individuals)
            m3.metric("Pending Review", report.pending_review)

            if report.exports:
                st.subheader("Exports")
                st.json(report.exports)

with tab2:
    st.header("Known Individuals")
    repo = get_repo()
    tigers = repo.list_tigers()
    if tigers:
        rows = []
        for t in tigers:
            sightings = repo.get_sightings_for_tiger(t.id)
            stations = repo.stations_used_by_tiger(t.id)
            rows.append({
                "Code": t.tiger_code,
                "Enrolled": t.enrolled_at.strftime("%Y-%m-%d"),
                "Sightings": len(sightings),
                "Stations": ", ".join(sorted(stations)),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        selected = st.selectbox("View sightings for", [t.tiger_code for t in tigers])
        tiger = next(t for t in tigers if t.tiger_code == selected)
        sightings = repo.get_sightings_for_tiger(tiger.id)
        if sightings:
            st.dataframe(pd.DataFrame([{
                "Station": s.station_id,
                "Date": s.captured_at,
                "Lat": s.latitude,
                "Lon": s.longitude,
                "Confidence": s.match_confidence,
                "Type": s.match_type,
            } for s in sightings]), use_container_width=True)
    else:
        st.info("No tigers enrolled yet. Run the pipeline first.")

with tab3:
    st.header("Ambiguous Matches — Human Review")
    repo = get_repo()
    reviews = repo.get_pending_reviews()
    if not reviews:
        st.info("No pending reviews.")
    else:
        for review in reviews:
            img = repo.session.get(ImageRecord, review.image_id)
            candidate = repo.get_tiger(review.candidate_tiger_id) if review.candidate_tiger_id else None

            with st.expander(
                f"Review #{review.id} — Confidence: {review.confidence:.2f} "
                f"→ {candidate.tiger_code if candidate else 'Unknown'}"
            ):
                col_a, col_b = st.columns(2)
                if img and img.flank_path and Path(img.flank_path).exists():
                    col_a.image(img.flank_path, caption="New capture (flank)")
                if candidate:
                    reps = repo.get_representatives(candidate.id)
                    if reps and Path(reps[0].flank_path).exists():
                        col_b.image(reps[0].flank_path, caption=f"Candidate: {candidate.tiger_code}")

                c1, c2, c3 = st.columns(3)
                catalogue = TigerCatalogue(repo)
                queue = ReviewQueue(repo, catalogue)

                if c1.button("Same Tiger", key=f"same_{review.id}"):
                    if candidate:
                        queue.resolve_same(
                            review.id, candidate.id, img.id, img.flank_path or "",
                            img.run_id or 0, img.station_id, img.captured_at,
                            img.latitude, img.longitude, review.confidence,
                        )
                        st.success(f"Matched to {candidate.tiger_code}")
                        st.rerun()

                if c2.button("New Individual", key=f"new_{review.id}"):
                    new_id = queue.resolve_new(
                        review.id, img.id, img.flank_path or "",
                        img.run_id or 0, img.station_id, img.captured_at,
                        img.latitude, img.longitude,
                    )
                    st.success(f"Enrolled as new tiger (ID: {new_id})")
                    st.rerun()

                if c3.button("Reject", key=f"reject_{review.id}"):
                    queue.resolve_reject(review.id)
                    st.warning("Rejected")
                    st.rerun()

with tab4:
    st.header("Deviation Alerts")
    repo = get_repo()
    runs = repo.get_runs(limit=5)
    if runs:
        run_id = st.selectbox("Run", [r.id for r in runs], format_func=lambda x: f"Run #{x}")
        alerts = repo.get_alerts(run_id=run_id)
        if alerts:
            for alert in alerts:
                icon = "⚠️" if not alert.is_survey_artifact else "📊"
                color = {"high": "red", "medium": "orange", "low": "blue"}.get(alert.severity, "gray")
                st.markdown(
                    f"{icon} **{alert.title}** "
                    f"— Confidence: {alert.confidence:.0%} "
                    f"— Severity: :{color}[{alert.severity}]"
                )
                st.write(alert.description)
                if alert.is_survey_artifact:
                    st.caption("Likely survey artefact (low capture count)")
                if alert.evidence_json:
                    with st.expander("Evidence"):
                        st.json(json.loads(alert.evidence_json))
                st.divider()
        else:
            st.info("No alerts for this run.")
    else:
        st.info("No processing runs yet.")

with tab5:
    st.header("Occupancy Maps & Exports")
    repo = get_repo()
    runs = repo.get_runs(limit=5)
    if runs:
        run_id = st.selectbox("Select run", [r.id for r in runs], key="map_run")
        exports_dir = settings.data_dir / "exports" / f"run_{run_id}"
        map_path = exports_dir / "occupancy_map.html"

        if map_path.exists():
            with open(map_path) as f:
                st.components.v1.html(f.read(), height=600, scrolling=True)
        else:
            st.info("No map generated for this run.")

        snaps = repo.get_occupancy_for_run(run_id)
        if snaps:
            rows = []
            for s in snaps:
                tiger = repo.get_tiger(s.tiger_id)
                rows.append({
                    "Tiger": tiger.tiger_code if tiger else "?",
                    "Centroid Lat": s.centroid_lat,
                    "Centroid Lon": s.centroid_lon,
                    "Area (sq km)": s.area_sq_km,
                    "Captures": s.capture_count,
                    "Stations": s.station_ids,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        overlap_path = exports_dir / "territorial_overlaps.json"
        if overlap_path.exists():
            with open(overlap_path) as f:
                overlaps = json.load(f)
            if overlaps:
                st.subheader("Territorial Overlaps")
                st.dataframe(pd.DataFrame(overlaps), use_container_width=True)
    else:
        st.info("Run the pipeline first to generate maps.")
