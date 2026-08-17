"""Streamlit dashboard: ingestion, occupancy, individual profiles, review and alerts."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import app_config, settings
from src.db.models import ImageRecord, get_session, init_db
from src.db.repository import Repository
from src.matching.catalogue import TigerCatalogue
from src.matching.review_queue import ReviewQueue
from src.occupancy.live_map import build_map, sightings_to_rows, territories_from_rows
from src.pipeline.run import TigerTrackingPipeline
from src.reporting.pdf_report import generate_survey_report

st.set_page_config(
    page_title="Pench Tiger Tracking | Viksit Nagpur", page_icon="🐯", layout="wide"
)
init_db()

reserve = app_config.reserve
ALERT_ICONS = {
    "range_shift": "🧭",
    "new_station": "📍",
    "buffer_movement": "🏘️",
    "prolonged_absence": "🕰️",
    "anomaly:injured": "🩸",
    "anomaly:mating": "💞",
    "anomaly:water": "💧",
}


def get_repo() -> Repository:
    return Repository(get_session())


def data_version() -> int:
    """Cache key that changes whenever the pipeline writes new sightings."""
    return get_repo().count_totals()["sightings"]


@st.cache_data(show_spinner=False)
def load_rows(version: int) -> list[dict]:
    repo = get_repo()
    codes = {t.id: t.tiger_code for t in repo.list_tigers()}
    return sightings_to_rows(repo.get_sightings_in_range(), codes)


@st.cache_data(show_spinner=False)
def load_territories(version: int, start: datetime, end: datetime) -> list[dict]:
    """Home ranges recomputed for just the scrubbed window."""
    rows = [r for r in load_rows(version) if r["captured_at"] and start <= r["captured_at"] <= end]
    return territories_from_rows(rows)


def rows_in_window(version: int, start: datetime, end: datetime) -> list[dict]:
    return [
        r for r in load_rows(version)
        if r["captured_at"] and start <= r["captured_at"] <= end
    ]


def alert_label(alert) -> str:
    icon = ALERT_ICONS.get(alert.alert_type, "⚠️")
    return f"{icon} {alert.title}"


def render_alert(alert, repo: Repository, show_evidence: bool = True) -> None:
    color = {"high": "red", "medium": "orange", "low": "blue"}.get(alert.severity, "gray")
    st.markdown(
        f"{alert_label(alert)} — :{color}[{alert.severity}] · "
        f"confidence {alert.confidence:.0%} · `{alert.alert_type}`"
    )
    st.write(alert.description)
    if alert.is_survey_artifact:
        st.caption("⚑ Flagged as a likely survey artefact (low capture count)")
    if show_evidence and alert.evidence_json:
        with st.expander("Evidence"):
            st.json(json.loads(alert.evidence_json))


# --------------------------------------------------------------------------
# Sidebar — ingestion, report, run history
# --------------------------------------------------------------------------

st.sidebar.title("🐯 Tiger Tracking")
st.sidebar.caption(f"{reserve.name} · {reserve.state}")

st.sidebar.subheader("Ingest camera-trap folder")
input_dir = st.sidebar.text_input(
    "Image folder", value=str(settings.data_dir / "raw" / "sample_run1")
)
station_csv = st.sidebar.text_input(
    "Station registry CSV", value=str(settings.data_dir / "sample_stations.csv")
)

if st.sidebar.button("Run pipeline", type="primary", use_container_width=True):
    progress = st.sidebar.progress(0, text="Starting pipeline…")
    pipeline = TigerTrackingPipeline()
    try:
        progress.progress(30, text="Filtering blanks and identifying individuals…")
        registry = Path(station_csv) if Path(station_csv).exists() else None
        report = pipeline.run(Path(input_dir), registry)
        progress.progress(100, text="Done")
    finally:
        pipeline.close()
    st.cache_data.clear()
    st.session_state["last_report"] = {
        "run_id": report.run_id,
        "total_frames": report.total_frames,
        "blank_removed": report.blank_removed,
        "tiger_detected": report.tiger_detected,
        "auto_matched": report.auto_matched,
        "new_individuals": report.new_individuals,
        "pending_review": report.pending_review,
        "alerts_raised": report.alerts_raised,
        "anomalies_raised": report.anomalies_raised,
        "bytes_quarantined": report.bytes_quarantined,
        "estimated_time_saved_sec": report.estimated_time_saved_sec,
    }
    st.sidebar.success(f"Run #{report.run_id} complete")

if st.sidebar.button("Generate survey report", use_container_width=True):
    with st.spinner("Building PDF…"):
        pdf_path = generate_survey_report(get_repo())
    st.session_state["report_pdf"] = str(pdf_path)

if st.session_state.get("report_pdf"):
    pdf_path = Path(st.session_state["report_pdf"])
    if pdf_path.exists():
        st.sidebar.download_button(
            "⬇ Download PDF",
            data=pdf_path.read_bytes(),
            file_name=pdf_path.name,
            mime="application/pdf",
            use_container_width=True,
        )

st.sidebar.divider()
sidebar_repo = get_repo()
runs = sidebar_repo.get_runs(limit=20)
if runs:
    st.sidebar.subheader("Run history")
    selected_run_id = st.sidebar.selectbox(
        "Processing run",
        [r.id for r in runs],
        format_func=lambda rid: f"Run #{rid}",
        key="sidebar_run",
    )
    run = sidebar_repo.get_run(selected_run_id)
    if run:
        st.sidebar.caption(
            f"{run.started_at:%d %b %Y %H:%M} · {run.total_frames} frames · "
            f"{run.blank_removed} blanks removed"
        )
        st.sidebar.caption(f"Source: `{Path(run.input_dir).name}`")
else:
    selected_run_id = None
    st.sidebar.info("No runs yet — point at a folder and run the pipeline.")

# --------------------------------------------------------------------------

st.title("Camera Trap Intelligence")
st.caption(
    f"{reserve.hackathon} · individual identification · occupancy · deviation alerts"
)

version = data_version()
repo = get_repo()
totals = repo.count_totals()

tab_dashboard, tab_map, tab_tigers, tab_review, tab_alerts = st.tabs(
    ["Dashboard", "Map", "Tigers", "Review Queue", "Alerts"]
)

# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------

with tab_dashboard:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Individuals", totals["tigers"])
    c2.metric("Captures", totals["sightings"])
    c3.metric("Frames processed", totals["images"])
    c4.metric("Blanks removed", totals["blanks"])
    c5.metric("Alerts", totals["alerts"])

    last = st.session_state.get("last_report")
    if last:
        st.subheader(f"Latest run — #{last['run_id']}")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Frames in", last["total_frames"])
        d2.metric("Blanks quarantined", last["blank_removed"])
        d3.metric("Auto-matched", last["auto_matched"])
        d4.metric("New individuals", last["new_individuals"])
        st.caption(
            f"Storage reclaimed: {last['bytes_quarantined'] / (1024 * 1024):.2f} MB · "
            f"Reviewer time saved: {last['estimated_time_saved_sec'] / 60:.1f} min · "
            f"{last['pending_review']} awaiting review · "
            f"{last['anomalies_raised']} of {last['alerts_raised']} alerts were anomalies"
        )
    elif runs:
        latest = runs[0]
        st.subheader(f"Most recent run — #{latest.id}")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Frames in", latest.total_frames)
        d2.metric("Blanks quarantined", latest.blank_removed)
        d3.metric("Auto-matched", latest.auto_matched)
        d4.metric("New individuals", latest.new_individuals)

    st.subheader("Highest-priority alerts")
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    top_alerts = sorted(
        repo.get_alerts(),
        key=lambda a: (severity_rank.get(a.severity, 3), -a.confidence),
    )[:5]
    if not top_alerts:
        st.info("No alerts yet. Process a second survey to surface deviations.")
    for alert in top_alerts:
        render_alert(alert, repo, show_evidence=False)
        st.divider()

# --------------------------------------------------------------------------
# Map — heatmap toggle + timeline scrubber
# --------------------------------------------------------------------------

with tab_map:
    all_rows = load_rows(version)
    dated = [r["captured_at"] for r in all_rows if r["captured_at"]]

    if not dated:
        st.info("No georeferenced captures yet. Run the pipeline to populate the map.")
    else:
        min_date, max_date = min(dated).date(), max(dated).date()

        left, right = st.columns([3, 2])
        with left:
            heatmap_on = st.toggle("Show reserve-wide activity heatmap", value=False)
        with right:
            st.caption(
                "Heatmap pools every individual's captures. Territory view colour-keys "
                "each individual's convex-hull home range."
            )

        if min_date == max_date:
            window = (min_date, max_date)
            st.caption(f"All captures fall on {min_date:%d %b %Y}.")
        else:
            window = st.slider(
                "Capture window",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                format="DD MMM YYYY",
            )

        start = datetime.combine(window[0], datetime.min.time())
        end = datetime.combine(window[1], datetime.max.time())

        rows = rows_in_window(version, start, end)
        territories = load_territories(version, start, end)

        m1, m2, m3 = st.columns(3)
        m1.metric("Captures in window", len(rows))
        m2.metric("Individuals in window", len({r["tiger_id"] for r in rows}))
        m3.metric("Stations active", len({r["station_id"] for r in rows if r["station_id"]}))

        folium_map = build_map(
            rows, territories, stations=repo.get_stations(), heatmap=heatmap_on
        )
        st_folium(folium_map, height=560, use_container_width=True, returned_objects=[])

        if territories and not heatmap_on:
            st.dataframe(
                pd.DataFrame([{
                    "Tiger": t["tiger_code"],
                    "Captures": t["capture_count"],
                    "Home range (km²)": t["area_sq_km"],
                    "Stations": ", ".join(sorted(t["station_ids"])),
                } for t in territories]),
                use_container_width=True,
                hide_index=True,
            )

# --------------------------------------------------------------------------
# Tigers — list and individual profile
# --------------------------------------------------------------------------

with tab_tigers:
    tigers = repo.list_tigers()
    if not tigers:
        st.info("No individuals enrolled yet. Run the pipeline first.")
    else:
        summary = []
        for tiger in tigers:
            sightings = repo.get_sightings_for_tiger(tiger.id)
            dates = [s.captured_at for s in sightings if s.captured_at]
            summary.append({
                "Code": tiger.tiger_code,
                "Captures": len(sightings),
                "Stations": len(repo.stations_used_by_tiger(tiger.id)),
                "First seen": min(dates).date() if dates else None,
                "Last seen": max(dates).date() if dates else None,
                "Alerts": len(repo.get_alerts_for_tiger(tiger.id)),
            })
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

        st.divider()
        codes = [t.tiger_code for t in tigers]
        selected_code = st.selectbox("Open profile", codes, key="profile_pick")
        tiger = next(t for t in tigers if t.tiger_code == selected_code)

        sightings = repo.get_sightings_for_tiger(tiger.id)
        dates = sorted(s.captured_at for s in sightings if s.captured_at)
        recent_alerts = repo.get_alerts_for_tiger(
            tiger.id, since=datetime.utcnow() - timedelta(days=30)
        )
        all_alerts = repo.get_alerts_for_tiger(tiger.id)

        st.header(f"🐯 {tiger.tiger_code}")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("First seen", f"{dates[0]:%d %b %Y}" if dates else "—")
        p2.metric("Last seen", f"{dates[-1]:%d %b %Y}" if dates else "—")
        p3.metric("Total captures", len(sightings))
        p4.metric("Alerts (30 days)", len(recent_alerts))

        prof_photos, prof_map = st.columns([3, 2])

        with prof_photos:
            st.subheader("Captures")
            images = [
                img for img in repo.get_images_for_tiger(tiger.id)
                if img.flank_path and Path(img.flank_path).exists()
            ]
            if not images:
                st.caption("No flank crops on disk for this individual.")
            else:
                per_page = 12
                pages = (len(images) + per_page - 1) // per_page
                page = 1
                if pages > 1:
                    page = st.number_input(
                        f"Page (1–{pages}, {len(images)} crops)",
                        min_value=1, max_value=pages, value=1, step=1,
                        key=f"page_{tiger.id}",
                    )
                chunk = images[(page - 1) * per_page: page * per_page]
                for row_start in range(0, len(chunk), 4):
                    cols = st.columns(4)
                    for col, img in zip(cols, chunk[row_start: row_start + 4]):
                        station = img.station_id or "—"
                        caption = (
                            f"{station} · {img.captured_at:%d %b}"
                            if img.captured_at
                            else station
                        )
                        col.image(img.flank_path, caption=caption, use_container_width=True)

        with prof_map:
            st.subheader("Territory")
            profile_rows = [r for r in load_rows(version) if r["tiger_id"] == tiger.id]
            profile_territory = territories_from_rows(profile_rows)
            if profile_rows:
                st_folium(
                    build_map(profile_rows, profile_territory, focus_tiger_id=tiger.id),
                    height=320,
                    use_container_width=True,
                    returned_objects=[],
                    key=f"map_{tiger.id}",
                )
                if profile_territory:
                    t = profile_territory[0]
                    st.caption(
                        f"Home range {t['area_sq_km']} km² across "
                        f"{len(t['station_ids'])} stations."
                    )
            else:
                st.caption("No georeferenced captures for this individual.")

        st.subheader("Movement — unique stations visited per week")
        if dates:
            frame = pd.DataFrame([
                {"week": pd.Timestamp(s.captured_at).to_period("W").start_time,
                 "station": s.station_id}
                for s in sightings if s.captured_at and s.station_id
            ])
            if not frame.empty:
                weekly = frame.groupby("week")["station"].nunique()
                st.line_chart(weekly, height=220)
            else:
                st.caption("No station-tagged captures to chart.")
        else:
            st.caption("No dated captures to chart.")

        st.subheader("Alert history")
        if not all_alerts:
            st.caption("No alerts have ever fired for this individual.")
        else:
            st.dataframe(
                pd.DataFrame([{
                    "When": a.created_at,
                    "Type": a.alert_type,
                    "Severity": a.severity,
                    "Confidence": f"{a.confidence:.0%}",
                    "Detail": a.description,
                } for a in all_alerts]),
                use_container_width=True,
                hide_index=True,
            )

# --------------------------------------------------------------------------
# Review queue
# --------------------------------------------------------------------------

with tab_review:
    st.header("Ambiguous matches — human review")
    reviews = repo.get_pending_reviews()
    if not reviews:
        st.success("Nothing pending. Every match cleared the auto-assign threshold.")
    else:
        st.caption(
            f"{len(reviews)} matches landed between the review "
            f"({app_config.matching.review_threshold:.0%}) and auto-match "
            f"({app_config.matching.auto_match_threshold:.0%}) thresholds."
        )
        for review in reviews:
            img = repo.session.get(ImageRecord, review.image_id)
            candidate = (
                repo.get_tiger(review.candidate_tiger_id)
                if review.candidate_tiger_id else None
            )
            header = (
                f"Review #{review.id} — similarity {review.confidence:.2f} → "
                f"{candidate.tiger_code if candidate else 'unknown'}"
            )
            with st.expander(header, expanded=True):
                col_a, col_b = st.columns(2)
                if img and img.flank_path and Path(img.flank_path).exists():
                    col_a.image(img.flank_path, caption="New capture (flank crop)")
                if candidate:
                    reps = repo.get_representatives(candidate.id)
                    if reps and Path(reps[0].flank_path).exists():
                        col_b.image(
                            reps[0].flank_path, caption=f"Catalogue: {candidate.tiger_code}"
                        )

                catalogue = TigerCatalogue(repo)
                queue = ReviewQueue(repo, catalogue)
                b1, b2, b3 = st.columns(3)

                if b1.button("✅ Same tiger", key=f"same_{review.id}") and candidate and img:
                    queue.resolve_same(
                        review.id, candidate.id, img.id, img.flank_path or "",
                        img.run_id or 0, img.station_id, img.captured_at,
                        img.latitude, img.longitude, review.confidence,
                    )
                    st.cache_data.clear()
                    st.rerun()

                if b2.button("🆕 New individual", key=f"new_{review.id}") and img:
                    queue.resolve_new(
                        review.id, img.id, img.flank_path or "",
                        img.run_id or 0, img.station_id, img.captured_at,
                        img.latitude, img.longitude,
                    )
                    st.cache_data.clear()
                    st.rerun()

                if b3.button("❌ Reject", key=f"reject_{review.id}"):
                    queue.resolve_reject(review.id)
                    st.cache_data.clear()
                    st.rerun()

# --------------------------------------------------------------------------
# Alerts — full filterable log
# --------------------------------------------------------------------------

with tab_alerts:
    st.header("Alert log")
    alerts = repo.get_alerts()
    if not alerts:
        st.info("No alerts yet.")
    else:
        code_by_id = {t.id: t.tiger_code for t in repo.list_tigers()}
        f1, f2, f3 = st.columns(3)
        types = sorted({a.alert_type for a in alerts})
        chosen_types = f1.multiselect("Type", types, default=types)
        individuals = sorted({code_by_id.get(a.tiger_id, "—") for a in alerts})
        chosen_individuals = f2.multiselect("Individual", individuals, default=individuals)
        run_options = ["All runs"] + [f"Run #{r.id}" for r in runs]
        chosen_run = f3.selectbox("Run", run_options)

        filtered = [
            a for a in alerts
            if a.alert_type in chosen_types
            and code_by_id.get(a.tiger_id, "—") in chosen_individuals
            and (chosen_run == "All runs" or a.run_id == int(chosen_run.split("#")[1]))
        ]

        st.caption(f"{len(filtered)} of {len(alerts)} alerts shown.")
        for alert in filtered:
            render_alert(alert, repo)
            st.divider()
