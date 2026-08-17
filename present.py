"""
Judge presentation UI — Viksit Nagpur Hackathon

Launch:
  streamlit run present.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import app_config, settings
from src.db.models import get_session, init_db
from src.db.repository import Repository
from src.demo.identify_service import IdentifyService
from src.ml.model_registry import ml_available, warmup_models

init_db()

RESERVE = app_config.reserve
STATIONS_CSV = settings.data_dir / "stations_pench.csv"
if not STATIONS_CSV.exists():
    STATIONS_CSV = settings.data_dir / "stations.csv"

st.set_page_config(
    page_title="Pench Tiger ID | Judge Demo",
    page_icon="🐯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(135deg, #1a472a 0%, #2d5a27 50%, #4a3728 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { color: white !important; margin: 0; font-size: 2.2rem; }
    .main-header p { color: #e8f5e9; margin: 0.5rem 0 0 0; font-size: 1.05rem; }
    .result-card {
        background: #f8faf8;
        border: 1px solid #c8e6c9;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .tiger-id {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1b5e20;
        letter-spacing: 2px;
    }
    .confidence-pill {
        display: inline-block;
        background: #2e7d32;
        color: white;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .camera-card {
        background: linear-gradient(135deg, #fff8e1 0%, #fff3e0 100%);
        border-left: 5px solid #ef6c00;
        padding: 1.25rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
    }
    .camera-card h3 { color: #e65100; margin: 0 0 0.5rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="main-header">
  <h1>🐯 Pench Tiger Reserve — Individual Identification</h1>
  <p>{RESERVE.hackathon} · Upload a camera-trap photo → recognize the tiger → see last camera device sighting</p>
</div>
""",
    unsafe_allow_html=True,
)

if "models_ready" not in st.session_state:
    if ml_available():
        with st.spinner("Loading MegaDetector + MiewID models (one-time)..."):
            warmup_models()
        st.session_state.models_ready = True
    else:
        st.session_state.models_ready = False

if not st.session_state.get("models_ready"):
    st.warning("ML models not loaded — run `python scripts/download_models.py` first.")

repo = Repository(get_session())
service = IdentifyService(repo)
tiger_count = len(repo.list_tigers())

col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
col_stat1.metric("Known Tigers", tiger_count)
col_stat2.metric("Reserve", "Pench")
col_stat3.metric("Models", "MegaDetector + MiewID")
col_stat4.metric("Match threshold", f"{app_config.matching.auto_match_threshold:.0%}")

if tiger_count == 0:
    st.info(
        "**First time?** Build the catalogue first:\n\n"
        "`python scripts/run_pipeline.py -i data/raw/imported -s data/stations_pench.csv`"
    )

tab_identify, tab_compare, tab_history = st.tabs(
    ["🔍 Identify Tiger", "↔️ Compare Two Images", "📋 Sighting History"]
)


def _render_map(lat: float, lon: float, station_id: str, zone: str | None) -> None:
    zone_colors = {"core": "green", "buffer": "orange", "village_adjacent": "red"}
    color = zone_colors.get(zone or "core", "blue")
    m = folium.Map(location=[lat, lon], zoom_start=13, tiles="CartoDB positron")
    folium.Marker(
        [lat, lon],
        popup=f"Camera {station_id} ({zone or 'core'})",
        tooltip=f"Last seen: {station_id}",
        icon=folium.Icon(color=color, icon="camera", prefix="fa"),
    ).add_to(m)
    folium.CircleMarker(
        [lat, lon],
        radius=12,
        color="#d84315",
        fill=True,
        fill_color="#ff5722",
        fill_opacity=0.7,
        popup="Last tiger sighting",
    ).add_to(m)
    st_folium(m, width=None, height=380)


def _action_badge(action: str) -> str:
    labels = {
        "auto_match": "✅ Recognized — same individual",
        "review": "⚠️ Ambiguous — needs review",
        "enroll": "🆕 New individual enrolled",
        "no_tiger": "❌ No tiger detected",
    }
    return labels.get(action, action)


with tab_identify:
    st.subheader("Upload a camera-trap image")
    st.caption("Detect tiger → extract stripe pattern → match identity → show last camera device.")

    uploaded = st.file_uploader(
        "Choose image",
        type=["jpg", "jpeg", "png"],
        key="identify_upload",
        label_visibility="collapsed",
    )
    record = st.checkbox(
        "Record sighting at camera station (updates last-known device location)",
        value=True,
    )

    if uploaded and st.button("Identify Tiger", type="primary", use_container_width=True):
        with st.spinner("Detecting → stripe fingerprint → matching..."):
            path = service.save_upload(uploaded.name, uploaded.getvalue())
            result = service.identify_file(
                path,
                STATIONS_CSV if STATIONS_CSV.exists() else None,
                record_sighting=record,
            )
            st.session_state.identify_result = result
            st.session_state.identify_path = str(path)
            st.session_state.identify_upload_name = uploaded.name

    result = st.session_state.get("identify_result")
    path_str = st.session_state.get("identify_path")
    same_upload = (
        uploaded
        and st.session_state.get("identify_upload_name") == uploaded.name
    )

    if result and path_str and same_upload:
        path = Path(path_str)
        if not result.has_tiger:
            st.error(result.message)
        else:
            left, right = st.columns([1, 1])

            with left:
                if path.exists():
                    st.image(str(path), caption="Uploaded image", use_container_width=True)
                if result.flank_path and Path(result.flank_path).exists():
                    st.image(
                        result.flank_path,
                        caption="Flank stripe region (MiewID input)",
                        use_container_width=True,
                    )

            with right:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown("**Identification result**")
                code = result.tiger_code or "NEW"
                st.markdown(f'<div class="tiger-id">{code}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<span class="confidence-pill">{result.confidence:.1%} match confidence</span>',
                    unsafe_allow_html=True,
                )
                st.write(_action_badge(result.action))
                st.write(result.message)
                st.metric("Sightings on record", result.total_sightings)
                st.markdown("</div>", unsafe_allow_html=True)

                if result.last_station_id and result.last_latitude and result.last_longitude:
                    st.markdown('<div class="camera-card">', unsafe_allow_html=True)
                    st.markdown("### 📷 Last spotted on camera device")
                    st.markdown(f"**Station ID:** `{result.last_station_id}`")
                    if result.last_zone:
                        st.markdown(f"**Zone:** {result.last_zone.replace('_', ' ').title()}")
                    if result.last_captured_at:
                        st.markdown(
                            f"**When:** {result.last_captured_at.strftime('%d %b %Y, %H:%M UTC')}"
                        )
                    st.markdown(
                        f"**GPS:** {result.last_latitude:.4f}, {result.last_longitude:.4f}"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                elif result.action == "enroll":
                    st.success("First sighting recorded for this new individual.")

            if result.last_latitude and result.last_longitude:
                st.subheader("Last known location — reserve map")
                _render_map(
                    result.last_latitude,
                    result.last_longitude,
                    result.last_station_id or "?",
                    result.last_zone,
                )

with tab_compare:
    st.subheader("Are these two photos the same tiger?")
    c1, c2 = st.columns(2)
    with c1:
        img_a = st.file_uploader("Image A", type=["jpg", "jpeg", "png"], key="cmp_a")
        if img_a:
            st.image(img_a, use_container_width=True)
    with c2:
        img_b = st.file_uploader("Image B", type=["jpg", "jpeg", "png"], key="cmp_b")
        if img_b:
            st.image(img_b, use_container_width=True)

    if img_a and img_b and st.button("Compare Stripe Patterns", type="primary"):
        with st.spinner("Comparing embeddings..."):
            path_a = service.save_upload(img_a.name, img_a.getvalue())
            path_b = service.save_upload(img_b.name, img_b.getvalue())
            cmp = service.compare_files(path_a, path_b)
            st.session_state.compare_result = cmp
            st.session_state.compare_names = (img_a.name, img_b.name)

    cmp = st.session_state.get("compare_result")
    same_pair = (
        img_a
        and img_b
        and st.session_state.get("compare_names") == (img_a.name, img_b.name)
    )

    if cmp and same_pair:
        verdict = cmp.get("verdict", "error")
        color_map = {"same": "green", "uncertain": "orange", "different": "red", "error": "red"}
        color = color_map.get(verdict, "gray")
        st.markdown(f"### :{color}[{cmp.get('message', 'Comparison failed')}]")

        if verdict != "error":
            st.metric("Stripe similarity", f"{cmp.get('confidence', 0):.1%}")
        else:
            st.warning(
                "Make sure both images clearly show a tiger flank/body. "
                "Try photos from your Amur Tigers dataset."
            )

        fc1, fc2 = st.columns(2)
        if cmp.get("flank_a") and Path(cmp["flank_a"]).exists():
            fc1.image(cmp["flank_a"], caption="Flank A", use_container_width=True)
        if cmp.get("flank_b") and Path(cmp["flank_b"]).exists():
            fc2.image(cmp["flank_b"], caption="Flank B", use_container_width=True)

with tab_history:
    st.subheader("Tiger catalogue — last camera sighting")
    tigers = repo.list_tigers()
    if not tigers:
        st.info("No tigers yet — run the batch pipeline first.")
    else:
        selected = st.selectbox("Select tiger", [t.tiger_code for t in tigers])
        tiger = next(t for t in tigers if t.tiger_code == selected)
        last = repo.get_last_sighting(tiger.id)
        sightings = repo.get_sightings_for_tiger(tiger.id)

        m1, m2, m3 = st.columns(3)
        m1.metric("Tiger ID", tiger.tiger_code)
        m2.metric("Total sightings", len(sightings))
        m3.metric("Last camera device", last.station_id if last and last.station_id else "—")

        if last and last.latitude and last.longitude:
            zone = None
            if last.station_id:
                st_obj = repo.get_station(last.station_id)
                zone = st_obj.zone if st_obj else None
            st.markdown(f"**Last seen at station `{last.station_id}`** — {last.captured_at or ''}")
            _render_map(last.latitude, last.longitude, last.station_id or "?", zone)

        if sightings:
            st.dataframe(
                pd.DataFrame([
                    {
                        "Camera device": s.station_id,
                        "Timestamp": s.captured_at,
                        "Latitude": s.latitude,
                        "Longitude": s.longitude,
                        "Confidence": f"{s.match_confidence:.1%}" if s.match_confidence else "—",
                        "Type": s.match_type,
                    }
                    for s in reversed(sightings[-20:])
                ]),
                use_container_width=True,
            )

st.markdown("---")
st.caption("MegaDetector V6 + MiewID-msv3 · Pench Tiger Reserve · Viksit Nagpur Hackathon")
