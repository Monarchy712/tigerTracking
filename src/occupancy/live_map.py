"""Date-filtered map rendering for the dashboard.

``map_export`` writes the per-run map that ships to the forest department as a
standalone HTML file. This module serves the live dashboard instead: it takes a
date window, recomputes territories for exactly the sightings inside it, and
renders either the per-tiger territory view or a reserve-wide heatmap.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import folium
from folium.plugins import HeatMap

from src.config import app_config
from src.occupancy.home_range import compute_occupancy
from src.occupancy.map_export import MAP_ATTR, MAP_TILES

# Colour-key individuals consistently across the territory view, the profile
# mini-map and the PDF report.
TIGER_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]


def color_for(index: int) -> str:
    return TIGER_COLORS[index % len(TIGER_COLORS)]


def sightings_to_rows(sightings: list, code_by_tiger_id: dict[int, str]) -> list[dict]:
    """Flatten ORM sightings into cache-friendly plain dicts."""
    rows = []
    for s in sightings:
        if s.latitude is None or s.longitude is None:
            continue
        rows.append({
            "tiger_id": s.tiger_id,
            "tiger_code": code_by_tiger_id.get(s.tiger_id, f"#{s.tiger_id}"),
            "lat": s.latitude,
            "lng": s.longitude,
            "station_id": s.station_id,
            "captured_at": s.captured_at,
            "confidence": s.match_confidence,
        })
    return rows


def territories_from_rows(rows: list[dict]) -> list[dict]:
    """Recompute one home range per individual from the filtered rows."""
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["tiger_id"], []).append(row)

    cfg = app_config.occupancy
    territories = []
    for tiger_id, group in sorted(grouped.items()):
        stand_ins = [
            SimpleNamespace(
                latitude=r["lat"], longitude=r["lng"], station_id=r["station_id"]
            )
            for r in group
        ]
        result = compute_occupancy(
            stand_ins, min_points=cfg.min_points_for_range, method=cfg.home_range_method
        )
        if not result:
            continue
        territories.append({
            "tiger_id": tiger_id,
            "tiger_code": group[0]["tiger_code"],
            "centroid_lat": result.centroid_lat,
            "centroid_lon": result.centroid_lon,
            "area_sq_km": result.area_sq_km,
            "capture_count": len(group),
            "station_ids": result.station_ids,
            "home_range_geojson": result.home_range_geojson,
        })
    return territories


def _base_map(rows: list[dict]) -> folium.Map:
    reserve = app_config.reserve
    if rows:
        center = [
            sum(r["lat"] for r in rows) / len(rows),
            sum(r["lng"] for r in rows) / len(rows),
        ]
    else:
        center = [reserve.center_lat, reserve.center_lon]
    return folium.Map(location=center, zoom_start=11, tiles=MAP_TILES, attr=MAP_ATTR)


def _add_stations(m: folium.Map, stations: list) -> None:
    if not stations:
        return
    group = folium.FeatureGroup(name="Camera Stations", show=False)
    zone_colors = {"core": "blue", "buffer": "orange", "village_adjacent": "red"}
    for st in stations:
        tags = []
        if getattr(st, "is_waterhole", False):
            tags.append("waterhole")
        if getattr(st, "is_sensitive", False):
            tags.append("sensitive")
        suffix = f" — {', '.join(tags)}" if tags else ""
        folium.Marker(
            [st.latitude, st.longitude],
            popup=f"Station {st.station_id} ({st.zone}){suffix}",
            icon=folium.Icon(
                color=zone_colors.get(st.zone, "gray"), icon="camera", prefix="fa"
            ),
        ).add_to(group)
    group.add_to(m)


def build_map(
    rows: list[dict],
    territories: list[dict],
    stations: list | None = None,
    heatmap: bool = False,
    focus_tiger_id: int | None = None,
) -> folium.Map:
    """Render the map for the current date window.

    With ``heatmap`` on, all captures across all individuals are drawn as a
    single density surface. With it off, each individual gets a colour-keyed
    home-range polygon, a ringed centroid and one dot per capture.
    """
    if focus_tiger_id is not None:
        rows = [r for r in rows if r["tiger_id"] == focus_tiger_id]
        territories = [t for t in territories if t["tiger_id"] == focus_tiger_id]

    m = _base_map(rows)

    if heatmap:
        if rows:
            HeatMap(
                [[r["lat"], r["lng"]] for r in rows],
                radius=18,
                blur=22,
                min_opacity=0.35,
                name="Reserve activity",
            ).add_to(m)
        _add_stations(m, stations or [])
        folium.LayerControl(collapsed=True).add_to(m)
        return m

    rows_by_tiger: dict[int, list[dict]] = {}
    for row in rows:
        rows_by_tiger.setdefault(row["tiger_id"], []).append(row)

    for index, territory in enumerate(territories):
        color = color_for(index)
        code = territory["tiger_code"]
        group = folium.FeatureGroup(name=code)

        geojson_str = territory.get("home_range_geojson")
        if geojson_str:
            try:
                gj = json.loads(geojson_str)
            except json.JSONDecodeError:
                gj = None
            if gj and gj.get("geometry", {}).get("type") == "Polygon":
                folium.GeoJson(
                    gj,
                    style_function=lambda _x, c=color: {
                        "fillColor": c,
                        "color": c,
                        "weight": 2,
                        "fillOpacity": 0.22,
                    },
                    tooltip=f"{code}: {territory.get('area_sq_km', '?')} sq km",
                ).add_to(group)

        for row in rows_by_tiger.get(territory["tiger_id"], []):
            folium.CircleMarker(
                [row["lat"], row["lng"]],
                radius=3,
                color=color,
                fill=True,
                fill_opacity=0.85,
                weight=1,
                tooltip=(
                    f"{code} · {row['station_id'] or 'no station'} · "
                    f"{_fmt_date(row['captured_at'])}"
                ),
            ).add_to(group)

        lat, lon = territory.get("centroid_lat"), territory.get("centroid_lon")
        if lat is not None and lon is not None:
            # Ringed dot: white halo underneath so the centroid reads over dots.
            folium.CircleMarker(
                [lat, lon], radius=10, color="#ffffff", weight=3, fill=False
            ).add_to(group)
            folium.CircleMarker(
                [lat, lon],
                radius=7,
                color=color,
                weight=2,
                fill=True,
                fill_opacity=0.95,
                popup=folium.Popup(
                    f"<b>{code}</b><br>"
                    f"Captures: {territory.get('capture_count', 0)}<br>"
                    f"Home range: {territory.get('area_sq_km', '?')} sq km<br>"
                    f"Stations: {', '.join(territory.get('station_ids', [])) or '—'}",
                    max_width=260,
                ),
            ).add_to(group)

        group.add_to(m)

    _add_stations(m, stations or [])
    folium.LayerControl(collapsed=True).add_to(m)
    return m


def _fmt_date(value) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    return str(value or "undated")


def render_static_map_png(
    rows: list[dict],
    territories: list[dict],
    output_path: Path,
    stations: list | None = None,
) -> Path | None:
    """Matplotlib rendering of the territory view, for embedding in the PDF.

    Folium output is HTML, which needs a headless browser to rasterise. A plain
    scatter plot in lat/lon needs no extra binaries and prints legibly.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
    except ImportError:
        return None

    if not rows and not territories:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=150)

    for index, territory in enumerate(territories):
        color = color_for(index)
        geojson_str = territory.get("home_range_geojson")
        if geojson_str:
            try:
                gj = json.loads(geojson_str)
                geom = gj.get("geometry", gj)
                if geom.get("type") == "Polygon":
                    coords = [(lon, lat) for lon, lat in geom["coordinates"][0]]
                    ax.add_patch(
                        MplPolygon(
                            coords, closed=True, facecolor=color, edgecolor=color,
                            alpha=0.20, linewidth=1.5,
                        )
                    )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass

        pts = [r for r in rows if r["tiger_id"] == territory["tiger_id"]]
        if pts:
            ax.scatter(
                [p["lng"] for p in pts], [p["lat"] for p in pts],
                s=12, color=color, alpha=0.8, zorder=3,
            )
        lat, lon = territory.get("centroid_lat"), territory.get("centroid_lon")
        if lat is not None and lon is not None:
            ax.scatter(
                [lon], [lat], s=110, color=color, edgecolor="white",
                linewidth=1.8, zorder=4, label=territory["tiger_code"],
            )

    if stations:
        ax.scatter(
            [s.longitude for s in stations], [s.latitude for s in stations],
            s=26, marker="^", color="#555555", alpha=0.7, zorder=2, label="Stations",
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(app_config.reserve.name)
    ax.grid(alpha=0.2, linestyle=":")
    ax.set_aspect("equal", adjustable="datalim")
    if territories:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9, ncol=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
