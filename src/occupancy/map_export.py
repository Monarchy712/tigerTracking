"""Map visualization and export for forest department staff."""

from __future__ import annotations

import json
from pathlib import Path

import folium
from folium import FeatureGroup

from src.config import app_config

# OpenStreetMap default tiles block local file:// opens (403, missing Referer).
# CartoDB works when opening exported HTML directly in a browser.
MAP_TILES = "CartoDB positron"
MAP_ATTR = "&copy; OpenStreetMap contributors &copy; CARTO"


def generate_occupancy_map(
    occupancy_data: list[dict],
    overlap_pairs: list[dict],
    output_path: Path,
    stations: list | None = None,
) -> Path:
    """Generate interactive Folium map with home ranges and overlaps."""
    reserve = app_config.reserve
    m = folium.Map(
        location=[reserve.center_lat, reserve.center_lon],
        zoom_start=11,
        tiles=MAP_TILES,
        attr=MAP_ATTR,
    )

    folium.Marker(
        [reserve.center_lat, reserve.center_lon],
        popup=reserve.name,
        icon=folium.Icon(color="green", icon="tree"),
    ).add_to(m)

    colors = [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    ]

    tiger_group = FeatureGroup(name="Tiger Home Ranges")
    for i, occ in enumerate(occupancy_data):
        color = colors[i % len(colors)]
        code = occ.get("tiger_code", f"T{i+1}")
        geojson_str = occ.get("home_range_geojson")
        if geojson_str:
            try:
                gj = json.loads(geojson_str)
                if gj.get("geometry", {}).get("type") == "Polygon":
                    folium.GeoJson(
                        gj,
                        name=code,
                        style_function=lambda x, c=color: {
                            "fillColor": c,
                            "color": c,
                            "weight": 2,
                            "fillOpacity": 0.25,
                        },
                        tooltip=f"{code}: {occ.get('area_sq_km', '?')} sq km",
                    ).add_to(tiger_group)
            except json.JSONDecodeError:
                pass

        lat = occ.get("centroid_lat")
        lon = occ.get("centroid_lon")
        if lat and lon:
            folium.CircleMarker(
                [lat, lon],
                radius=8,
                color=color,
                fill=True,
                fill_opacity=0.8,
                popup=(
                    f"<b>{code}</b><br>"
                    f"Captures: {occ.get('capture_count', 0)}<br>"
                    f"Area: {occ.get('area_sq_km', '?')} sq km<br>"
                    f"Stations: {', '.join(occ.get('station_ids', []))}"
                ),
            ).add_to(tiger_group)

    tiger_group.add_to(m)

    if overlap_pairs:
        overlap_group = FeatureGroup(name="Territorial Overlaps")
        for pair in overlap_pairs:
            if pair.get("overlap_pct", 0) >= app_config.occupancy.overlap_threshold_pct:
                folium.PolyLine(
                    locations=[
                        [pair["tiger_a_lat"], pair["tiger_a_lon"]],
                        [pair["tiger_b_lat"], pair["tiger_b_lon"]],
                    ],
                    color="red",
                    weight=3,
                    opacity=0.7,
                    tooltip=f"{pair['tiger_a']} ↔ {pair['tiger_b']}: {pair['overlap_pct']}% overlap",
                ).add_to(overlap_group)
        overlap_group.add_to(m)

    if stations:
        station_group = FeatureGroup(name="Camera Stations")
        zone_colors = {"core": "blue", "buffer": "orange", "village_adjacent": "red"}
        for st in stations:
            zone = getattr(st, "zone", "core")
            folium.Marker(
                [st.latitude, st.longitude],
                popup=f"Station {st.station_id} ({zone})",
                icon=folium.Icon(color=zone_colors.get(zone, "gray"), icon="camera", prefix="fa"),
            ).add_to(station_group)
        station_group.add_to(m)

    folium.LayerControl().add_to(m)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    return output_path


def export_geojson_bundle(occupancy_data: list[dict], output_path: Path) -> Path:
    """Export all home ranges as a single GeoJSON FeatureCollection."""
    features = []
    for occ in occupancy_data:
        geojson_str = occ.get("home_range_geojson")
        if not geojson_str:
            continue
        try:
            feat = json.loads(geojson_str)
            if "properties" not in feat:
                feat["properties"] = {}
            feat["properties"].update({
                "tiger_code": occ.get("tiger_code"),
                "area_sq_km": occ.get("area_sq_km"),
                "capture_count": occ.get("capture_count"),
                "station_ids": occ.get("station_ids", []),
            })
            features.append(feat)
        except json.JSONDecodeError:
            continue

    collection = {"type": "FeatureCollection", "features": features}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(collection, f, indent=2)
    return output_path


def export_csv_report(occupancy_data: list[dict], output_path: Path) -> Path:
    """Export occupancy summary CSV for forest department."""
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tiger_code",
                "centroid_lat",
                "centroid_lon",
                "area_sq_km",
                "capture_count",
                "stations",
            ],
        )
        writer.writeheader()
        for occ in occupancy_data:
            writer.writerow({
                "tiger_code": occ.get("tiger_code"),
                "centroid_lat": occ.get("centroid_lat"),
                "centroid_lon": occ.get("centroid_lon"),
                "area_sq_km": occ.get("area_sq_km"),
                "capture_count": occ.get("capture_count"),
                "stations": ";".join(occ.get("station_ids", [])),
            })
    return output_path
