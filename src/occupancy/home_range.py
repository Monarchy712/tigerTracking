"""Home range estimation and territorial overlap analysis."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon, mapping
from shapely.ops import unary_union


EARTH_RADIUS_KM = 6371.0


@dataclass
class OccupancyResult:
    centroid_lat: float
    centroid_lon: float
    area_sq_km: float
    capture_count: int
    station_ids: list[str]
    home_range_geojson: str
    points: list[tuple[float, float]]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _latlon_to_local_km(points: list[tuple[float, float]]) -> np.ndarray:
    """Project lat/lon to local km coordinates for area calculation."""
    if not points:
        return np.array([])
    ref_lat = np.mean([p[0] for p in points])
    ref_lon = np.mean([p[1] for p in points])
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(ref_lat))

    local = []
    for lat, lon in points:
        x = (lon - ref_lon) * km_per_deg_lon
        y = (lat - ref_lat) * km_per_deg_lat
        local.append([x, y])
    return np.array(local)


def _polygon_area_sq_km(polygon: Polygon) -> float:
    if polygon.is_empty:
        return 0.0
    return abs(polygon.area)


def compute_occupancy(
    sightings: list,
    min_points: int = 3,
    method: str = "convex_hull",
) -> OccupancyResult | None:
    """Compute home range from sighting GPS coordinates."""
    points = [(s.latitude, s.longitude) for s in sightings if s.latitude and s.longitude]
    stations = list({s.station_id for s in sightings if s.station_id})

    if not points:
        return None

    centroid_lat = float(np.mean([p[0] for p in points]))
    centroid_lon = float(np.mean([p[1] for p in points]))

    if len(points) < min_points:
        area = 0.5
        geojson = json.dumps(
            mapping(Point(centroid_lon, centroid_lat)),
            default=str,
        )
        return OccupancyResult(
            centroid_lat=centroid_lat,
            centroid_lon=centroid_lon,
            area_sq_km=area,
            capture_count=len(sightings),
            station_ids=stations,
            home_range_geojson=geojson,
            points=points,
        )

    local = _latlon_to_local_km(points)
    shapely_points = [Point(x, y) for x, y in local]

    if method == "convex_hull" and len(shapely_points) >= 3:
        hull = MultiPoint(shapely_points).convex_hull
        if isinstance(hull, Polygon) and not hull.is_empty:
            area = _polygon_area_sq_km(hull)
            ref_lat = centroid_lat
            ref_lon = centroid_lon
            km_per_deg_lat = 111.0
            km_per_deg_lon = 111.0 * math.cos(math.radians(ref_lat))

            geo_coords = []
            for x, y in hull.exterior.coords:
                lon = ref_lon + x / km_per_deg_lon
                lat = ref_lat + y / km_per_deg_lat
                geo_coords.append([lon, lat])

            geojson = json.dumps(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [geo_coords]},
                    "properties": {"area_sq_km": area},
                }
            )
            return OccupancyResult(
                centroid_lat=centroid_lat,
                centroid_lon=centroid_lon,
                area_sq_km=round(area, 2),
                capture_count=len(sightings),
                station_ids=stations,
                home_range_geojson=geojson,
                points=points,
            )

    max_dist = max(_haversine_km(centroid_lat, centroid_lon, p[0], p[1]) for p in points)
    area = math.pi * (max_dist ** 2)
    geojson = json.dumps(mapping(Point(centroid_lon, centroid_lat)), default=str)

    return OccupancyResult(
        centroid_lat=centroid_lat,
        centroid_lon=centroid_lon,
        area_sq_km=round(area, 2),
        capture_count=len(sightings),
        station_ids=stations,
        home_range_geojson=geojson,
        points=points,
    )


def compute_overlap_pct(poly_a: dict | None, poly_b: dict | None) -> float:
    """Compute overlap percentage between two GeoJSON geometries."""
    if not poly_a or not poly_b:
        return 0.0
    try:
        from shapely.geometry import shape

        geom_a = shape(poly_a.get("geometry", poly_a))
        geom_b = shape(poly_b.get("geometry", poly_b))
        if geom_a.is_empty or geom_b.is_empty:
            return 0.0
        intersection = geom_a.intersection(geom_b)
        if intersection.is_empty:
            return 0.0
        union = unary_union([geom_a, geom_b])
        return round((intersection.area / union.area) * 100, 1) if union.area > 0 else 0.0
    except Exception:
        return 0.0
