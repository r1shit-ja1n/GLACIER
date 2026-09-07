"""
infrastructure_damage.py

Modular infrastructure damage assessment for GLOF (Glacial Lake Outburst Flood)
simulations. Overlays OpenStreetMap (OSM) vector data — roads, critical
amenities, power infrastructure, and waterways/dams — onto gridded flood
depth/velocity rasters to quantify physical damage.

Design notes
------------
* The flood "damage" surface is computed once as a boolean raster, then
  vectorized to a single (multi)polygon with `rasterio.features.shapes`.
  All subsequent vector/vector overlays (GEOS, via GeoPandas/Shapely) are
  then fast, because we never re-touch the raster after this point and we
  never loop pixel-by-pixel over geometries.
* Everything is done in the metric `utm_crs` so that lengths (km) and
  buffers (meters) are correct without extra unit conversion.
* Every OSM fetch and CRS operation is wrapped so a missing feature class
  (e.g. no hospitals in the bbox) degrades gracefully to zero, rather than
  raising.

Dependencies
------------
    pip install osmnx geopandas rasterio shapely numpy

Tested against osmnx >= 1.9 (features_from_bbox signature changed across
versions; both call conventions are attempted, see `_fetch_osm_layer`).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import geopandas as gpd
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from rasterio import features

try:
    import osmnx as ox
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "osmnx is required for infrastructure_damage.py. "
        "Install with `pip install osmnx`."
    ) from exc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("infrastructure_damage")

# ---------------------------------------------------------------------------
# Physics thresholds
# ---------------------------------------------------------------------------

DEPTH_THRESHOLD_M = 0.5          # meters — structural damage onset
INTENSITY_THRESHOLD = 1.5        # depth(m) * velocity(m/s) — hydrodynamic force proxy

# ---------------------------------------------------------------------------
# OSM tag definitions per infrastructure category
# ---------------------------------------------------------------------------

OSM_TAGS = {
    "highway": {"highway": True},
    "amenity": {"amenity": ["hospital", "clinic", "school", "fire_station", "police"]},
    "power": {"power": ["plant", "substation", "generator"]},
    "waterway": {"waterway": ["dam", "weir"]},
}


# ---------------------------------------------------------------------------
# OSM fetch helpers
# ---------------------------------------------------------------------------

def _fetch_osm_layer(
    bounds_lonlat: Tuple[float, float, float, float],
    tags: dict,
    label: str,
) -> Optional[gpd.GeoDataFrame]:
    """Fetch a single OSM feature layer inside bounds_lonlat.

    bounds_lonlat = (west, south, east, north) in EPSG:4326.
    Returns None (and logs a warning/info) instead of raising if the
    request fails or returns nothing — a missing feature class must never
    crash the whole damage assessment.
    """
    west, south, east, north = bounds_lonlat
    gdf = None

    # osmnx's bbox argument order/signature has changed across releases.
    # Try the modern (bbox=(west, south, east, north)) signature first,
    # then fall back to the older positional (north, south, east, west).
    try:
        gdf = ox.features_from_bbox(bbox=(west, south, east, north), tags=tags)
    except TypeError:
        try:
            gdf = ox.features_from_bbox(north, south, east, west, tags=tags)
        except Exception as exc:
            logger.warning("Failed to fetch OSM layer '%s': %s", label, exc)
            return None
    except Exception as exc:
        logger.warning("Failed to fetch OSM layer '%s': %s", label, exc)
        return None

    if gdf is None or gdf.empty:
        logger.info("No OSM features found for layer '%s' in bounding box.", label)
        return None

    return gdf


def _reproject(gdf: Optional[gpd.GeoDataFrame], utm_crs) -> Optional[gpd.GeoDataFrame]:
    if gdf is None or gdf.empty:
        return None
    try:
        return gdf.to_crs(utm_crs)
    except Exception as exc:
        logger.warning("Reprojection to %s failed: %s", utm_crs, exc)
        return None


# ---------------------------------------------------------------------------
# Raster -> damage polygon
# ---------------------------------------------------------------------------

def _compute_damage_mask(h_max_array: np.ndarray, v_max_array: np.ndarray) -> np.ndarray:
    """Boolean raster: True where the pixel is considered structurally damaged.

    Damaged := depth > DEPTH_THRESHOLD_M  OR  (depth * velocity) > INTENSITY_THRESHOLD
    NaN/inf values (dry cells from GPU solver) are treated as zero before comparison.
    """
    # Sanitize: NaN / inf from GPU dry-cells must not propagate into boolean logic
    h = np.nan_to_num(h_max_array, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(v_max_array, nan=0.0, posinf=0.0, neginf=0.0)
    depth_flag = h > DEPTH_THRESHOLD_M
    intensity_flag = (h * v) > INTENSITY_THRESHOLD
    return (depth_flag | intensity_flag)


def _damage_mask_to_polygon(damage_mask: np.ndarray, transform) -> Optional[BaseGeometry]:
    """Vectorize the boolean damage raster into a single (multi)polygon in
    the raster's own CRS (== utm_crs, since `transform` maps pixel -> utm).

    Doing this once means every downstream vector/vector test (roads,
    buildings, points) is a GEOS operation, not a pixel loop.
    """
    if not damage_mask.any():
        return None

    # rasterio.features.shapes requires:
    #   * dtype uint8 (not bool)
    #   * C-contiguous memory layout (PyTorch .numpy() views may not be)
    mask_uint8 = np.ascontiguousarray(damage_mask, dtype=np.uint8)
    shapes_gen = features.shapes(mask_uint8, mask=mask_uint8, transform=transform)
    polys = [shape(geom) for geom, val in shapes_gen if val == 1]
    if not polys:
        return None
    return unary_union(polys)


# ---------------------------------------------------------------------------
# Damage tallies
# ---------------------------------------------------------------------------

def _names_from(gdf: gpd.GeoDataFrame) -> List[str]:
    if "name" not in gdf.columns:
        return []
    return [n.strip() for n in gdf["name"].dropna().astype(str) if n.strip()]


def _road_damage(
    highways: Optional[gpd.GeoDataFrame], damage_polygon: BaseGeometry
) -> Tuple[float, int]:
    """Return (roads_destroyed_km, bridges_destroyed)."""
    if highways is None or highways.empty:
        return 0.0, 0

    # Cheap pre-filter (bounding-box test happens inside .intersects via GEOS
    # prepared geometries under the hood) before the more expensive
    # .intersection() call, which we only run on the reduced subset.
    hit_mask = highways.geometry.intersects(damage_polygon)
    hit = highways.loc[hit_mask]
    if hit.empty:
        return 0.0, 0

    clipped_len_m = hit.geometry.intersection(damage_polygon).length
    roads_destroyed_km = float(clipped_len_m.sum()) / 1000.0

    bridges_destroyed = 0
    if "bridge" in hit.columns:
        bridge_flag = hit["bridge"].astype(str).str.lower().isin(
            ["yes", "viaduct", "movable", "1", "true"]
        )
        bridges_destroyed = int(bridge_flag.sum())

    return roads_destroyed_km, bridges_destroyed


def _point_or_polygon_damage(
    gdf: Optional[gpd.GeoDataFrame],
    damage_polygon: BaseGeometry,
    pixel_size_m: float,
) -> gpd.GeoDataFrame:
    """Return the subset of gdf (points, lines, or polygons) that overlap the
    damage polygon. Point geometries are buffered by half a pixel to absorb
    rasterization/GPS edge noise; polygons/lines are tested as-is.
    """
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame()

    geom_types = gdf.geometry.geom_type
    is_point = geom_types.isin(["Point", "MultiPoint"])

    test_geom = gdf.geometry.copy()
    if is_point.any():
        buffer_radius = max(pixel_size_m / 2.0, 0.0)
        test_geom.loc[is_point] = gdf.geometry.loc[is_point].buffer(buffer_radius)

    hit_mask = test_geom.intersects(damage_polygon)
    return gdf.loc[hit_mask]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_infrastructure_damage(
    h_max_array: np.ndarray,
    v_max_array: np.ndarray,
    transform,
    utm_crs,
    bounds_lonlat: Tuple[float, float, float, float],
    pixel_size_m: float,
) -> Dict:
    """Assess OSM infrastructure damage against a GLOF flood simulation.

    Parameters
    ----------
    h_max_array : np.ndarray
        2D array of maximum flood depth (m).
    v_max_array : np.ndarray
        2D array of maximum flood velocity (m/s), same shape as h_max_array.
    transform : affine.Affine
        Raster transform (pixel -> utm_crs coordinates) shared by both arrays.
    utm_crs : CRS-like (e.g. "EPSG:32645")
        Metric CRS of the simulation grid.
    bounds_lonlat : (west, south, east, north)
        Bounding box in EPSG:4326 used to fetch OSM data.
    pixel_size_m : float
        Grid cell size in meters (used to buffer point features for the
        vector overlay).

    Returns
    -------
    dict
        Damage summary, safe to json.dumps() and append to glof_metadata.json.
        Example:
            {
                "hospitals_destroyed": 2,
                "schools_destroyed": 1,
                "roads_destroyed_km": 14.5,
                "bridges_destroyed": 3,
                "power_facilities_destroyed": 1,
                "dams_at_risk": 1,
                "critical_assets_at_risk": ["Chungthang Dam", "Mangan District Hospital"]
            }
    """
    empty_result = {
        "hospitals_destroyed": 0,
        "schools_destroyed": 0,
        "roads_destroyed_km": 0.0,
        "bridges_destroyed": 0,
        "power_facilities_destroyed": 0,
        "dams_at_risk": 0,
        "critical_assets_at_risk": [],
    }

    if h_max_array.shape != v_max_array.shape:
        logger.error("h_max_array and v_max_array shapes differ; aborting damage assessment.")
        return empty_result

    try:
        damage_mask = _compute_damage_mask(h_max_array, v_max_array)
    except Exception as exc:
        logger.error("Failed to compute damage mask: %s", exc)
        return empty_result

    damage_polygon = _damage_mask_to_polygon(damage_mask, transform)
    if damage_polygon is None:
        logger.info("No cells exceed the damage thresholds; returning zero-damage result.")
        return empty_result

    # --- Fetch + reproject OSM layers (each isolated so one failure doesn't
    #     take down the others) -------------------------------------------------
    highways = _reproject(_fetch_osm_layer(bounds_lonlat, OSM_TAGS["highway"], "highway"), utm_crs)
    amenities = _reproject(_fetch_osm_layer(bounds_lonlat, OSM_TAGS["amenity"], "amenity"), utm_crs)
    power = _reproject(_fetch_osm_layer(bounds_lonlat, OSM_TAGS["power"], "power"), utm_crs)
    waterway = _reproject(_fetch_osm_layer(bounds_lonlat, OSM_TAGS["waterway"], "waterway"), utm_crs)

    critical_assets_at_risk: List[str] = []

    # --- Roads / bridges --------------------------------------------------------
    try:
        roads_destroyed_km, bridges_destroyed = _road_damage(highways, damage_polygon)
    except Exception as exc:
        logger.warning("Road damage calculation failed: %s", exc)
        roads_destroyed_km, bridges_destroyed = 0.0, 0

    # --- Amenities (hospitals, clinics, schools, etc.) --------------------------
    hospitals_destroyed = 0
    schools_destroyed = 0
    try:
        damaged_amenities = _point_or_polygon_damage(amenities, damage_polygon, pixel_size_m)
        if not damaged_amenities.empty and "amenity" in damaged_amenities.columns:
            hospitals_destroyed = int(
                damaged_amenities["amenity"].isin(["hospital", "clinic"]).sum()
            )
            schools_destroyed = int(damaged_amenities["amenity"].isin(["school"]).sum())
        critical_assets_at_risk.extend(_names_from(damaged_amenities))
    except Exception as exc:
        logger.warning("Amenity damage calculation failed: %s", exc)

    # --- Power infrastructure ----------------------------------------------------
    power_facilities_destroyed = 0
    try:
        damaged_power = _point_or_polygon_damage(power, damage_polygon, pixel_size_m)
        power_facilities_destroyed = int(len(damaged_power))
        critical_assets_at_risk.extend(_names_from(damaged_power))
    except Exception as exc:
        logger.warning("Power infrastructure damage calculation failed: %s", exc)

    # --- Waterway structures (dams, weirs) ---------------------------------------
    dams_at_risk = 0
    try:
        damaged_waterway = _point_or_polygon_damage(waterway, damage_polygon, pixel_size_m)
        dams_at_risk = int(len(damaged_waterway))
        critical_assets_at_risk.extend(_names_from(damaged_waterway))
    except Exception as exc:
        logger.warning("Waterway damage calculation failed: %s", exc)

    result = {
        "hospitals_destroyed": hospitals_destroyed,
        "schools_destroyed": schools_destroyed,
        "roads_destroyed_km": round(roads_destroyed_km, 2),
        "bridges_destroyed": bridges_destroyed,
        "power_facilities_destroyed": power_facilities_destroyed,
        "dams_at_risk": dams_at_risk,
        "critical_assets_at_risk": sorted(set(critical_assets_at_risk)),
    }
    return result


# ---------------------------------------------------------------------------
# Example usage (integration sketch — not executed on import)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    # These would come from your GPU simulation output, e.g.:
    #   h_max_np, v_max_np, transform, utm_crs = run_glof_simulation(...)
    example_shape = (500, 500)
    h_max_np = np.random.uniform(0, 3, example_shape).astype(np.float32)
    v_max_np = np.random.uniform(0, 4, example_shape).astype(np.float32)

    from affine import Affine
    pixel_size = 10.0  # meters
    example_transform = Affine(pixel_size, 0.0, 550000.0, 0.0, -pixel_size, 3070000.0)
    example_utm_crs = "EPSG:32645"

    # Example AOI over Mangan district, Sikkim (west, south, east, north)
    example_bounds_lonlat = (88.50, 27.50, 88.60, 27.58)

    damage_summary = calculate_infrastructure_damage(
        h_max_array=h_max_np,
        v_max_array=v_max_np,
        transform=example_transform,
        utm_crs=example_utm_crs,
        bounds_lonlat=example_bounds_lonlat,
        pixel_size_m=pixel_size,
    )

    print(json.dumps(damage_summary, indent=2))

    # Append to an existing glof_metadata.json:
    #   with open("glof_metadata.json", "r+") as f:
    #       meta = json.load(f)
    #       meta["infrastructure_damage"] = damage_summary
    #       f.seek(0)
    #       json.dump(meta, f, indent=2)
    #       f.truncate()