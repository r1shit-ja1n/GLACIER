#!/usr/bin/env python3
"""
glof_simulator.py

GPU-accelerated 2D Shallow Water Equation (SWE) simulator for Glacial Lake
Outburst Flood (GLOF) hazard modeling.

Pipeline
--------
1. Load a Copernicus 30 m DEM GeoTIFF (EPSG:4326), auto-detect the local UTM
   zone, and reproject to a metric CRS with rasterio.
2. Initialize a glacial lake (radius, polygon, or elevation-fill threshold)
   and a time-dependent moraine dam breach that progressively lowers the
   terrain at the breach zone.
3. Run an explicit, GPU-resident 2D SWE solver (PyTorch CUDA tensors) with
   Manning friction, a wetting/drying threshold, sediment/debris bulking,
   and CFL-adaptive time stepping.
4. Track per-cell peak depth, arrival time, peak velocity, and sediment
   fraction, then pack them into a 4-channel RGBA texture for a Babylon.js
   shader (flood_packed.png).
5. Log telemetry every 30 s of simulation time and assess per-settlement
   hydrodynamic damage (from a CSV or an OSM fallback query), writing
   everything to glof_metadata.json.

This is a research / hazard-assessment tool. It is a hydraulic approximation
intended for scenario screening, not a certified engineering model — treat
outputs as indicative, and validate against observed events / detailed 2D
codes (e.g. BASEMENT, HEC-RAS 2D, TELEMAC) before using for life-safety
decisions.

Dependencies: torch (CUDA build), rasterio, numpy, pillow, scipy.
Optional: geopandas, osmnx (for the OSM settlement fallback).
"""

from __future__ import annotations

import json
import math
import time
import warnings
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Literal

import numpy as np

try:
    import torch
except ImportError as e:
    raise ImportError(
        "This script requires PyTorch with CUDA support. "
        "Install from https://pytorch.org/get-started/locally/"
    ) from e

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds
from rasterio.transform import rowcol, xy

from PIL import Image

try:
    from scipy.ndimage import distance_transform_edt
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# infrastructure_damage is optional — requires osmnx + geopandas
try:
    from infrastructure_damage import calculate_infrastructure_damage
    _HAS_INFRA_DAMAGE = True
except ImportError:
    _HAS_INFRA_DAMAGE = False
    def calculate_infrastructure_damage(*args, **kwargs):  # type: ignore[misc]
        return {"hospitals_destroyed": 0, "schools_destroyed": 0,
                "roads_destroyed_km": 0.0, "bridges_destroyed": 0,
                "power_facilities_destroyed": 0, "dams_at_risk": 0,
                "critical_assets_at_risk": [],
                "note": "osmnx/geopandas not installed; install with pip install osmnx geopandas"}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class LakeConfig:
    """Defines the initial glacial lake extent and depth."""
    mode: Literal["radius", "polygon", "elevation"] = "radius"
    center_lat: float = 0.0
    center_lon: float = 0.0
    radius_m: float = 300.0
    polygon_lonlat: Optional[Sequence[tuple]] = None  # [(lon, lat), ...]
    elevation_threshold_m: Optional[float] = None     # fill below this Z
    initial_depth_m: float = 25.0


@dataclass
class DamBreachConfig:
    """Time-dependent moraine dam breach parameters."""
    dam_lat: float = 0.0
    dam_lon: float = 0.0
    breach_width_m: float = 60.0
    breach_depth_m: float = 35.0        # total elevation drop at breach
    breach_duration_s: float = 15 * 60  # time to fully open (linear ramp)
    breach_start_s: float = 0.0         # sim time the breach begins


@dataclass
class SimConfig:
    dem_path: str
    settlements_csv: Optional[str] = None
    output_dir: str = "./glof_output"

    # AOI cropping (in EPSG:4326 lon/lat); None = use full DEM
    bbox_lonlat: Optional[tuple] = None  # (min_lon, min_lat, max_lon, max_lat)

    lake: LakeConfig = field(default_factory=LakeConfig)
    breach: DamBreachConfig = field(default_factory=DamBreachConfig)

    # Hydraulics
    manning_n: float = 0.045
    h_dry: float = 0.01          # wetting/drying threshold [m]
    gravity: float = 9.81

    # Sediment / debris bulking
    bulking_factor: float = 1.6           # 1.4-1.8 typical for GLOF debris flows
    sediment_slope_gain: float = 0.08     # Cs increase per unit slope per step
    sediment_max_fraction: float = 0.55

    # CFL / time stepping
    cfl_number: float = 0.4
    dt_min: float = 1e-4
    dt_max: float = 2.0

    # Simulation duration & logging
    sim_duration_s: float = 3 * 3600.0    # 3 hours of simulated flood time
    telemetry_interval_s: float = 30.0

    # Normalization caps for texture packing (tune per-catchment)
    h_norm_max_m: float = 15.0
    v_norm_max_ms: float = 12.0

    # Numerics
    device: str = "cuda"
    dtype: str = "float32"

    # Arrival-time detection threshold
    arrival_depth_threshold_m: float = 0.1

    # Wall-clock safety cap (stop early if the GPU loop runs long); None = off
    max_wallclock_s: Optional[float] = None


# --------------------------------------------------------------------------- #
# 1. DEM loading, UTM auto-detection, reprojection, cropping
# --------------------------------------------------------------------------- #

def _utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    """Return the EPSG code of the UTM zone containing (lon, lat)."""
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    zone = max(1, min(60, zone))
    return (32600 if lat >= 0 else 32700) + zone


def load_and_reproject_dem(cfg: SimConfig):
    """
    Load a Copernicus DEM GeoTIFF (EPSG:4326), crop to an optional bbox,
    auto-detect the local UTM zone, and reproject to that metric CRS.

    Returns
    -------
    Z_np : np.ndarray (float32)   elevation raster in the UTM CRS
    transform : rasterio.Affine   affine transform of the reprojected raster
    utm_crs : rasterio.crs.CRS
    pixel_size_m : float
    """
    with rasterio.open(cfg.dem_path) as src:
        if src.crs is None:
            raise ValueError("Input DEM has no CRS; expected EPSG:4326.")

        if cfg.bbox_lonlat is not None:
            window = from_bounds(*cfg.bbox_lonlat, transform=src.transform)
            src_data = src.read(1, window=window)
            src_transform = src.window_transform(window)
        else:
            src_data = src.read(1)
            src_transform = src.transform

        src_crs = src.crs
        nodata = src.nodata

        # Determine UTM zone from bbox / lake center, whichever is defined.
        if cfg.bbox_lonlat is not None:
            lon_c = (cfg.bbox_lonlat[0] + cfg.bbox_lonlat[2]) / 2.0
            lat_c = (cfg.bbox_lonlat[1] + cfg.bbox_lonlat[3]) / 2.0
        else:
            lon_c, lat_c = cfg.lake.center_lon, cfg.lake.center_lat

        utm_epsg = _utm_epsg_for_lonlat(lon_c, lat_c)
        utm_crs = rasterio.crs.CRS.from_epsg(utm_epsg)

        dst_transform, width, height = calculate_default_transform(
            src_crs, utm_crs,
            src_data.shape[1], src_data.shape[0],
            *rasterio.transform.array_bounds(
                src_data.shape[0], src_data.shape[1], src_transform
            ),
        )

        dst_data = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=utm_crs,
            resampling=Resampling.bilinear,
            src_nodata=nodata,
            dst_nodata=np.nan,
        )

    # Fill any nodata gaps with nearest-valid elevation so the solver never
    # sees NaNs (edge slivers from reprojection, etc.).
    if np.isnan(dst_data).any():
        if _HAS_SCIPY:
            mask = np.isnan(dst_data)
            idx = distance_transform_edt(
                mask, return_distances=False, return_indices=True
            )
            dst_data = dst_data[tuple(idx)]
        else:
            fill_val = np.nanmean(dst_data)
            dst_data = np.where(np.isnan(dst_data), fill_val, dst_data)

    pixel_size_m = abs(dst_transform.a)
    return dst_data.astype(np.float32), dst_transform, utm_crs, pixel_size_m


# --------------------------------------------------------------------------- #
# 2. Lake initialization & dam breach
# --------------------------------------------------------------------------- #

def lonlat_to_rowcol(transform, utm_crs, lon: float, lat: float):
    """Project a WGS84 lon/lat point into the raster's row/col index."""
    from rasterio.warp import transform as warp_transform
    xs, ys = warp_transform("EPSG:4326", utm_crs, [lon], [lat])
    row, col = rowcol(transform, xs[0], ys[0])
    return int(row), int(col)


def initialize_lake(Z: torch.Tensor, cfg: SimConfig, transform, utm_crs,
                     pixel_size_m: float) -> torch.Tensor:
    """
    Build the initial water-depth field h0 for the lake basin.
    Returns an (H, W) float tensor on the same device as Z.
    """
    device, dtype = Z.device, Z.dtype
    H, W = Z.shape
    h0 = torch.zeros_like(Z)

    if cfg.lake.mode == "radius":
        row_c, col_c = lonlat_to_rowcol(
            transform, utm_crs, cfg.lake.center_lon, cfg.lake.center_lat
        )
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij",
        )
        dist_m = torch.sqrt((yy - row_c) ** 2 + (xx - col_c) ** 2) * pixel_size_m
        mask = dist_m <= cfg.lake.radius_m
        h0[mask] = cfg.lake.initial_depth_m

    elif cfg.lake.mode == "polygon":
        if cfg.lake.polygon_lonlat is None:
            raise ValueError("polygon mode requires lake.polygon_lonlat")
        from matplotlib.path import Path as MplPath  # lightweight, no extra dep beyond mpl
        rows_cols = [
            lonlat_to_rowcol(transform, utm_crs, lon, lat)
            for lon, lat in cfg.lake.polygon_lonlat
        ]
        poly_path = MplPath([(c, r) for r, c in rows_cols])
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        inside = poly_path.contains_points(pts).reshape(H, W)
        mask = torch.from_numpy(inside).to(device)
        h0[mask] = cfg.lake.initial_depth_m

    elif cfg.lake.mode == "elevation":
        if cfg.lake.elevation_threshold_m is None:
            raise ValueError("elevation mode requires lake.elevation_threshold_m")
        mask = Z <= cfg.lake.elevation_threshold_m
        # Depth = fill level minus bed elevation, capped by initial_depth_m.
        fill = torch.clamp(cfg.lake.elevation_threshold_m - Z, min=0.0)
        h0 = torch.where(mask, torch.clamp(fill, max=cfg.lake.initial_depth_m), h0)

    else:
        raise ValueError(f"Unknown lake mode: {cfg.lake.mode}")

    return h0


class DamBreach:
    """
    Applies a progressive elevation drop at the dam/breach zone.

    The breach is modeled as a Gaussian-weighted notch in terrain elevation
    that widens/deepens linearly from t=breach_start_s to
    t=breach_start_s + breach_duration_s, after which the terrain stays at
    its fully-breached elevation. This creates a realistic outflow hydrograph
    rather than an instantaneous dam-break.
    """

    def __init__(self, cfg: SimConfig, Z0: torch.Tensor, transform, utm_crs,
                 pixel_size_m: float):
        self.cfg = cfg
        self.Z0 = Z0.clone()
        self.pixel_size_m = pixel_size_m
        device, dtype = Z0.device, Z0.dtype
        H, W = Z0.shape

        row_c, col_c = lonlat_to_rowcol(
            transform, utm_crs, cfg.breach.dam_lon, cfg.breach.dam_lat
        )
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij",
        )
        dist_m = torch.sqrt((yy - row_c) ** 2 + (xx - col_c) ** 2) * pixel_size_m
        sigma = max(cfg.breach.breach_width_m / 2.355, 1e-3)  # FWHM -> sigma
        self.breach_weight = torch.exp(-0.5 * (dist_m / sigma) ** 2)  # 0..1

    def elevation_at(self, t_sec: float) -> torch.Tensor:
        b = self.cfg.breach
        if t_sec <= b.breach_start_s:
            frac = 0.0
        else:
            frac = min(1.0, (t_sec - b.breach_start_s) / max(b.breach_duration_s, 1e-6))
        drop = frac * b.breach_depth_m
        return self.Z0 - drop * self.breach_weight


# --------------------------------------------------------------------------- #
# 3. GPU-resident 2D Shallow Water Equation solver
# --------------------------------------------------------------------------- #

class SWESolver:
    """
    Explicit, conservative-form 2D shallow water solver on a regular grid,
    fully resident on the GPU as PyTorch tensors.

    Formulation
    -----------
    Conserved variables: h, hu, hv.
    Fluxes use a Lax-Friedrichs (local) numerical flux for stability across
    wet/dry fronts and transcritical flow, which is the standard robust
    choice for GLOF/dam-break style problems without a full Riemann solver.
    Bed-slope source term: -g h dZ/dx, -g h dZ/dy (central differences).
    Friction source term: semi-implicit Manning drag, applied after the
    advective/pressure update so shallow cells cannot overshoot to negative
    depth or blow up in velocity.
    """

    def __init__(self, cfg: SimConfig, Z0: torch.Tensor, h0: torch.Tensor,
                 pixel_size_m: float, dam_breach: DamBreach):
        self.cfg = cfg
        self.dx = pixel_size_m
        self.g = cfg.gravity
        self.n_manning = cfg.manning_n
        self.h_dry = cfg.h_dry
        self.dam_breach = dam_breach

        self.device = Z0.device
        self.dtype = Z0.dtype

        self.Z = Z0.clone()
        self.h = h0.clone()
        self.hu = torch.zeros_like(self.h)
        self.hv = torch.zeros_like(self.h)

        H, W = self.h.shape
        self.h_max = torch.zeros_like(self.h)
        self.v_max = torch.zeros_like(self.h)
        self.t_arrival = torch.full((H, W), float("nan"), device=self.device, dtype=self.dtype)
        self.t_peak = torch.full((H, W), float("nan"), device=self.device, dtype=self.dtype)
        self.t_recede = torch.full((H, W), float("nan"), device=self.device, dtype=self.dtype)
        self.Cs = torch.zeros_like(self.h)  # sediment/debris fraction

        # Precompute slope magnitude (updated each time the dam elevation changes)
        self._update_slopes()

        self.t = 0.0

    # -- helpers -------------------------------------------------------- #

    def _update_slopes(self):
        Z = self.Z
        dZdx = torch.zeros_like(Z)
        dZdy = torch.zeros_like(Z)
        dZdx[:, 1:-1] = (Z[:, 2:] - Z[:, :-2]) / (2 * self.dx)
        dZdy[1:-1, :] = (Z[2:, :] - Z[:-2, :]) / (2 * self.dx)
        dZdx[:, 0] = (Z[:, 1] - Z[:, 0]) / self.dx
        dZdx[:, -1] = (Z[:, -1] - Z[:, -2]) / self.dx
        dZdy[0, :] = (Z[1, :] - Z[0, :]) / self.dx
        dZdy[-1, :] = (Z[-1, :] - Z[-2, :]) / self.dx
        self.dZdx = dZdx
        self.dZdy = dZdy
        self.slope_mag = torch.sqrt(dZdx ** 2 + dZdy ** 2)

    def _velocities(self, h, hu, hv):
        h_safe = torch.clamp(h, min=self.h_dry)
        u = torch.where(h > self.h_dry, hu / h_safe, torch.zeros_like(h))
        v = torch.where(h > self.h_dry, hv / h_safe, torch.zeros_like(h))
        return u, v

    def _adaptive_dt(self, h, u, v):
        h_max_val = torch.clamp(h.max(), min=self.h_dry).item()
        speed = torch.sqrt(u ** 2 + v ** 2)
        speed_max_val = speed.max().item()
        wave_speed = math.sqrt(self.g * h_max_val) + speed_max_val
        dt = self.cfg.cfl_number * self.dx / max(wave_speed, 1e-6)
        return float(np.clip(dt, self.cfg.dt_min, self.cfg.dt_max))

    @staticmethod
    def _grad_x(f):
        g = torch.zeros_like(f)
        g[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / 2.0
        g[:, 0] = f[:, 1] - f[:, 0]
        g[:, -1] = f[:, -1] - f[:, -2]
        return g

    @staticmethod
    def _grad_y(f):
        g = torch.zeros_like(f)
        g[1:-1, :] = (f[2:, :] - f[:-2, :]) / 2.0
        g[0, :] = f[1, :] - f[0, :]
        g[-1, :] = f[-1, :] - f[-2, :]
        return g

    @staticmethod
    def _laplacian(f):
        lap = torch.zeros_like(f)
        lap[1:-1, 1:-1] = (
            f[2:, 1:-1] + f[:-2, 1:-1] + f[1:-1, 2:] + f[1:-1, :-2] - 4 * f[1:-1, 1:-1]
        )
        return lap

    # -- one explicit step ------------------------------------------------ #

    def step(self) -> float:
        cfg = self.cfg
        h, hu, hv, Z = self.h, self.hu, self.hv, self.Z
        g = self.g
        dx = self.dx

        u, v = self._velocities(h, hu, hv)
        dt = self._adaptive_dt(h, u, v)

        wet = (h > self.h_dry).to(self.dtype)

        # Conservative fluxes (mass, x-momentum, y-momentum)
        Fh_x = hu
        Fh_y = hv
        Fhu_x = hu * u + 0.5 * g * h ** 2
        Fhu_y = hu * v
        Fhv_x = hv * u
        Fhv_y = hv * v + 0.5 * g * h ** 2

        dFh_x = self._grad_x(Fh_x) / dx
        dFh_y = self._grad_y(Fh_y) / dx
        dFhu_x = self._grad_x(Fhu_x) / dx
        dFhu_y = self._grad_y(Fhu_y) / dx
        dFhv_x = self._grad_x(Fhv_x) / dx
        dFhv_y = self._grad_y(Fhv_y) / dx

        # Local Lax-Friedrichs style numerical diffusion for shock/front
        # stability (keeps the explicit scheme well-behaved at wet/dry
        # boundaries and the breach jet without a full Riemann solver).
        wave_speed = torch.sqrt(g * torch.clamp(h, min=self.h_dry)) + torch.sqrt(u ** 2 + v ** 2)
        alpha = 0.5 * wave_speed.mean().clamp(min=1e-6)
        diff_h = alpha * self._laplacian(h)
        diff_hu = alpha * self._laplacian(hu)
        diff_hv = alpha * self._laplacian(hv)

        # Bed-slope source term
        Sx_bed = -g * h * self.dZdx
        Sy_bed = -g * h * self.dZdy

        h_new = h - dt * (dFh_x + dFh_y) + dt * diff_h
        hu_new = hu - dt * (dFhu_x + dFhu_y) + dt * Sx_bed + dt * diff_hu
        hv_new = hv - dt * (dFhv_x + dFhv_y) + dt * Sy_bed + dt * diff_hv

        h_new = torch.clamp(h_new, min=0.0)

        # Semi-implicit Manning friction (applied to momentum, stable at
        # shallow depth): hu_{n+1} = hu* / (1 + dt * g * n^2 * |vel| / h^{4/3})
        h_safe = torch.clamp(h_new, min=self.h_dry)
        u_new, v_new = self._velocities(h_new, hu_new, hv_new)
        speed = torch.sqrt(u_new ** 2 + v_new ** 2)
        friction_denom = 1.0 + dt * g * (self.n_manning ** 2) * speed / (h_safe ** (4.0 / 3.0))
        hu_new = hu_new / friction_denom
        hv_new = hv_new / friction_denom

        # Re-impose dry-cell zero velocity to avoid drift/noise in dry areas.
        dry_mask = h_new <= self.h_dry
        hu_new = torch.where(dry_mask, torch.zeros_like(hu_new), hu_new)
        hv_new = torch.where(dry_mask, torch.zeros_like(hv_new), hv_new)

        # -- sediment / debris bulking ------------------------------------
        # Effective (bulked) depth used for hazard tracking; Cs grows with
        # local slope while the cell is actively wetting/flowing, saturating
        # at sediment_max_fraction.
        u_f, v_f = self._velocities(h_new, hu_new, hv_new)
        speed_f = torch.sqrt(u_f ** 2 + v_f ** 2)
        flowing = (h_new > self.h_dry) & (speed_f > 0.05)
        self.Cs = torch.where(
            flowing,
            torch.clamp(
                self.Cs + cfg.sediment_slope_gain * self.slope_mag * dt,
                max=cfg.sediment_max_fraction,
            ),
            self.Cs,
        )
        h_effective = h_new * (1.0 + (cfg.bulking_factor - 1.0) * self.Cs)

        # -- update trackers ------------------------------------------------
       # -- update trackers ------------------------------------------------
        self.t += dt
        
        # Track Time to Peak Depth
        new_peak_mask = h_effective > self.h_max
        self.h_max = torch.where(new_peak_mask, h_effective, self.h_max)
        self.t_peak = torch.where(new_peak_mask, torch.full_like(self.t_peak, self.t), self.t_peak)
        self.v_max = torch.maximum(self.v_max, speed_f)

        # Track Arrival Time
        newly_arrived = torch.isnan(self.t_arrival) & (h_new > cfg.arrival_depth_threshold_m)
        self.t_arrival = torch.where(
            newly_arrived, torch.full_like(self.t_arrival, self.t), self.t_arrival
        )
        
        # Track Recession Time (Continually updates as long as water drains below threshold)
        receded_mask = (~torch.isnan(self.t_arrival)) & (h_new <= cfg.arrival_depth_threshold_m)
        self.t_recede = torch.where(
            receded_mask, torch.full_like(self.t_recede, self.t), self.t_recede
        )
        # -- apply dam breach terrain update for the *next* step -----------
        self.Z = self.dam_breach.elevation_at(self.t)
        self._update_slopes()

        self.h, self.hu, self.hv = h_new, hu_new, hv_new
        return dt


# --------------------------------------------------------------------------- #
# 4. Texture packing
# --------------------------------------------------------------------------- #

def pack_rgba_texture(h_max: torch.Tensor, t_arrival: torch.Tensor, v_max: torch.Tensor,
                       Cs: torch.Tensor, cfg: SimConfig, sim_duration_s: float,
                       out_path: str):
    """
    Normalize the four hazard channels and pack them into an 8-bit RGBA PNG
    for consumption by a Babylon.js custom shader:
      R = peak depth, G = arrival time, B = peak velocity, A = sediment/debris.
    Cells never inundated get G=255 ("never arrived" sentinel) and R=B=A=0.
    """
    h_np = h_max.detach().cpu().numpy()
    v_np = v_max.detach().cpu().numpy()
    cs_np = Cs.detach().cpu().numpy()
    t_np = t_arrival.detach().cpu().numpy()

    never_arrived = np.isnan(t_np)

    R = np.clip(h_np / max(cfg.h_norm_max_m, 1e-6), 0.0, 1.0) * 255.0
    B = np.clip(v_np / max(cfg.v_norm_max_ms, 1e-6), 0.0, 1.0) * 255.0
    A = np.clip(cs_np, 0.0, 1.0) * 255.0

    G = np.where(
        never_arrived,
        255.0,
        np.clip(t_np / max(sim_duration_s, 1e-6), 0.0, 1.0) * 255.0,
    )
    # Zero-out hazard channels where the cell was never wetted.
    R = np.where(never_arrived, 0.0, R)
    B = np.where(never_arrived, 0.0, B)
    A = np.where(never_arrived, 0.0, A)

    rgba = np.stack([R, G, B, A], axis=-1).astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(out_path)
    return out_path


# --------------------------------------------------------------------------- #
# 5. Settlement ingestion & damage assessment
# --------------------------------------------------------------------------- #

def load_settlements(cfg: SimConfig, bounds_lonlat: tuple) -> list:
    """
    Load settlements from CSV if provided; otherwise (or additionally) query
    OSM via osmnx for place nodes within the bounding box. Returns a list of
    dicts: {name, latitude, longitude, estimated_population, type}.
    """
    settlements = []

    if cfg.settlements_csv and Path(cfg.settlements_csv).exists():
        import csv
        with open(cfg.settlements_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                settlements.append({
                    "name": row.get("name", "unnamed"),
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "estimated_population": int(float(row.get("estimated_population", 0) or 0)),
                    "estimated_value_usd": float(row.get("estimated_value_usd", 0) or 0), # <--- ADDED HERE
                    "type": row.get("type", "unknown"),
                    "source": "csv",
                })

    if not settlements:
        try:
            import osmnx as ox
            min_lon, min_lat, max_lon, max_lat = bounds_lonlat
            tags = {"place": ["hamlet", "village", "town"]}
            gdf = ox.features_from_bbox(max_lat, min_lat, max_lon, min_lon, tags)
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None:
                    continue
                pt = geom.centroid
                settlements.append({
                    "name": row.get("name", "unnamed"),
                    "latitude": pt.y,
                    "longitude": pt.x,
                    "estimated_population": int(row.get("population", 0) or 0),
                    "type": row.get("place", "unknown"),
                    "source": "osm",
                })
        except Exception as e:  # osmnx optional / network optional
            warnings.warn(f"OSM settlement fallback unavailable: {e}")

    return settlements


def classify_damage(h: float, v: float) -> str:
    intensity = h * v
    if intensity >= 1.5 or h >= 3.0:
        return "Severe / Catastrophic Failure"
    if h >= 0.5 or v >= 1.5:
        return "Moderate Inundation"
    if h < 0.3 and v < 1.0:
        return "Safe / Minimal"
    return "Moderate Inundation"  # anything in-between defaults to caution


def assess_settlement_damage(settlements: list, solver: SWESolver, transform, utm_crs) -> list:
    reports = []
    H, W = solver.h_max.shape
    
    # Pull arrays from GPU
    h_max_np = solver.h_max.detach().cpu().numpy()
    v_max_np = solver.v_max.detach().cpu().numpy()
    t_arr_np = solver.t_arrival.detach().cpu().numpy()
    t_peak_np = solver.t_peak.detach().cpu().numpy()
    t_recede_np = solver.t_recede.detach().cpu().numpy()
    Z_np = solver.Z.detach().cpu().numpy() # For elevation ASL

    for s in settlements:
        try:
            row, col = lonlat_to_rowcol(transform, utm_crs, s["longitude"], s["latitude"])
        except Exception:
            continue
        if not (0 <= row < H and 0 <= col < W):
            continue

        h = float(h_max_np[row, col])
        v = float(v_max_np[row, col])
        z_bed = float(Z_np[row, col])
        
        t_arr = None if np.isnan(t_arr_np[row, col]) else float(t_arr_np[row, col])
        t_peak = None if np.isnan(t_peak_np[row, col]) else float(t_peak_np[row, col])
        t_recede = None if np.isnan(t_recede_np[row, col]) else float(t_recede_np[row, col])

        # Durations & Warnings
        warning_lead_time_sec = max(0, t_arr - solver.cfg.breach.breach_start_s) if t_arr else None
        
        flood_duration_sec = None
        if t_arr is not None:
            if t_recede is not None and t_recede > t_arr:
                flood_duration_sec = t_recede - t_arr
            else:
                flood_duration_sec = solver.t - t_arr # Still flooded at sim end

        I = h * v
        Ik = h * (v ** 2)
        damage_cat = classify_damage(h, v)

        # Financial Loss Logic
        base_value = s.get("estimated_value_usd", 0)
        est_loss = 0
        if damage_cat == "Severe / Catastrophic Failure":
            est_loss = base_value * 1.0 # 100% loss
        elif damage_cat == "Moderate Inundation":
            est_loss = base_value * 0.4 # 40% loss

        reports.append({
            "name": s["name"],
            "type": s.get("type", "unknown"),
            "estimated_population": s.get("estimated_population", 0),
            "latitude": s["latitude"],
            "longitude": s["longitude"],
            
            # New Advanced Metrics
            "eta_arrival_sec": t_arr,
            "warning_lead_time_sec": warning_lead_time_sec,
            "time_to_peak_sec": t_peak,
            "flood_duration_sec": flood_duration_sec,
            "peak_water_elevation_asl_m": (z_bed + h) if h > 0 else None,
            "estimated_financial_loss_usd": est_loss,
            "is_network_severed": True if (s.get("type") in ["bridge", "highway"] and damage_cat == "Severe / Catastrophic Failure") else False,
            
            # Standard Metrics
            "peak_depth_m": h,
            "peak_velocity_ms": v,
            "hydrodynamic_intensity_hv": I,
            "damage_category": damage_cat,
            "source": s.get("source", "unknown"),
        })

    reports.sort(key=lambda r: (r["eta_arrival_sec"] is None, r["eta_arrival_sec"]))
    return reports
# --------------------------------------------------------------------------- #
# 6. Main simulation driver
# --------------------------------------------------------------------------- #

def run_simulation(cfg: SimConfig):
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available() and cfg.device == "cuda":
        warnings.warn("CUDA not available; falling back to CPU (will be slow).")
        cfg.device = "cpu"

    device = torch.device(cfg.device)
    dtype = getattr(torch, cfg.dtype)

    print(f"[1/7] Loading & reprojecting DEM: {cfg.dem_path}")
    Z_np, transform, utm_crs, pixel_size_m = load_and_reproject_dem(cfg)
    print(f"      Grid: {Z_np.shape}, pixel size ~{pixel_size_m:.2f} m, CRS: {utm_crs}")

    Z = torch.from_numpy(Z_np).to(device=device, dtype=dtype)

    print("[2/7] Initializing glacial lake")
    h0 = initialize_lake(Z, cfg, transform, utm_crs, pixel_size_m)
    print(f"      Initial inundated cells: {int((h0 > 0).sum().item())}")

    print("[3/7] Setting up dam breach mechanism")
    dam_breach = DamBreach(cfg, Z, transform, utm_crs, pixel_size_m)

    print("[4/7] Running GPU-accelerated SWE solver")
    solver = SWESolver(cfg, Z, h0, pixel_size_m, dam_breach)

    telemetry = []
    next_log_t = 0.0
    wall_start = time.time()
    step_count = 0
    cell_area_km2 = (pixel_size_m ** 2) / 1e6

    while solver.t < cfg.sim_duration_s:
        dt = solver.step()
        step_count += 1

        if solver.t >= next_log_t:
            inundated_mask = solver.h > cfg.h_dry
            inundated_area_km2 = float(inundated_mask.sum().item()) * cell_area_km2
            # Split Total Volume into Water and Debris
            cell_vol = solver.h * (pixel_size_m ** 2)
            active_volume_m3 = float(cell_vol.sum().item())
            debris_volume_m3 = float((cell_vol * solver.Cs).sum().item())
            water_volume_m3 = active_volume_m3 - debris_volume_m3
            
            u, v = solver._velocities(solver.h, solver.hu, solver.hv)
            # Discharge proxy: sum(|q|) across the widest active cross-section
            # row (max inundated row-width) as a peak-discharge estimate.
            speed = torch.sqrt(u ** 2 + v ** 2)
            q_per_cell = solver.h * speed * pixel_size_m  # m^3/s contributed per cell (1D proxy)
            peak_discharge_m3s = float(q_per_cell.sum(dim=1).max().item())

            telemetry.append({
                "time_sec": round(solver.t, 2),
                "inundated_area_km2": round(inundated_area_km2, 5),
                "active_volume_m3": round(active_volume_m3, 2),
                "water_volume_m3": round(water_volume_m3, 2),
                "debris_volume_m3": round(debris_volume_m3, 2),
                "peak_discharge_m3s": round(peak_discharge_m3s, 2),
            })
            next_log_t += cfg.telemetry_interval_s

        if cfg.max_wallclock_s is not None and (time.time() - wall_start) > cfg.max_wallclock_s:
            warnings.warn(
                f"Stopping early at sim t={solver.t:.1f}s due to wall-clock cap."
            )
            break

    wall_elapsed = time.time() - wall_start
    print(f"      Completed {step_count} steps, sim time {solver.t:.1f}s, "
          f"wall time {wall_elapsed:.1f}s")

    print("[5/7] Packing RGBA hazard texture")
    png_path = out_dir / "flood_packed.png"
    pack_rgba_texture(
        solver.h_max, solver.t_arrival, solver.v_max, solver.Cs,
        cfg, cfg.sim_duration_s, str(png_path),
    )

    print("[6/7] Assessing settlement damage")
    if cfg.bbox_lonlat is not None:
        bounds_lonlat = cfg.bbox_lonlat
    else:
        # Fall back to a bbox around the lake center for the OSM query.
        d = 0.05  # ~5 km in degrees, rough
        bounds_lonlat = (
            cfg.lake.center_lon - d, cfg.lake.center_lat - d,
            cfg.lake.center_lon + d, cfg.lake.center_lat + d,
        )
    settlements = load_settlements(cfg, bounds_lonlat)
    damage_reports = assess_settlement_damage(settlements, solver, transform, utm_crs)

    # ---------------------------------------------------------
    # Infrastructure Damage Assessment (requires osmnx + geopandas)
    # ---------------------------------------------------------
    print("[7/7] Calculating detailed infrastructure damage via OSM & writing metadata")

    _infra_fallback = {
        "hospitals_destroyed": 0, "schools_destroyed": 0,
        "roads_destroyed_km": 0.0, "bridges_destroyed": 0,
        "power_facilities_destroyed": 0, "dams_at_risk": 0,
        "critical_assets_at_risk": [],
    }

    try:
        # Ensure C-contiguous float32 arrays (PyTorch tensors may be non-contiguous)
        import numpy as np
        h_max_np = np.ascontiguousarray(solver.h_max.detach().cpu().numpy(), dtype=np.float32)
        v_max_np = np.ascontiguousarray(solver.v_max.detach().cpu().numpy(), dtype=np.float32)
        # Replace any NaN/inf that can come from dry cells with 0
        np.nan_to_num(h_max_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(v_max_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        infra_damage_summary = calculate_infrastructure_damage(
            h_max_array=h_max_np,
            v_max_array=v_max_np,
            transform=transform,
            utm_crs=utm_crs,
            bounds_lonlat=bounds_lonlat,
            pixel_size_m=pixel_size_m,
        )
    except Exception as _exc:
        import logging
        logging.getLogger(__name__).warning(
            "Infrastructure damage assessment failed (non-fatal): %s", _exc
        )
        infra_damage_summary = _infra_fallback
    # ---------------------------------------------------------

    metadata = {
        "simulation": {
            "dem_path": cfg.dem_path,
            "grid_shape": list(Z_np.shape),
            "pixel_size_m": pixel_size_m,
            "utm_crs": str(utm_crs),
            "sim_duration_s": cfg.sim_duration_s,
            "steps": step_count,
            "wall_clock_s": round(wall_elapsed, 2),
            "manning_n": cfg.manning_n,
            "bulking_factor": cfg.bulking_factor,
            "h_norm_max_m": cfg.h_norm_max_m,
            "v_norm_max_ms": cfg.v_norm_max_ms,
        },
        "dam_breach": dataclasses.asdict(cfg.breach),
        "lake": dataclasses.asdict(cfg.lake),
        "telemetry": telemetry,
        "settlement_damage_reports": damage_reports,
        
        # ---> ADDED THE NEW JSON KEY HERE <---
        "infrastructure_damage_summary": infra_damage_summary, 
        
        "texture_channel_map": {
            "R": "peak_depth / h_norm_max_m",
            "G": "arrival_time / sim_duration_s (255 = never inundated)",
            "B": "peak_velocity / v_norm_max_ms",
            "A": "sediment_debris_fraction",
        },
    }

    json_path = out_dir / "glof_metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done.\n  Texture:  {png_path}\n  Metadata: {json_path}")
    return str(png_path), str(json_path)

# --------------------------------------------------------------------------- #
# Demo entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    demo_cfg = SimConfig(
        dem_path="./data/output_hh.tif",  # IMPORTANT: Change this to your actual file name
        settlements_csv="settlements.csv",                    # Set to None to let OSM auto-fetch Chungthang data
        output_dir="./glof_output",

        # Your exact bounding box (Xmin, Ymin, Xmax, Ymax)
        bbox_lonlat=(88.15, 27.55, 88.70, 27.95),

        lake=LakeConfig(
            mode="radius",
            center_lat=27.913,   # South Lhonak Lake (Y)
            center_lon=88.199,   # South Lhonak Lake (X)
            radius_m=800.0,      
            initial_depth_m=40.0,
        ),

        breach=DamBreachConfig(
            # Positioned slightly east of the lake center, right on the terminal moraine
            dam_lat=27.910,      
            dam_lon=88.205,      
            breach_width_m=100.0,
            breach_depth_m=40.0,
            breach_duration_s=15 * 60, # The moraine collapses over 15 minutes
            breach_start_s=60.0,
        ),

        manning_n=0.045,
        bulking_factor=1.6,          # Debris multiplier (crucial for Sikkim GLOFs)
        sim_duration_s=2 * 3600.0,   # 2 hours of flood simulation
        telemetry_interval_s=30.0,

        # Normalization caps for the PNG image
        h_norm_max_m=20.0,           
        v_norm_max_ms=15.0,

        device="cuda",               # Will use your RTX 4060 Ti
        dtype="float32",
    )

    run_simulation(demo_cfg)