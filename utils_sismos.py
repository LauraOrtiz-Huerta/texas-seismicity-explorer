#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils_sismos.py
===============
Shared helper functions and constants for the Texas Seismicity Explorer.

Used by both:
  - sismos_v1.py  (standalone script)
  - app.py        (Streamlit web application)

Author : Dr. Laura Ortiz-Huerta
Version: v1.0
Date   : March 2026
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.transforms as mtransforms
import requests

warnings.filterwarnings("ignore")

# ============================================================
# APP METADATA
# ============================================================
APP_NAME    = "Texas Seismicity Explorer"
APP_AUTHOR  = "Dr. Laura Ortiz-Huerta"
APP_VERSION = "v1.0"
APP_DATE    = "March 2026"
APP_CREDIT  = f"{APP_NAME} {APP_VERSION} — {APP_AUTHOR}, {APP_DATE}"

# ============================================================
# DATA SOURCES
# ============================================================
TEXNET_CATALOG_MAPSERVER = (
    "https://maps.texnet.beg.utexas.edu/arcgis/rest/services/catalog/catalog_all/MapServer"
)
TEXNET_EQ_LAYER_ID = 0

SRA_MAPSERVER  = "https://maps.texnet.beg.utexas.edu/arcgis/rest/services/sra/sra/MapServer"
SRA_LAYER_ID   = 12
SRA_LINE_COLOR = "#000000"
SRA_LINEWIDTH  = 1.6
SRA_ALPHA      = 0.9

SIR_MAPSERVER = "https://maps.texnet.beg.utexas.edu/arcgis/rest/services/sra/sir/MapServer"
SIR_LAYERS = [(0, "2 km"), (1, "4.5 km"), (2, "9 km"), (3, "15 km"), (4, "25 km")]
SIR_LINEWIDTH = {0: 1.0, 1: 0.8, 2: 0.7, 3: 0.6, 4: 0.5}

BASINS_MAPSERVER = "https://maps.texnet.beg.utexas.edu/arcgis/rest/services/BasinsAndPlays/MapServer"
BASINS_LAYER_ID  = 0

# ============================================================
# MAGNITUDE BINS + PALETTE
# ============================================================
MAG_BINS = [
    ("<1.9",    lambda m: m < 2.0),
    ("2.0–2.9", lambda m: 2.0 <= m < 3.0),
    ("3.0–3.4", lambda m: 3.0 <= m < 3.5),
    ("3.5–3.9", lambda m: 3.5 <= m < 4.0),
    ("4.0–4.9", lambda m: 4.0 <= m < 5.0),
    ("≥5.0",    lambda m: m >= 5.0),
]

MAG_COLORS = {
    "<1.9":    "#7daa65",
    "2.0–2.9": "#b7ed4d",
    "3.0–3.4": "#ffc34b",
    "3.5–3.9": "#ff7f7e",
    "4.0–4.9": "#d62728",
    "≥5.0":    "#bf00ff",
}

STATION_MARKERS = {
    "TexNet Permanent": ("^", "#000000", "#ffffff", "TexNet Permanent"),
    "TexNet Portable":  ("^", "#00bfff", "#ffffff", "TexNet Portable"),
    "Non-TexNet":       ("^", "#cccccc", "#ffffff", "Non-TexNet"),
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def assign_mag_class(mag):
    for label, test in MAG_BINS:
        if test(mag):
            return label
    return "<1.9"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def lonlat_to_mercator(lon, lat):
    x = np.asarray(lon) * 20037508.34 / 180.0
    lat_r = np.radians(np.clip(np.asarray(lat), -85, 85))
    y = np.log(np.tan(np.pi / 4 + lat_r / 2)) * 20037508.34 / np.pi
    return x, y


def mercator_to_lon(x):
    return x * 180.0 / 20037508.34


def mercator_to_lat(y):
    return np.degrees(2 * np.arctan(np.exp(y * np.pi / 20037508.34)) - np.pi / 2)


def _to_epoch_ms(date_str):
    if date_str is None:
        return None
    ts = pd.Timestamp(date_str, tz="UTC")
    return int(ts.value // 1_000_000)


def _patch_requests_ssl():
    """Disable SSL verification globally (corporate proxy workaround)."""
    import ssl
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception:
        pass
    requests.packages.urllib3.disable_warnings()
    _old_send = requests.adapters.HTTPAdapter.send

    def _patched_send(self, request, **kwargs):
        kwargs["verify"] = False
        return _old_send(self, request, **kwargs)

    requests.adapters.HTTPAdapter.send = _patched_send


_patch_requests_ssl()

# ============================================================
# DATA FETCHING
# ============================================================

def fetch_texnet_events_arcgis(
    mapserver_url, layer_id, aoi_lat, aoi_lon, radius_km,
    start_date=None, end_date=None, where="1=1", out_fields=None,
):
    if out_fields is None:
        out_fields = [
            "EventId", "Magnitude", "MagType", "Latitude", "Longitude",
            "Depth", "Event_Date", "EvaluationStatus",
            "MomentMagnitude", "RegionName", "CountyName",
        ]

    query_url = f"{mapserver_url}/{layer_id}/query"
    geom = {"x": float(aoi_lon), "y": float(aoi_lat), "spatialReference": {"wkid": 4326}}

    t0 = _to_epoch_ms(start_date) if start_date else None
    t1 = _to_epoch_ms(end_date) if end_date else None
    time_param = None
    if t0 is not None or t1 is not None:
        left  = "" if t0 is None else str(t0)
        right = "" if t1 is None else str(t1)
        time_param = f"{left},{right}"

    max_per_req = 2000
    offset = 0
    all_rows = []

    while True:
        params = {
            "f": "json", "where": where,
            "outFields": ",".join(out_fields),
            "returnGeometry": "false",
            "geometry": json.dumps(geom),
            "geometryType": "esriGeometryPoint",
            "inSR": "4326", "spatialRel": "esriSpatialRelIntersects",
            "distance": str(float(radius_km)), "units": "esriSRUnit_Kilometer",
            "outSR": "4326",
            "resultOffset": str(offset),
            "resultRecordCount": str(max_per_req),
            "orderByFields": "Event_Date ASC",
        }
        if time_param is not None:
            params["time"] = time_param

        try:
            r = requests.get(query_url, params=params, timeout=60)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"WARNING: request failed at offset={offset}. Retrying… ({e})")
            r = requests.get(query_url, params=params, timeout=90)
            r.raise_for_status()

        data  = r.json()
        feats = data.get("features", []) or []
        if not feats:
            break

        for f in feats:
            all_rows.append(f.get("attributes", {}) or {})

        if len(feats) < max_per_req:
            break
        offset += max_per_req

    return pd.DataFrame(all_rows)


def fetch_arcgis_geojson(mapserver_url, layer_id, where="1=1"):
    qurl  = f"{mapserver_url}/{layer_id}/query"
    params = {
        "f": "geojson", "where": where, "outFields": "*",
        "returnGeometry": "true", "outSR": "3857", "resultRecordCount": 2000,
    }
    r = requests.get(qurl, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_arcgis_geojson_polygons_near_point(
    mapserver_url, layer_id, center_lat, center_lon,
    radius_km=100, where="1=1", out_sr=4326,
):
    query_url = f"{mapserver_url}/{layer_id}/query"
    geom = {"x": float(center_lon), "y": float(center_lat), "spatialReference": {"wkid": 4326}}

    max_per_req = 2000
    offset = 0
    features_all = []

    while True:
        params = {
            "f": "geojson", "where": where, "outFields": "*",
            "returnGeometry": "true", "outSR": str(out_sr),
            "geometry": json.dumps(geom), "geometryType": "esriGeometryPoint",
            "inSR": "4326", "spatialRel": "esriSpatialRelIntersects",
            "distance": str(float(radius_km)), "units": "esriSRUnit_Kilometer",
            "resultOffset": str(offset), "resultRecordCount": str(max_per_req),
        }
        r = requests.get(query_url, params=params, timeout=60)
        r.raise_for_status()
        gj   = r.json()
        feats = gj.get("features", []) or []
        if not feats:
            break
        features_all.extend(feats)
        if len(feats) < max_per_req:
            break
        offset += max_per_req

    rings = []
    for feat in features_all:
        geom2  = feat.get("geometry", {})
        if not geom2:
            continue
        gtype  = geom2.get("type", "")
        coords = geom2.get("coordinates", [])
        polys  = [coords] if gtype == "Polygon" else (coords if gtype == "MultiPolygon" else [])
        for poly in polys:
            if not poly:
                continue
            arr = np.asarray(poly[0], dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 2:
                rings.append(arr)
    return rings


def plot_geojson_lines(ax, gj, color="#ff3b30", lw=1.5, alpha=0.9, zorder=4):
    for feat in gj.get("features", []):
        geom  = feat.get("geometry", {})
        gtype = geom.get("type")
        if gtype == "LineString":
            xs, ys = zip(*geom["coordinates"])
            ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=zorder)
        elif gtype == "MultiLineString":
            for line in geom["coordinates"]:
                xs, ys = zip(*line)
                ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=zorder)


# ============================================================
# DATA PROCESSING
# ============================================================

def process_events_raw(events_raw):
    """Convert raw ArcGIS JSON DataFrame to clean plotting DataFrame."""
    ev = pd.DataFrame()
    ev["Latitude (WGS84)"]  = pd.to_numeric(events_raw.get("Latitude"),  errors="coerce")
    ev["Longitude (WGS84)"] = pd.to_numeric(events_raw.get("Longitude"), errors="coerce")
    ev["Local Magnitude"]   = pd.to_numeric(events_raw.get("Magnitude"),        errors="coerce")
    ev["Moment Magnitude"]  = pd.to_numeric(events_raw.get("MomentMagnitude"),  errors="coerce")
    ev["Depth of Hypocenter (Km.  Rel to MSL)"] = pd.to_numeric(
        events_raw.get("Depth"), errors="coerce"
    )
    ev["Evaluation Status"] = events_raw.get("EvaluationStatus")
    ev["EventId"]           = events_raw.get("EventId")
    ev["RegionName"]        = events_raw.get("RegionName")
    ev["CountyName"]        = events_raw.get("CountyName")

    dt = (
        pd.to_datetime(events_raw.get("Event_Date"), unit="ms", utc=True, errors="coerce")
        .dt.tz_convert(None)
    )
    ev["DateTime"]    = dt
    ev["Origin Date"] = ev["DateTime"].dt.strftime("%Y-%m-%d")
    ev["Origin Time"] = ev["DateTime"].dt.strftime("%H:%M:%S")

    for col in ["Latitude (WGS84)", "Longitude (WGS84)", "Local Magnitude", "Moment Magnitude",
                "Depth of Hypocenter (Km.  Rel to MSL)"]:
        ev[col] = pd.to_numeric(ev[col], errors="coerce")

    ev["Magnitude"]    = ev["Local Magnitude"].fillna(ev["Moment Magnitude"])
    ev = ev.dropna(subset=["Latitude (WGS84)", "Longitude (WGS84)", "Magnitude", "DateTime"]).copy()
    ev["Depth_MSL_km"] = ev["Depth of Hypocenter (Km.  Rel to MSL)"]
    ev["Year"]         = ev["DateTime"].dt.year
    ev["MagClass"]     = ev["Magnitude"].apply(assign_mag_class)
    ev["MagColor"]     = ev["MagClass"].map(MAG_COLORS)
    return ev


def filter_final_events(events):
    """Keep only 'final' evaluation status; fall back to all if none found."""
    if "Evaluation Status" in events.columns:
        mask = events["Evaluation Status"].astype(str).str.lower() == "final"
        if mask.any():
            return events[mask].copy()
        print("  WARNING: No 'final' events found — using all evaluation statuses.")
    return events.copy()


def process_stations(stations_df, centroid_lat, centroid_lon,
                     analysis_radius_km, map_radius_km, event_start, event_end):
    """Filter stations by distance and clip operation windows."""
    stations_df = stations_df.copy()
    stations_df["Latitude (WGS84)"]  = pd.to_numeric(stations_df["Latitude (WGS84)"],  errors="coerce")
    stations_df["Longitude (WGS84)"] = pd.to_numeric(stations_df["Longitude (WGS84)"], errors="coerce")
    stations_df = stations_df.dropna(subset=["Latitude (WGS84)", "Longitude (WGS84)"])

    stations_df["dist_km"] = stations_df.apply(
        lambda r: haversine_km(centroid_lat, centroid_lon,
                               r["Latitude (WGS84)"], r["Longitude (WGS84)"]),
        axis=1,
    )

    nearby = stations_df[stations_df["dist_km"] <= analysis_radius_km].copy()
    st_map = stations_df[stations_df["dist_km"] <= map_radius_km].copy()

    for df_st in (nearby, st_map):
        df_st["Start Date"] = pd.to_datetime(df_st["Start Date"], errors="coerce")
        df_st["End Date"]   = pd.to_datetime(df_st["End Date"],   errors="coerce")

    nearby["End Date"]    = nearby["End Date"].fillna(event_end)
    nearby["Start Clip"]  = nearby["Start Date"].clip(lower=event_start, upper=event_end)
    nearby["End Clip"]    = nearby["End Date"].clip(lower=event_start, upper=event_end)
    nearby = nearby[
        nearby["End Clip"].notna() & nearby["Start Clip"].notna() &
        (nearby["End Clip"] >= event_start) & (nearby["Start Clip"] <= event_end)
    ].copy()

    return nearby, st_map


# ============================================================
# STATIC FIGURE GENERATION  (4-panel matplotlib)
# ============================================================

def generate_static_figure(events, events_map, nearby_stations, stations_map, params):
    """
    Generate the 4-panel publication-quality seismic figure.

    Parameters
    ----------
    events          : pd.DataFrame  – AOI events (used in Panels B, C, D)
    events_map      : pd.DataFrame  – broader map events (Panel A)
    nearby_stations : pd.DataFrame  – analysis stations (Panels B, D)
    stations_map    : pd.DataFrame  – map-only stations (Panel A)
    params          : dict with keys:
        AOI_LAT, AOI_LON, AOI_RADIUS_KM, MAP_RADIUS_KM,
        STATIONS_ANALYSIS_RADIUS_KM, STATIONS_MAP_RADIUS_KM,
        STATIONS_CENTER_MODE, AREA_NAME,
        centroid_lat, centroid_lon

    Returns
    -------
    matplotlib.figure.Figure
    """
    import contextily as ctx

    AOI_LAT   = params["AOI_LAT"]
    AOI_LON   = params["AOI_LON"]
    AOI_RADIUS_KM = params["AOI_RADIUS_KM"]
    MAP_RADIUS_KM = params["MAP_RADIUS_KM"]
    STATIONS_ANALYSIS_RADIUS_KM = params["STATIONS_ANALYSIS_RADIUS_KM"]
    STATIONS_MAP_RADIUS_KM      = params["STATIONS_MAP_RADIUS_KM"]
    STATIONS_CENTER_MODE        = params.get("STATIONS_CENTER_MODE", "AOI")
    AREA_NAME     = params.get("AREA_NAME", "Area")
    centroid_lat  = params.get("centroid_lat", AOI_LAT)
    centroid_lon  = params.get("centroid_lon", AOI_LON)

    if STATIONS_CENTER_MODE.upper() == "AOI":
        center_label = "AOI center"
        center_lat, center_lon = AOI_LAT, AOI_LON
    else:
        center_label = "seismic centroid"
        center_lat, center_lon = centroid_lat, centroid_lon

    date_min = events["DateTime"].min().strftime("%Y-%m-%d")
    date_max = events["DateTime"].max().strftime("%Y-%m-%d")
    max_mag  = events["Magnitude"].max()

    # ---- Dark theme ----
    plt.rcParams.update({
        "figure.facecolor": "#0d1117", "axes.facecolor": "#0d1117",
        "axes.edgecolor": "#30363d", "axes.labelcolor": "#c9d1d9",
        "xtick.color": "#8b949e", "ytick.color": "#8b949e",
        "text.color": "#c9d1d9", "grid.color": "#21262d",
        "grid.alpha": 0.5, "font.family": "sans-serif", "font.size": 9,
    })

    fig = plt.figure(figsize=(26, 20), facecolor="#0d1117")
    gs  = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[1.4, 1.35], width_ratios=[1, 1, 0.7],
        hspace=0.28, wspace=0.25,
        left=0.05, right=0.92, top=0.93, bottom=0.06,
    )

    # ----------------------------------------------------------------
    # PANEL A — MAP
    # ----------------------------------------------------------------
    ax_map = fig.add_subplot(gs[0, 0:2])

    events_for_map = events_map if len(events_map) > 0 else events

    ev_mx, ev_my = lonlat_to_mercator(
        events_for_map["Longitude (WGS84)"].values,
        events_for_map["Latitude (WGS84)"].values,
    )
    if len(stations_map) > 0:
        st_mx, st_my = lonlat_to_mercator(
            stations_map["Longitude (WGS84)"].values,
            stations_map["Latitude (WGS84)"].values,
        )
    else:
        st_mx = np.array([])
        st_my = np.array([])

    order       = np.argsort(events_for_map["Magnitude"].values)
    ev_mx_s     = ev_mx[order]
    ev_my_s     = ev_my[order]
    mag_s       = events_for_map["Magnitude"].values[order]
    colors_s    = events_for_map["MagColor"].values[order]
    sizes       = np.clip((np.abs(mag_s) ** 2.2) * 4, 3, None)

    ax_map.scatter(ev_mx_s, ev_my_s, c=colors_s, s=sizes,
                   alpha=0.6, edgecolors="white", linewidths=0.3, zorder=5)

    # AOI events on top
    ev_in_mx, ev_in_my = lonlat_to_mercator(
        events["Longitude (WGS84)"].values, events["Latitude (WGS84)"].values
    )
    order_in   = np.argsort(events["Magnitude"].values)
    ev_in_mx   = ev_in_mx[order_in]
    ev_in_my   = ev_in_my[order_in]
    mag_in     = events["Magnitude"].values[order_in]
    col_in     = events["MagColor"].values[order_in]
    sizes_in   = np.clip((np.abs(mag_in) ** 2.2) * 4, 3, None)
    ax_map.scatter(ev_in_mx, ev_in_my, c=col_in, s=sizes_in,
                   alpha=0.6, edgecolors="white", linewidths=0.3, zorder=6)

    # Magnitude labels
    MAG_LABEL_THRESHOLD = 3.0
    MAG_LABEL_COLOR     = "#3a3a3a"
    for xi, yi, mi in zip(ev_mx_s, ev_my_s, mag_s):
        if mi >= MAG_LABEL_THRESHOLD:
            kw = dict(fontsize=3.5, color=MAG_LABEL_COLOR, ha="center",
                      va="center_baseline", zorder=7)
            if mi >= 4.0:
                kw.update(fontweight="bold",
                          path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
            ax_map.text(xi, yi, f"{mi:.1f}", **kw)

    for xi, yi, mi in zip(ev_in_mx, ev_in_my, mag_in):
        if mi >= MAG_LABEL_THRESHOLD:
            kw = dict(fontsize=3.5, color=MAG_LABEL_COLOR, ha="center",
                      va="center_baseline", zorder=9)
            if mi >= 4.0:
                kw.update(fontweight="bold",
                          path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
            ax_map.text(xi, yi, f"{mi:.1f}", **kw)

    # Stations
    for stype, (marker, facecolor, edgecolor, label) in STATION_MARKERS.items():
        mask = stations_map["Station Type"] == stype
        if mask.any():
            smx, smy = lonlat_to_mercator(
                stations_map.loc[mask, "Longitude (WGS84)"].values,
                stations_map.loc[mask, "Latitude (WGS84)"].values,
            )
            ax_map.scatter(smx, smy, marker=marker, c=facecolor, s=70,
                           edgecolors=edgecolor, linewidths=0.8, zorder=6, label=label, alpha=0.95)
            codes = stations_map.loc[mask, "Station Code"].astype(str).fillna("").values
            for x, y, code in zip(smx, smy, codes):
                if code and code.lower() != "nan":
                    ax_map.text(x, y, f"\n\n{code}", fontsize=5, color="#e6edf3",
                                ha="center", va="top", zorder=7,
                                path_effects=[pe.withStroke(linewidth=1, foreground="#0d1117")])

    # AOI circle
    aoi_x, aoi_y = lonlat_to_mercator(AOI_LON, AOI_LAT)
    aoi_radius_m = AOI_RADIUS_KM * 1000.0
    ax_map.add_patch(mpatches.Circle(
        (aoi_x, aoi_y), radius=aoi_radius_m,
        fill=False, edgecolor="#ff3b3b", linewidth=2.0,
        linestyle="-", alpha=0.9, zorder=4,
    ))
    ax_map.plot(aoi_x, aoi_y, marker="+", markersize=6, markeredgecolor="#ff3b3b",
                markerfacecolor="none", markeredgewidth=1.0, linestyle="none", zorder=20,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    # Map extent
    all_x = np.concatenate([ev_mx, st_mx, [aoi_x - aoi_radius_m, aoi_x + aoi_radius_m]]) if len(st_mx) else np.concatenate([ev_mx, [aoi_x - aoi_radius_m, aoi_x + aoi_radius_m]])
    all_y = np.concatenate([ev_my, st_my, [aoi_y - aoi_radius_m, aoi_y + aoi_radius_m]]) if len(st_my) else np.concatenate([ev_my, [aoi_y - aoi_radius_m, aoi_y + aoi_radius_m]])
    xm = (all_x.max() - all_x.min()) * 0.08
    ym = (all_y.max() - all_y.min()) * 0.08
    ax_map.set_xlim(all_x.min() - xm, all_x.max() + xm)
    ax_map.set_ylim(all_y.min() - ym, all_y.max() + ym)

    # Basemap
    try:
        ctx.add_basemap(ax_map, source=ctx.providers.Esri.WorldImagery,
                        zoom="auto", attribution="")
    except Exception as e:
        print(f"  Basemap failed ({e}), continuing without tiles.")

    # Basins & Plays
    try:
        basins_rings = fetch_arcgis_geojson_polygons_near_point(
            BASINS_MAPSERVER, BASINS_LAYER_ID,
            center_lat=AOI_LAT, center_lon=AOI_LON,
            radius_km=max(150, MAP_RADIUS_KM), where="1=1", out_sr=3857,
        )
        for ring in basins_rings:
            ax_map.plot(ring[:, 0], ring[:, 1], color="#a5b4fc", linewidth=1.0, alpha=0.35, zorder=6)
    except Exception as e:
        print(f"  Basins & Plays failed ({e})")

    # SIR layers
    for sir_id, sir_label in SIR_LAYERS:
        try:
            sir_gj = fetch_arcgis_geojson(SIR_MAPSERVER, sir_id)
            plot_geojson_lines(ax_map, sir_gj, color="#d1d5db",
                               lw=SIR_LINEWIDTH.get(sir_id, 1.2), alpha=0.8, zorder=7)
        except Exception as e:
            print(f"  SIR {sir_id} failed ({e})")

    # SRA layer
    try:
        sra_gj = fetch_arcgis_geojson(SRA_MAPSERVER, SRA_LAYER_ID)
        plot_geojson_lines(ax_map, sra_gj, color=SRA_LINE_COLOR,
                           lw=SRA_LINEWIDTH, alpha=SRA_ALPHA, zorder=10)
    except Exception as e:
        print(f"  SRA failed ({e})")

    # Tick labels
    x_ticks = np.linspace(ax_map.get_xlim()[0], ax_map.get_xlim()[1], 6)
    y_ticks = np.linspace(ax_map.get_ylim()[0], ax_map.get_ylim()[1], 6)
    ax_map.set_xticks(x_ticks)
    ax_map.set_yticks(y_ticks)
    ax_map.set_xticklabels([f"{mercator_to_lon(x):.2f}°" for x in x_ticks], fontsize=7)
    ax_map.set_yticklabels([f"{mercator_to_lat(y):.2f}°" for y in y_ticks], fontsize=7)
    ax_map.set_xlabel("Longitude", fontsize=10, labelpad=6)
    ax_map.set_ylabel("Latitude", fontsize=10, labelpad=6)
    ax_map.grid(True, alpha=0.15, linestyle="--", linewidth=0.3, color="#c7d2fe")

    # Legend
    mag_handles = [
        mlines.Line2D([], [], marker="o", color="none", markerfacecolor=MAG_COLORS[lbl],
                      markeredgecolor="white", markeredgewidth=0.3, markersize=7, label=lbl, alpha=0.8)
        for lbl, _ in MAG_BINS
    ]
    st_handles = [
        mlines.Line2D([], [], marker=mk, color="none", markerfacecolor=fc,
                      markeredgecolor=ec, markeredgewidth=0.8, markersize=8, label=lb)
        for _, (mk, fc, ec, lb) in STATION_MARKERS.items()
    ]
    all_handles = mag_handles + [mlines.Line2D([], [], color="none", label="")] + st_handles
    leg = ax_map.legend(
        all_handles, [h.get_label() for h in all_handles],
        loc="center left", bbox_to_anchor=(1.05, 0.5), bbox_transform=ax_map.transAxes,
        fontsize=7, ncol=1, framealpha=0.85, facecolor="#1a1a2e", edgecolor="#3a3a5a",
        labelcolor="#d0d0e0", title="Magnitude / Station Type", title_fontsize=8,
        borderaxespad=0,
    )
    leg.get_title().set_color("#c7d2fe")

    ax_map.set_title("A — Seismic Events & Nearby Stations",
                     fontsize=13, fontweight="bold", color="#e6edf3", pad=12)
    ax_map.text(
        0.01, 0.01,
        f"Stations within {STATIONS_MAP_RADIUS_KM} km of {center_label} "
        f"({center_lat:.3f}°N, {abs(center_lon):.3f}°W)",
        transform=ax_map.transAxes, fontsize=7, color="#8b949e", ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22", edgecolor="#30363d", alpha=0.8),
    )
    ax_map.set_aspect("equal", adjustable="box")

    # ----------------------------------------------------------------
    # PANEL D — SUMMARY
    # ----------------------------------------------------------------
    ax_summary = fig.add_subplot(gs[0, 2])
    ax_summary.set_xlim(0, 1)
    ax_summary.set_ylim(0, 1)
    ax_summary.axis("off")
    ax_summary.add_patch(mpatches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.02", linewidth=1,
        edgecolor="#30363d", facecolor="#161b22", alpha=0.6,
        transform=ax_summary.transAxes, zorder=0,
    ))

    mean_depth = events["Depth_MSL_km"].mean()
    years_str  = ", ".join(str(y) for y in sorted(events["Year"].unique()))

    null_depth_per_year = events[events["Depth_MSL_km"].isna()].groupby("Year").size()
    total_null_depth    = int(events["Depth_MSL_km"].isna().sum())
    null_str = (", ".join(f"{int(yr)}: {int(cnt)}" for yr, cnt in null_depth_per_year.items())
                if total_null_depth > 0 else "none")

    summary_lines = [
        ("D — Summary",   16, "bold",   "#e6edf3"),
        ("",               6, "normal", "#8b949e"),
        (f"Total events:  {len(events)}",                                 11, "normal", "#c9d1d9"),
        (f"Date range:  {date_min}  →  {date_max}",                       11, "normal", "#c9d1d9"),
        (f"Years:  {years_str}",                                           11, "normal", "#c9d1d9"),
        (f"Max magnitude:  {max_mag:.1f}",                                 11, "normal", "#c9d1d9"),
        (f"Mean depth (MSL):  {mean_depth:.1f} km",                        11, "normal", "#c9d1d9"),
        (f"Events without depth:  {total_null_depth}",                     11, "normal", "#ffc34b"),
        (f"  (by year)  {null_str}",                                       10, "normal", "#ffc34b"),
        (f"Stations in analysis (≤{STATIONS_ANALYSIS_RADIUS_KM} km of {center_label}):  {len(nearby_stations)}",
                                                                           11, "normal", "#c9d1d9"),
        (f"Station radius (analysis):  {STATIONS_ANALYSIS_RADIUS_KM} km", 11, "normal", "#c9d1d9"),
        (f"Station radius (map):  {STATIONS_MAP_RADIUS_KM} km",           11, "normal", "#c9d1d9"),
        (f"Centroid:  {centroid_lat:.3f}N, {abs(centroid_lon):.3f}W",     11, "normal", "#c9d1d9"),
        ("",               6, "normal", "#8b949e"),
        ("Evaluation Status: final",                                       10, "italic", "#8b949e"),
        (f"Source: TexNet ArcGIS JSON query (AOI {AOI_RADIUS_KM} km)",     9, "italic", "#8b949e"),
    ]
    y_pos = 0.92
    for text, size, weight, color in summary_lines:
        fs = "italic" if weight == "italic" else "normal"
        fw = "normal" if weight == "italic" else weight
        ax_summary.text(0.08, y_pos, text, fontsize=size, fontweight=fw, fontstyle=fs,
                        color=color, transform=ax_summary.transAxes, va="top")
        y_pos -= 0.065

    # ----------------------------------------------------------------
    # PANEL B — TIME vs DEPTH
    # ----------------------------------------------------------------
    ax_td = fig.add_subplot(gs[1, 0:2])

    for label, _ in MAG_BINS:
        mask = events["MagClass"] == label
        if mask.any():
            sub   = events[mask]
            _sz   = np.clip((np.abs(sub["Magnitude"].values) ** 2.2) * 4, 3, None)
            ax_td.scatter(sub["DateTime"], sub["Depth_MSL_km"],
                          c=MAG_COLORS[label], s=_sz, alpha=0.6,
                          edgecolors="white", linewidths=0.2, zorder=5, label=label)

    _labeled = events[events["Magnitude"] >= 3.0].dropna(subset=["Depth_MSL_km"])
    for _, row in _labeled.iterrows():
        mi, xi, yi = row["Magnitude"], row["DateTime"], row["Depth_MSL_km"]
        kw = dict(fontsize=3.5, ha="center", va="center_baseline", zorder=7)
        if mi >= 4.0:
            kw.update(color="#3a3a3a", fontweight="bold",
                      path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
        else:
            kw.update(color="black", fontweight="normal",
                      path_effects=[pe.withStroke(linewidth=0.7, foreground="white")])
        ax_td.text(xi, yi, f"{mi:.1f}", **kw)

    ax_td.invert_yaxis()

    _yr_min = int(events["Year"].min())
    _yr_max = int(events["Year"].max()) + 1
    for _yr in range(_yr_min, _yr_max + 1):
        _jan1 = pd.Timestamp(f"{_yr}-01-01")
        ax_td.axvline(_jan1, color="#8b949e", linestyle="--", linewidth=0.7, alpha=0.35, zorder=3)
        ax_td.text(_jan1, 1.0, f" {_yr}", transform=ax_td.get_xaxis_transform(),
                   fontsize=6, color="#8b949e", alpha=0.7, va="bottom", ha="left")

    if len(nearby_stations) > 0:
        depth_min = events["Depth_MSL_km"].min()
        seg_y_start = depth_min - 1.0
        seg_spacing = 0.4
        ns_sorted = nearby_stations.sort_values("Start Date").reset_index(drop=True)
        for i, (_, row) in enumerate(ns_sorted.iterrows()):
            stype  = row.get("Station Type", "Non-TexNet")
            color  = STATION_MARKERS.get(stype, ("^", "#cccccc", "#ffffff", ""))[1]
            start  = row["Start Clip"]
            end    = row["End Clip"]
            if pd.isna(start):
                continue
            y_seg = seg_y_start - i * seg_spacing
            ax_td.plot([start, end], [y_seg, y_seg],
                       color=color, linewidth=2, alpha=0.7, solid_capstyle="round", zorder=4)
            ax_td.text(end, y_seg, f" {row.get('Station Code', '')}",
                       fontsize=5, color=color, va="center", ha="left", alpha=0.8, zorder=6)

    ax_td.set_xlabel("Date", fontsize=10, labelpad=6)
    ax_td.set_ylabel("Depth (km, rel. to MSL)", fontsize=10, labelpad=6)
    ax_td.grid(True, alpha=0.2, linestyle="--", linewidth=0.3)
    ax_td.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_td.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax_td.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    leg_td = ax_td.legend(fontsize=6, loc="lower right", ncol=3, framealpha=0.7,
                          facecolor="#1a1a2e", edgecolor="#3a3a5a",
                          labelcolor="#d0d0e0", title="Magnitude Class", title_fontsize=7)
    leg_td.get_title().set_color("#c7d2fe")
    ax_td.set_title("B — Time vs Depth", fontsize=13, fontweight="bold",
                    color="#e6edf3", pad=12)

    # ----------------------------------------------------------------
    # PANEL C — YEARLY DEPTH HISTOGRAMS
    # ----------------------------------------------------------------
    years  = sorted(events["Year"].unique())
    n_years = len(years)
    gs_hist = gridspec.GridSpecFromSubplotSpec(n_years, 1, subplot_spec=gs[1, 2], hspace=0.60)
    hist_axes_list = []

    d_min_all = events["Depth_MSL_km"].dropna().min() if events["Depth_MSL_km"].notna().any() else 0.0
    d_max_all = events["Depth_MSL_km"].dropna().max() if events["Depth_MSL_km"].notna().any() else 30.0
    if d_min_all == d_max_all:
        d_min_all -= 1.0
        d_max_all += 1.0
    bins = np.linspace(d_min_all, d_max_all, 20)

    xmax_global = 1
    for y in years:
        yd = events.loc[events["Year"] == y, "Depth_MSL_km"].dropna()
        if len(yd):
            c, _ = np.histogram(yd, bins=bins)
            xmax_global = max(xmax_global, int(c.max()))
    xmax_global = int(np.ceil(xmax_global * 1.05))

    hist_label_data = []

    for idx, year in enumerate(years):
        ax_h = fig.add_subplot(gs_hist[idx])
        hist_axes_list.append(ax_h)
        year_data = events.loc[events["Year"] == year, "Depth_MSL_km"].dropna()

        if len(year_data) == 0:
            ax_h.text(0.5, 0.5, f"{year}\nNo data", transform=ax_h.transAxes,
                      ha="center", va="center", fontsize=8, color="#8b949e")
            ax_h.axis("off")
            continue

        mu, sigma = year_data.mean(), year_data.std()
        counts, bin_edges = np.histogram(year_data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax_h.barh(bin_centers, counts, height=(bin_edges[1] - bin_edges[0]) * 0.85,
                  color="#6366f1", alpha=0.6, edgecolor="#818cf8", linewidth=0.3)
        ax_h.set_xlim(0, xmax_global)

        line_styles = [
            (mu + 2 * sigma, ":", 0.6, f"+2σ={mu + 2 * sigma:.1f}"),
            (mu + sigma,     "--", 0.8, f"+1σ={mu + sigma:.1f}"),
            (mu,             "-",  1.5, f"μ={mu:.1f}"),
            (mu - sigma,     "--", 0.8, f"−1σ={mu - sigma:.1f}"),
            (mu - 2 * sigma, ":", 0.6, f"−2σ={mu - 2 * sigma:.1f}"),
        ]
        valid_lines = []
        for depth_val, ls, lw, lbl in line_styles:
            if d_min_all <= depth_val <= d_max_all:
                ax_h.axhline(depth_val, color="#ff6b6b", linestyle=ls,
                             linewidth=lw, alpha=0.8, zorder=10)
                valid_lines.append((depth_val, lbl))
        hist_label_data.append((ax_h, valid_lines, d_min_all, d_max_all))

        ax_h.invert_yaxis()
        ax_h.set_ylabel("Depth (km)", fontsize=6, labelpad=2)
        ax_h.set_xlabel("")
        ax_h.tick_params(labelbottom=(idx % 2 == 0), labelsize=5.5)
        ax_h.grid(True, alpha=0.15, linestyle="--", linewidth=0.2)
        n_nan_year = int(events.loc[events["Year"] == year, "Depth_MSL_km"].isna().sum())
        nan_tag = f" (+{n_nan_year} sin prof.)" if n_nan_year > 0 else ""
        ax_h.set_title(f"{year}  (n={len(year_data)}{nan_tag})",
                       fontsize=8, fontweight="bold", color="#c9d1d9", pad=3)

    # Layout finalize → second pass for labels
    fig.canvas.draw()
    MIN_GAP_PT = 9.0

    for _ax, _vlines, _dmin, _dmax in hist_label_data:
        if not _vlines:
            continue
        _trans = mtransforms.blended_transform_factory(_ax.transAxes, _ax.transData)

        def _to_disp(d, ax=_ax):
            return ax.transData.transform((0, d))[1]

        def _to_data(py, ax=_ax):
            return ax.transData.inverted().transform((0, py))[1]

        _min_gap_px = MIN_GAP_PT * (fig.dpi / 72.0)
        _ylim = _ax.get_ylim()
        _py_lo = min(_to_disp(_ylim[0]), _to_disp(_ylim[1]))
        _py_hi = max(_to_disp(_ylim[0]), _to_disp(_ylim[1]))
        _items = sorted([(_to_disp(d), d, lbl) for d, lbl in _vlines], key=lambda x: x[0])
        _adj = [float(np.clip(it[0], _py_lo, _py_hi)) for it in _items]

        for _i in range(1, len(_adj)):
            if _adj[_i] - _adj[_i - 1] < _min_gap_px:
                _adj[_i] = min(_py_hi, _adj[_i - 1] + _min_gap_px)
        for _i in range(len(_adj) - 2, -1, -1):
            if _adj[_i + 1] - _adj[_i] < _min_gap_px:
                _adj[_i] = max(_py_lo, _adj[_i + 1] - _min_gap_px)
        _adj = [float(np.clip(p, _py_lo, _py_hi)) for p in _adj]

        for (_, _dr, _lbl), _py in zip(_items, _adj):
            _ax.text(1.02, _to_data(_py), _lbl, transform=_trans,
                     fontsize=5.5, color="#ff6b6b", va="center", ha="left", clip_on=False,
                     bbox=dict(boxstyle="round,pad=0.10", facecolor="#0d1117",
                               edgecolor="none", alpha=0.70), zorder=20)

    # Panel C title & x-label
    x_left   = min(ax.get_position().x0 for ax in hist_axes_list)
    x_right  = max(ax.get_position().x1 for ax in hist_axes_list)
    y_bottom = min(ax.get_position().y0 for ax in hist_axes_list)
    y_top    = max(ax.get_position().y1 for ax in hist_axes_list)
    x_center = 0.5 * (x_left + x_right)

    fig.text(x_center, y_bottom - 0.02, "Count",
             ha="center", va="top", fontsize=8, color="#c9d1d9")
    fig.text(x_center, min(0.98, y_top + 0.015), "C — Yearly Depth Histograms",
             fontsize=13, fontweight="bold", color="#e6edf3", ha="center", va="bottom")

    # ----------------------------------------------------------------
    # SUPER TITLE + BRANDING
    # ----------------------------------------------------------------
    fig.suptitle(f"{AREA_NAME} Seismic Analysis — TexNet",
                 fontsize=20, fontweight="bold", color="#e6edf3", y=0.975)
    fig.text(0.5, 0.955,
             f"{len(events)} events  ·  {date_min} to {date_max}  ·  "
             f"Magnitudes {events['Magnitude'].min():.1f}–{max_mag:.1f}",
             ha="center", fontsize=10, color="#8b949e")

    # ----------------------------------------------------------------
    # TOP-LEFT BRANDING
    # ----------------------------------------------------------------
    import matplotlib.image as mpimg
    icon_path = os.path.join(os.path.dirname(__file__), "TexasSeismicExplorer_icon_2.png")
    
    title_str = APP_NAME
    subtitle_str = f"{APP_AUTHOR} | {APP_VERSION.upper()}"
    
    if os.path.exists(icon_path):
        try:
            icon_img = mpimg.imread(icon_path)
            ax_icon = fig.add_axes([0.02, 0.935, 0.045, 0.045], anchor='NW', zorder=10)
            ax_icon.imshow(icon_img)
            ax_icon.axis('off')
            
            fig.text(0.068, 0.962, title_str,
                     fontsize=15, fontweight="bold", color="#e6edf3", ha="left", va="center")
            fig.text(0.068, 0.942, subtitle_str,
                     fontsize=10, color="#8b949e", ha="left", va="center")
        except Exception as e:
            print(f"  Could not render icon: {e}")
            fig.text(0.02, 0.962, title_str,
                     fontsize=15, fontweight="bold", color="#e6edf3", ha="left", va="center")
            fig.text(0.02, 0.942, subtitle_str,
                     fontsize=10, color="#8b949e", ha="left", va="center")
    else:
        fig.text(0.02, 0.962, title_str,
                 fontsize=15, fontweight="bold", color="#e6edf3", ha="left", va="center")
        fig.text(0.02, 0.942, subtitle_str,
                 fontsize=10, color="#8b949e", ha="left", va="center")

    return fig
