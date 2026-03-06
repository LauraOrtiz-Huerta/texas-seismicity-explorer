#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app.py — Texas Seismicity Explorer
====================================
Streamlit web application for interactive seismic analysis using the
TexNet earthquake catalog. Combines a Folium interactive map (Panel A)
with a Plotly time-depth scatter chart (Panel B), plus the original
4-panel static figure available for PNG/PDF download.

Author : Dr. Laura Ortiz-Huerta
Version: v1.0
Date   : March 2026
"""

import os
import sys
import io
import warnings
from datetime import date, datetime

# Fix Windows terminal encoding
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

warnings.filterwarnings("ignore")
os.environ["CURL_CA_BUNDLE"]     = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# Page config MUST be first Streamlit call
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Texas Seismicity Explorer",
    page_icon=os.path.join(os.path.dirname(__file__), "TexasSeismicExplorer_icon_1.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

import folium
from folium.plugins import MeasureControl, MarkerCluster
from streamlit_folium import st_folium
import plotly.graph_objects as go

from utils_sismos import (
    APP_NAME, APP_AUTHOR, APP_VERSION, APP_DATE, APP_CREDIT,
    MAG_BINS, MAG_COLORS, STATION_MARKERS,
    TEXNET_CATALOG_MAPSERVER, TEXNET_EQ_LAYER_ID,
    SRA_MAPSERVER, SRA_LAYER_ID,
    SIR_MAPSERVER, SIR_LAYERS,
    BASINS_MAPSERVER, BASINS_LAYER_ID,
    fetch_texnet_events_arcgis, fetch_arcgis_geojson,
    fetch_arcgis_geojson_polygons_near_point,
    process_events_raw, filter_final_events, process_stations,
    generate_static_figure, assign_mag_class, haversine_km,
)

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
DATA_DIR     = os.path.dirname(__file__)
FIGURES_DIR  = os.path.join(DATA_DIR, "figures")
STATIONS_CSV = os.path.join(DATA_DIR, "texnet_stations.csv")
os.makedirs(FIGURES_DIR, exist_ok=True)

DEFAULT_LAT         = 32.325
DEFAULT_LON         = -101.789
DEFAULT_AREA_NAME   = "Example"
DEFAULT_AOI_KM      = 25
DEFAULT_MAP_KM      = 80
DEFAULT_STA_ANLY_KM = 25
DEFAULT_STA_MAP_KM  = 80

# ---------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------
for key, val in {
    "results": None,
    "selected_event_id": None,
    "analysis_done": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ===============================================================
# HEADER
# ===============================================================
col_logo, col_title = st.columns([0.18, 0.82])
with col_logo:
    icon_path = os.path.join(os.path.dirname(__file__), "TexasSeismicExplorer_icon_2.png")
    if os.path.exists(icon_path):
        st.image(icon_path, use_container_width=True)
    else:
        st.markdown("## 🌎")
with col_title:
    st.markdown(f"# {APP_NAME}")
    st.markdown(
        f"<span style='color:#8b949e; font-size:0.9rem;'>"
        f"<b>{APP_AUTHOR}</b>&nbsp;&nbsp;|&nbsp;&nbsp;{APP_VERSION}</span>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# Attribution
with st.expander("📋 Data Sources & Attribution", expanded=False):
    st.markdown(
        """
**Earthquake catalog & seismic stations:**
[TexNet Earthquake Catalog](https://catalog.texnet.beg.utexas.edu/)
— Bureau of Economic Geology (BEG), The University of Texas at Austin.

**Seismic Response Areas (SRA):**
Texas Railroad Commission (RRC) — boundaries delineating areas of
seismic response under the RRC's Oil and Gas Division.

**Texas Basins and Plays:**
Bureau of Economic Geology (BEG), The University of Texas at Austin
— geologic basin and play outlines for the state of Texas.
        """
    )

# ===============================================================
# SIDEBAR — INPUT FORM
# ===============================================================
with st.sidebar:
    # --- Center the logo in the sidebar ---
    _sb_col1, _sb_col2, _sb_col3 = st.columns([1, 2, 1])
    with _sb_col2:
        icon_path_sb = os.path.join(os.path.dirname(__file__), "TexasSeismicExplorer_icon_2.png")
        if os.path.exists(icon_path_sb):
            st.image(icon_path_sb, use_container_width=True)
            
    st.markdown(
        f"<div style='text-align:center; color:#c7d2fe; font-weight:bold; font-size:1.05rem;'>"
        f"{APP_NAME}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='text-align:center; color:#8b949e; font-size:0.78rem; margin-bottom:12px;'>"
        f"{APP_AUTHOR} · {APP_VERSION}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # --- Area ---
    st.subheader("📍 Area of Interest")
    area_name = st.text_input("Area Name", value=DEFAULT_AREA_NAME,
                              help="Label used in figure titles and output filenames")
    aoi_lat = st.number_input("Center Latitude (°N)", value=DEFAULT_LAT, format="%.4f",
                              help="Decimal degrees, positive northward (e.g. 32.3250)")
    aoi_lon = st.number_input("Center Longitude (°W, enter as negative)",
                              value=DEFAULT_LON, format="%.4f",
                              help="Decimal degrees, negative westward (e.g. -101.7890)")

    st.markdown("---")

    # --- Radii ---
    st.subheader("🔵 Search Radii")
    st.caption("**AOI Radius** — defines the geographic zone used to download earthquakes for analysis (Panels B, C & D). Smaller values focus on a tighter area.")
    _aoi_preset = st.radio(
        "Quick select:", ["9.08 km", "25 km", "Custom..."],
        horizontal=True, index=1, key="aoi_preset",
    )
    if _aoi_preset == "9.08 km":
        aoi_radius = 9.08
    elif _aoi_preset == "25 km":
        aoi_radius = 25.0
    else:
        aoi_radius = st.number_input(
            "Custom AOI Radius (km)",
            min_value=0.5, max_value=500.0, value=25.0, step=0.5, format="%.2f",
            key="aoi_custom",
        )

    st.caption("**Map Display Radius** — broader radius shown on the interactive map (Panel A) for regional context. Does **not** affect the analysis.")
    map_radius = st.number_input(
        "Map Display Radius (km)",
        min_value=1.0, max_value=600.0, value=float(DEFAULT_MAP_KM), step=5.0, format="%.2f",
        help="Stations and earthquakes within this radius appear on Panel A for regional context.",
    )

    st.markdown("---")

    # --- Stations ---
    st.subheader("📡 Seismic Stations")

    st.caption("**Station Analysis Radius** — seismic stations within this radius are included in the time analysis (Panel B) and station count (Panel D).")
    _sta_preset = st.radio(
        "Quick select:", ["Same as AOI Radius", "Custom..."],
        horizontal=True, index=0, key="sta_preset",
        help="'Same as AOI Radius' links the station search zone to your AOI radius automatically.",
    )
    if _sta_preset == "Same as AOI Radius":
        sta_anly_radius = aoi_radius
        st.caption(f"ℹ️ Station analysis radius set to **{sta_anly_radius:.2f} km** (matching AOI).")
    else:
        sta_anly_radius = st.number_input(
            "Custom Station Analysis Radius (km)",
            min_value=0.5, max_value=500.0, value=float(aoi_radius), step=0.5, format="%.2f",
            key="sta_custom",
        )

    st.caption("**Station Map Radius** — stations displayed on the map (Panel A) for visual reference only. Does not affect analysis.")
    _sta_map_preset = st.radio(
        "Select Station Map Radius:", ["Same as Map Display Radius", "Custom..."],
        horizontal=True, index=0, key="sta_map_preset",
        help="'Same as Map Display Radius' links the station display radius to your Map Display Radius automatically.",
    )
    if _sta_map_preset == "Same as Map Display Radius":
        sta_map_radius = map_radius
        st.caption(f"ℹ️ Station map radius set to **{sta_map_radius:.2f} km** (matching Map Display Radius).")
    else:
        sta_map_radius = st.number_input(
            "Custom Station Map Radius (km)",
            min_value=1.0, max_value=600.0, value=float(DEFAULT_STA_MAP_KM), step=5.0, format="%.2f",
            help="All stations within this radius will appear on the interactive map as triangles.",
        )

    st.markdown("---")

    # --- Time window ---
    st.subheader("📅 Time Window")
    st.caption(
        "Seismic events are sourced from the **TexNet Earthquake Catalog** "
        "([catalog.texnet.beg.utexas.edu](https://catalog.texnet.beg.utexas.edu/)). "
        "You can use the full catalog history or filter by a custom date range."
    )
    time_mode = st.radio("Time range", ["All TexNet history", "Custom date range"],
                         index=0, horizontal=True)
    start_date = None
    end_date   = None
    if time_mode == "Custom date range":
        col_s, col_e = st.columns(2)
        with col_s:
            sd = st.date_input("Start date", value=date(2017, 1, 1))
            start_date = sd.strftime("%Y-%m-%d")
        with col_e:
            ed = st.date_input("End date", value=date.today())
            end_date = ed.strftime("%Y-%m-%d")

    st.markdown("---")

    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# ===============================================================
# ANALYSIS EXECUTION
# ===============================================================
if run_btn:
    st.session_state.analysis_done   = False
    st.session_state.results         = None
    st.session_state.selected_event_id = None

    with st.status("Running analysis… please wait.", expanded=True) as status_bar:

        # 1) Fetch AOI events
        st.write("📡 Downloading earthquakes (AOI radius)…")
        events_raw = fetch_texnet_events_arcgis(
            TEXNET_CATALOG_MAPSERVER, TEXNET_EQ_LAYER_ID,
            aoi_lat, aoi_lon, aoi_radius,
            start_date=start_date, end_date=end_date,
        )
        if events_raw.empty:
            status_bar.update(label="❌ No events found.", state="error")
            st.error(
                f"No events returned from TexNet for the selected AOI and time window. "
                f"Try increasing the AOI radius or adjusting the date range."
            )
            st.stop()

        events = process_events_raw(events_raw)
        events = filter_final_events(events)
        if events.empty:
            status_bar.update(label="❌ No 'final' events.", state="error")
            st.error("No 'final' events found in the AOI. Try adjusting the parameters.")
            st.stop()
        st.write(f"  ✅ {len(events)} events loaded (AOI)")

        # 2) Fetch map events (broader radius)
        st.write("📡 Downloading earthquakes (map radius)…")
        events_map_raw = fetch_texnet_events_arcgis(
            TEXNET_CATALOG_MAPSERVER, TEXNET_EQ_LAYER_ID,
            aoi_lat, aoi_lon, map_radius,
            start_date=start_date, end_date=end_date,
        )
        events_map = process_events_raw(events_map_raw)
        if "EvaluationStatus" in events_map_raw.columns:
            events_map = filter_final_events(events_map)
        st.write(f"  ✅ {len(events_map)} events loaded (map radius)")

        # 3) Load stations CSV
        st.write("📂 Loading station data…")
        if not os.path.isfile(STATIONS_CSV):
            st.warning(f"Stations CSV not found at {STATIONS_CSV}. Continuing without stations.")
            stations_df = pd.DataFrame()
        else:
            stations_df = pd.read_csv(STATIONS_CSV)

        # 4) Process stations
        centroid_lat = events["Latitude (WGS84)"].mean()
        centroid_lon = events["Longitude (WGS84)"].mean()
        event_start  = events["DateTime"].min()
        event_end    = events["DateTime"].max()

        if not stations_df.empty:
            nearby_stations, stations_map_df = process_stations(
                stations_df, aoi_lat, aoi_lon,
                sta_anly_radius, sta_map_radius,
                event_start, event_end,
            )
        else:
            empty_cols = ["Latitude (WGS84)", "Longitude (WGS84)", "Station Code",
                          "Station Type", "Start Date", "End Date", "Start Clip", "End Clip", "dist_km"]
            nearby_stations = pd.DataFrame(columns=empty_cols)
            stations_map_df = pd.DataFrame(columns=empty_cols)
        st.write(f"  ✅ {len(nearby_stations)} analysis stations | {len(stations_map_df)} map stations")

        # 5) Fetch SRA lines
        st.write("🗺️ Downloading map layers (SRA, SIR, Basins)…")
        sra_gj, sir_gj_list, basins_rings = None, [], []
        try:
            sra_gj = fetch_arcgis_geojson(SRA_MAPSERVER, SRA_LAYER_ID)
        except Exception as e:
            st.warning(f"SRA layer unavailable: {e}")
        for sir_id, sir_label in SIR_LAYERS:
            try:
                sir_gj_list.append((sir_id, sir_label, fetch_arcgis_geojson(SIR_MAPSERVER, sir_id)))
            except Exception:
                pass
        try:
            basins_rings = fetch_arcgis_geojson_polygons_near_point(
                BASINS_MAPSERVER, BASINS_LAYER_ID,
                center_lat=aoi_lat, center_lon=aoi_lon,
                radius_km=max(150, map_radius), where="1=1", out_sr=4326,
            )
        except Exception as e:
            st.warning(f"Basins layer unavailable: {e}")
        st.write("  ✅ Map layers loaded")

        # 6) Generate static figure
        st.write("🖼️ Generating static figure (PNG / PDF)…")
        fig_params = {
            "AOI_LAT": aoi_lat, "AOI_LON": aoi_lon,
            "AOI_RADIUS_KM": aoi_radius, "MAP_RADIUS_KM": map_radius,
            "STATIONS_ANALYSIS_RADIUS_KM": sta_anly_radius,
            "STATIONS_MAP_RADIUS_KM": sta_map_radius,
            "STATIONS_CENTER_MODE": "AOI",
            "AREA_NAME": area_name,
            "centroid_lat": centroid_lat, "centroid_lon": centroid_lon,
        }
        static_fig = generate_static_figure(
            events, events_map, nearby_stations, stations_map_df, fig_params
        )

        # Save to disk
        run_date    = datetime.now().strftime("%Y-%m-%d")
        safe_name   = area_name.replace(" ", "_")
        fname_base  = f"{safe_name}_{run_date}_seismicity_v1"
        out_png     = os.path.join(FIGURES_DIR, f"{fname_base}.png")
        out_pdf     = os.path.join(FIGURES_DIR, f"{fname_base}.pdf")

        import matplotlib
        matplotlib.use("Agg")
        static_fig.savefig(out_png, dpi=300, facecolor="#0d1117",
                           bbox_inches="tight", pad_inches=0.3)
        static_fig.savefig(out_pdf, facecolor="#0d1117",
                           bbox_inches="tight", pad_inches=0.3)

        # Bytes for download buttons
        buf_png = io.BytesIO()
        static_fig.savefig(buf_png, format="png", dpi=300, facecolor="#0d1117",
                           bbox_inches="tight", pad_inches=0.3)
        buf_png.seek(0)
        buf_pdf = io.BytesIO()
        static_fig.savefig(buf_pdf, format="pdf", facecolor="#0d1117",
                           bbox_inches="tight", pad_inches=0.3)
        buf_pdf.seek(0)
        import matplotlib.pyplot as plt
        plt.close(static_fig)

        # Store everything in session_state
        st.session_state.results = {
            "events": events,
            "events_map": events_map,
            "nearby_stations": nearby_stations,
            "stations_map_df": stations_map_df,
            "params": fig_params,
            "buf_png": buf_png,
            "buf_pdf": buf_pdf,
            "fname_base": fname_base,
            "sra_gj": sra_gj,
            "sir_gj_list": sir_gj_list,
            "basins_rings": basins_rings,
            "area_name": area_name,
            "aoi_lat": aoi_lat,
            "aoi_lon": aoi_lon,
            "aoi_radius": aoi_radius,
        }
        st.session_state.analysis_done = True
        status_bar.update(label="✅ Analysis complete!", state="complete", expanded=False)


# ===============================================================
# RESULTS — displayed only after a successful run
# ===============================================================
if st.session_state.analysis_done and st.session_state.results is not None:
    R = st.session_state.results

    events          = R["events"]
    events_map      = R["events_map"]
    nearby_stations = R["nearby_stations"]
    stations_map_df = R["stations_map_df"]
    params          = R["params"]
    area_name       = R["area_name"]
    aoi_lat         = R["aoi_lat"]
    aoi_lon         = R["aoi_lon"]
    aoi_radius      = R["aoi_radius"]
    sra_gj          = R["sra_gj"]
    sir_gj_list     = R["sir_gj_list"]
    basins_rings    = R["basins_rings"]

    date_min = events["DateTime"].min().strftime("%Y-%m-%d")
    date_max = events["DateTime"].max().strftime("%Y-%m-%d")
    max_mag  = events["Magnitude"].max()

    # ---- Key stats row ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Events (AOI)", len(events))
    c2.metric("Max Magnitude", f"{max_mag:.1f}")
    c3.metric("Date Range", f"{date_min[:4]} – {date_max[:4]}")
    c4.metric("Analysis Stations", len(nearby_stations))

    st.markdown("---")

    # ---- Download buttons ----
    st.subheader("⬇️ Download Static Figure")
    dl1, dl2, dl3 = st.columns([1, 1, 3])
    with dl1:
        st.download_button(
            label="📥 Download PNG",
            data=R["buf_png"],
            file_name=f"{R['fname_base']}.png",
            mime="image/png",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            label="📥 Download PDF",
            data=R["buf_pdf"],
            file_name=f"{R['fname_base']}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.markdown("---")

    # ===========================================================
    # PANEL A — INTERACTIVE FOLIUM MAP
    # ===========================================================
    st.subheader("🗺️ Panel A — Interactive Seismic Map")
    st.caption(
        f"Click any earthquake to see its details. Use the ruler 📏 tool (top-left) to measure "
        f"distances in km and miles. Toggle map layers with the control (top-right). "
        f"| *{APP_CREDIT}*"
    )

    events_for_map = events_map if len(events_map) > 0 else events

    # Build Folium map centred on AOI — min_zoom prevents zooming out past display area
    def _radius_km_to_min_zoom(r_km):
        if r_km <= 10:   return 12
        elif r_km <= 25: return 11
        elif r_km <= 60: return 9
        elif r_km <= 120: return 8
        elif r_km <= 250: return 7
        else:            return 6

    m = folium.Map(
        location=[aoi_lat, aoi_lon],
        zoom_start=10,
        min_zoom=_radius_km_to_min_zoom(map_radius),
        tiles=None,
        control_scale=True,
    )

    # Base tiles
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite (Esri)",
        show=True,
    ).add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark (CartoDB)", show=False).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=False).add_to(m)

    # Measure Control (km and miles)
    MeasureControl(
        position="bottomright",
        primary_length_unit="kilometers",
        secondary_length_unit="miles",
        primary_area_unit="sqkilometers",
        secondary_area_unit="sqmiles",
        active_color="#c7d2fe",
        completed_color="#7daa65",
    ).add_to(m)

    # AOI circle
    folium.Circle(
        location=[aoi_lat, aoi_lon],
        radius=aoi_radius * 1000,
        color="#ff3b3b",
        weight=2.5,
        fill=False,
        tooltip=f"Area of Interest: {aoi_radius} km radius",
        popup=f"AOI Center: {aoi_lat:.4f}°N, {aoi_lon:.4f}°W | Radius: {aoi_radius} km",
    ).add_to(m)
    folium.CircleMarker(
        location=[aoi_lat, aoi_lon],
        radius=4, color="#ff3b3b", fill=True, fill_color="#ff3b3b",
        tooltip="AOI Center",
    ).add_to(m)

    # SRA lines layer
    if sra_gj:
        sra_layer = folium.FeatureGroup(name="SRA (RRC)", show=True)
        for feat in sra_gj.get("features", []):
            geom  = feat.get("geometry", {})
            gtype = geom.get("type", "")
            lines = [geom["coordinates"]] if gtype == "LineString" else (
                geom["coordinates"] if gtype == "MultiLineString" else []
            )
            for line in lines:
                # Coords from ArcGIS in 3857; convert to lat/lon for Folium
                try:
                    latlons = []
                    for pt in line:
                        _x, _y = pt[0], pt[1]
                        _lon = _x * 180.0 / 20037508.34
                        _lat = np.degrees(2 * np.arctan(np.exp(_y * np.pi / 20037508.34)) - np.pi / 2)
                        latlons.append([_lat, _lon])
                    if latlons:
                        folium.PolyLine(latlons, color="#00e5ff", weight=1.5,
                                        opacity=0.85, tooltip="SRA (RRC)").add_to(sra_layer)
                except Exception:
                    pass
        sra_layer.add_to(m)

    # SIR lines layer
    sir_layer = folium.FeatureGroup(name="SIR Depth Contours", show=True)
    for sir_id, sir_label, sir_gj in sir_gj_list:
        for feat in sir_gj.get("features", []):
            geom  = feat.get("geometry", {})
            gtype = geom.get("type", "")
            lines = [geom["coordinates"]] if gtype == "LineString" else (
                geom["coordinates"] if gtype == "MultiLineString" else []
            )
            for line in lines:
                try:
                    latlons = []
                    for pt in line:
                        _lon = pt[0] * 180.0 / 20037508.34
                        _lat = np.degrees(2 * np.arctan(np.exp(pt[1] * np.pi / 20037508.34)) - np.pi / 2)
                        latlons.append([_lat, _lon])
                    if latlons:
                        folium.PolyLine(latlons, color="#d1d5db", weight=1.0,
                                        opacity=0.7, tooltip=f"SIR {sir_label}").add_to(sir_layer)
                except Exception:
                    pass
    sir_layer.add_to(m)

    # Basins & Plays layer
    if basins_rings:
        basins_layer = folium.FeatureGroup(name="Basins & Plays (BEG)", show=True)
        for ring in basins_rings:
            try:
                latlons = [[pt[1], pt[0]] for pt in ring]
                folium.PolyLine(latlons, color="#32cd32", weight=1.5,
                                opacity=0.6, tooltip="Texas Basins & Plays").add_to(basins_layer)
            except Exception:
                pass
        basins_layer.add_to(m)

    # Earthquake markers (broader map radius)
    eq_layer = folium.FeatureGroup(name="Earthquakes", show=True)
    events_display = events_for_map.copy()

    for _, row in events_display.sort_values("Magnitude").iterrows():
        mag     = row["Magnitude"]
        lat_eq  = row["Latitude (WGS84)"]
        lon_eq  = row["Longitude (WGS84)"]
        color   = MAG_COLORS.get(row.get("MagClass", assign_mag_class(mag)), "#7daa65")
        radius  = max(3, float(abs(float(mag)) ** 1.5) * 0.8)
        depth   = row.get("Depth_MSL_km", float("nan"))
        ev_date = row.get("Origin Date", "N/A")
        ev_time = row.get("Origin Time", "N/A")
        ev_id   = str(row.get("EventId", "N/A"))
        status  = str(row.get("Evaluation Status", "N/A"))
        region  = str(row.get("RegionName", "N/A"))
        county  = str(row.get("CountyName", "N/A"))

        depth_str = f"{depth:.2f} km" if not (pd.isna(depth) if not isinstance(depth, str) else False) else "N/A"

        popup_html = f"""
        <div style="font-family:sans-serif; min-width:200px;">
          <b style="font-size:1.05rem; color:#1a1a2e;">M {mag:.1f} Earthquake</b><br>
          <hr style="margin:4px 0;">
          <b>Date:</b> {ev_date} {ev_time}<br>
          <b>Latitude:</b> {lat_eq:.4f}°N<br>
          <b>Longitude:</b> {lon_eq:.4f}°W<br>
          <b>Magnitude:</b> {mag:.2f}<br>
          <b>Depth (MSL):</b> {depth_str}<br>
          <b>Status:</b> {status}<br>
          <b>Region:</b> {region}<br>
          <b>County:</b> {county}<br>
          <b>Event ID:</b> {ev_id}
        </div>
        """

        # Determine if this is a "selected" event (highlighted from Panel B)
        is_selected = (ev_id == str(st.session_state.selected_event_id))
        edge_color  = "#ffffff" if not is_selected else "#ffff00"
        edge_width  = 0.4       if not is_selected else 3.0
        fill_opacity = 0.65     if not is_selected else 0.95

        # CircleMarker — pixel-based radius, stays constant on zoom (never shrinks or grows)
        if mag < 2.0:   px_r = 4
        elif mag < 3.0: px_r = 6
        elif mag < 4.0: px_r = 8
        elif mag < 5.0: px_r = 12
        else:           px_r = 17
        if is_selected: px_r = int(px_r * 1.6)

        cm = folium.CircleMarker(
            location=[lat_eq, lon_eq],
            radius=px_r,
            color=edge_color,
            weight=edge_width,
            fill=True,
            fill_color=color,
            fill_opacity=fill_opacity,
            tooltip=f"M {mag:.1f} | {ev_date} | Depth: {depth_str}",
            popup=folium.Popup(popup_html, max_width=280),
        )
        cm.add_to(eq_layer)

        # Magnitude label — only for M >= 3.0, bold+outlined stroke for M >= 4.0
        if mag >= 3.0:
            bold    = "bold"   if mag >= 4.0 else "normal"
            f_size  = "9px"    if mag >= 4.0 else "8px"
            txt_col = "#000000" if mag >= 4.0 else "#4a4a4a"
            shadow  = "-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff" if mag >= 4.0 else "none"
            folium.Marker(
                location=[lat_eq, lon_eq],
                icon=folium.DivIcon(
                    html=(
                        f'<div style="'
                        f'width:30px; height:30px;'
                        f'display:flex; align-items:center; justify-content:center;'
                        f'font-size:{f_size};'
                        f'font-weight:{bold};'
                        f'color:{txt_col};'
                        f'text-shadow:{shadow};'
                        f'pointer-events:none;'
                        f'">{mag:.1f}</div>'
                    ),
                    icon_size=(30, 30),
                    icon_anchor=(15, 15),
                ),
            ).add_to(eq_layer)

    eq_layer.add_to(m)

    # Station markers
    if len(stations_map_df) > 0:
        sta_layer = folium.FeatureGroup(name="Seismic Stations", show=True)
        for _, row in stations_map_df.iterrows():
            stype   = str(row.get("Station Type", "Non-TexNet"))
            _, fc, _, _ = STATION_MARKERS.get(stype, ("^", "#cccccc", "#ffffff", stype))
            code    = str(row.get("Station Code", ""))
            st_lat  = row["Latitude (WGS84)"]
            st_lon  = row["Longitude (WGS84)"]
            st_start = str(row.get("Start Date", ""))
            st_end   = str(row.get("End Date",   "ongoing"))

            popup_html = f"""
            <div style="font-family:sans-serif;">
              <b>{code}</b><br>
              <b>Type:</b> {stype}<br>
              <b>Start:</b> {st_start}<br>
              <b>End:</b> {st_end}<br>
              <b>Lat/Lon:</b> {st_lat:.4f}°N, {st_lon:.4f}°W
            </div>
            """
            folium.Marker(
                location=[st_lat, st_lon],
                icon=folium.DivIcon(
                    html=(
                        f'<div style="text-align:center; pointer-events:none;">'
                        f'<svg width="18" height="18" viewBox="0 0 18 18" style="display:block; margin:auto;">'
                        f'<polygon points="9,2 16,16 2,16" fill="{fc}" stroke="white" stroke-width="1.5" />'
                        f'</svg>'
                        f'<div style="'
                        f'font-size:9px; font-family:sans-serif;'
                        f'color:#e6edf3; font-weight:bold;'
                        f'text-shadow:0 0 2px #000, 0 0 2px #000;'
                        f'white-space:nowrap; margin-top:1px;'
                        f'">{code}</div>'
                        f'</div>'
                    ),
                    icon_size=(50, 28),
                    icon_anchor=(25, 13),
                ),
                tooltip=f"{code} ({stype})",
                popup=folium.Popup(popup_html, max_width=220),
            ).add_to(sta_layer)
        sta_layer.add_to(m)

    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    # Add custom HTML legend to match the static PNG
    from branca.element import Template, MacroElement
    legend_html = """
    {% macro html(this, kwargs) %}
    <div style="position: absolute; 
                bottom: 70px; left: 10px; width: 170px; height: auto; 
                background-color: rgba(26, 26, 46, 0.85);
                border: 1px solid #3a3a5a; border-radius: 6px;
                z-index:9999; font-family: sans-serif;
                font-size: 11px; color: #d0d0e0; padding: 10px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.5);">
        <div style="margin-bottom:8px; font-weight:bold; color:#c7d2fe; text-align:center;">Magnitude Class</div>
        <div style="margin-bottom:5px; display:flex; align-items:center;">
            <div style="background:#7daa65; width:10px; height:10px; margin-right:8px; border-radius:50%; border:1px solid white;"></div> &lt;1.9
        </div>
        <div style="margin-bottom:5px; display:flex; align-items:center;">
            <div style="background:#b7ed4d; width:11px; height:11px; margin-right:8px; border-radius:50%; border:1px solid white;"></div> 2.0–2.9
        </div>
        <div style="margin-bottom:5px; display:flex; align-items:center;">
            <div style="background:#ffc34b; width:12px; height:12px; margin-right:8px; border-radius:50%; border:1px solid white;"></div> 3.0–3.4
        </div>
        <div style="margin-bottom:5px; display:flex; align-items:center;">
            <div style="background:#ff7f7e; width:13px; height:13px; margin-right:8px; border-radius:50%; border:1px solid white;"></div> 3.5–3.9
        </div>
        <div style="margin-bottom:5px; display:flex; align-items:center;">
            <div style="background:#d62728; width:14px; height:14px; margin-right:8px; border-radius:50%; border:1px solid white;"></div> 4.0–4.9
        </div>
        <div style="margin-bottom:12px; display:flex; align-items:center;">
            <div style="background:#bf00ff; width:15px; height:15px; margin-right:8px; border-radius:50%; border:1px solid white;"></div> ≥5.0
        </div>
        <div style="margin-bottom:8px; font-weight:bold; color:#c7d2fe; text-align:center; border-top:1px solid #3a3a5a; padding-top:8px;">Station Type</div>
        <div style="margin-bottom:5px; display:flex; align-items:center;">
            <svg width="14" height="14" style="margin-right:6px; margin-left:1px;"><polygon points="7,1 13,13 1,13" fill="#000000" stroke="white" stroke-width="1.2"/></svg> TexNet Permanent
        </div>
        <div style="margin-bottom:5px; display:flex; align-items:center;">
            <svg width="14" height="14" style="margin-right:6px; margin-left:1px;"><polygon points="7,1 13,13 1,13" fill="#00bfff" stroke="white" stroke-width="1.2"/></svg> TexNet Portable
        </div>
        <div style="margin-bottom:5px; display:flex; align-items:center;">
            <svg width="14" height="14" style="margin-right:6px; margin-left:1px;"><polygon points="7,1 13,13 1,13" fill="#cccccc" stroke="white" stroke-width="1.2"/></svg> Non-TexNet
        </div>
    </div>
    {% endmacro %}
    """
    macro = MacroElement()
    macro._template = Template(legend_html)
    m.get_root().add_child(macro)

    # Render the Folium map — capture click events
    map_out = st_folium(m, width="100%", height=550, returned_objects=["last_object_clicked_popup"])

    # ---- Capture earthquake click from map ----
    clicked_info = None
    if map_out and map_out.get("last_object_clicked_popup"):
        popup_raw = map_out["last_object_clicked_popup"]
        if popup_raw and "Event ID" in popup_raw:
            import re
            match = re.search(r"Event ID.*?</b>\s*([\w.-]+)", popup_raw)
            if match:
                clicked_ev_id = match.group(1).strip()
                if clicked_ev_id != st.session_state.selected_event_id:
                    st.session_state.selected_event_id = clicked_ev_id

    # Show selected event details
    sel_id = st.session_state.selected_event_id
    if sel_id and str(sel_id) != "None":
        match_ev = events[events["EventId"].astype(str) == str(sel_id)]
        if len(match_ev):
            row = match_ev.iloc[0]
            st.info(
                f"**Selected earthquake** — M {row['Magnitude']:.1f} | "
                f"{row.get('Origin Date','N/A')} {row.get('Origin Time','')} | "
                f"Depth: {row.get('Depth_MSL_km', float('nan')):.1f} km | "
                f"Lat: {row['Latitude (WGS84)']:.4f}°N | Lon: {row['Longitude (WGS84)']:.4f}°W | "
                f"EventId: {sel_id}"
            )

    st.markdown("---")

    # ===========================================================
    # PANEL B — PLOTLY TIME vs DEPTH
    # ===========================================================
    st.subheader("📈 Panel B — Time vs Depth")
    st.caption(
        "Hover over any point to see earthquake details. "
        "The selected earthquake (clicked in Panel A) is highlighted in yellow. "
        f"| *{APP_CREDIT}*"
    )

    fig_b = go.Figure()

    selected_row = None
    if sel_id and str(sel_id) != "None":
        match_ev = events[events["EventId"].astype(str) == str(sel_id)]
        if len(match_ev):
            selected_row = match_ev.iloc[0]

    for label, _ in MAG_BINS:
        subset = events[events["MagClass"] == label].dropna(subset=["Depth_MSL_km"])
        if subset.empty:
            continue
        mags_b = np.abs(subset["Magnitude"].values)
        
        # Calculate sizes using fixed rules based on magnitude ranges
        sizes_b = np.zeros_like(mags_b)
        sizes_b[mags_b <= 1.9] = 3.0
        sizes_b[(mags_b >= 2.0) & (mags_b < 3.0)] = 4.0
        sizes_b[(mags_b >= 3.0) & (mags_b < 3.5)] = 15.0
        sizes_b[(mags_b >= 3.5) & (mags_b < 4.0)] = 20.0
        sizes_b[(mags_b >= 4.0) & (mags_b < 5.0)] = 25.0
        sizes_b[mags_b >= 5.0] = 35.0
        color   = MAG_COLORS[label]

        # Is each point the selected one?
        is_sel_arr = (subset["EventId"].astype(str) == str(sel_id)).values if sel_id else np.zeros(len(subset), dtype=bool)

        # Normal points
        normal = subset[~is_sel_arr]
        sz_n   = sizes_b[~is_sel_arr]
        if len(normal):
            fig_b.add_trace(go.Scatter(
                x=normal["DateTime"],
                y=normal["Depth_MSL_km"],
                mode="markers+text",
                name=label,
                legendgroup=label,
                marker=dict(color=color, size=sz_n, opacity=0.7,
                            line=dict(color="white", width=0.4)),
                text=normal["Magnitude"].apply(lambda m: f'<span style="text-shadow: -1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff;">{m:.1f}</span>' if m >= 3.0 else ""),
                textposition="middle center",
                textfont=dict(color="black", size=9),
                hovertext=normal.apply(lambda r: (
                    f"<b>M {r['Magnitude']:.2f}</b><br>"
                    f"Date: {r.get('Origin Date','N/A')} {r.get('Origin Time','')}<br>"
                    f"Depth: {r.get('Depth_MSL_km', float('nan')):.2f} km<br>"
                    f"Lat: {r['Latitude (WGS84)']:.4f}°N<br>"
                    f"Lon: {r['Longitude (WGS84)']:.4f}°W<br>"
                    f"Status: {r.get('Evaluation Status','N/A')}<br>"
                    f"EventId: {r.get('EventId','N/A')}"
                ), axis=1),
                hovertemplate="%{hovertext}<extra></extra>",
                showlegend=True,
            ))

        # Highlighted (selected) point
        sel_pts = subset[is_sel_arr]
        sz_s    = sizes_b[is_sel_arr]
        if len(sel_pts):
            fig_b.add_trace(go.Scatter(
                x=sel_pts["DateTime"],
                y=sel_pts["Depth_MSL_km"],
                mode="markers+text",
                name=f"Selected ({label})",
                legendgroup=label,
                marker=dict(color="#ffff00", size=sz_s * 2.0, opacity=1.0,
                            line=dict(color="white", width=2.5),
                            symbol="circle"),
                text=sel_pts["Magnitude"].apply(lambda m: f'<span style="text-shadow: -1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff;">{m:.1f}</span>' if m >= 3.0 else ""),
                textposition="middle center",
                textfont=dict(color="black", size=9),
                hovertext=sel_pts.apply(lambda r: (
                    f"<b>⭐ SELECTED: M {r['Magnitude']:.2f}</b><br>"
                    f"Date: {r.get('Origin Date','N/A')}<br>"
                    f"Depth: {r.get('Depth_MSL_km', float('nan')):.2f} km<br>"
                    f"EventId: {r.get('EventId','N/A')}"
                ), axis=1),
                hovertemplate="%{hovertext}<extra></extra>",
                showlegend=True,
            ))

    # Annual vertical lines
    yr_min = int(events["Year"].min())
    yr_max = int(events["Year"].max()) + 1
    for yr in range(yr_min, yr_max + 1):
        fig_b.add_vline(
            x=pd.Timestamp(f"{yr}-01-01").timestamp() * 1000,
            line=dict(color="#8b949e", width=0.8, dash="dash"),
            opacity=0.35,
        )
        fig_b.add_annotation(
            x=pd.Timestamp(f"{yr}-01-01"), y=1.02, yref="paper",
            text=str(yr), showarrow=False, font=dict(size=9, color="#8b949e"),
            xanchor="left",
        )

    # Station operation segments
    if len(nearby_stations) > 0:
        depth_min_ev = events["Depth_MSL_km"].min()
        seg_y_start  = depth_min_ev - 1.0
        seg_spacing  = 0.8  # Increased from 0.4 to space lines out
        ns_sorted    = nearby_stations.sort_values("Start Date").reset_index(drop=True)
        for i, (_, row) in enumerate(ns_sorted.iterrows()):
            stype = row.get("Station Type", "Non-TexNet")
            color = STATION_MARKERS.get(stype, ("^", "#cccccc", "#ffffff", ""))[1]
            start = row.get("Start Clip")
            end   = row.get("End Clip")
            if pd.isna(start):
                continue
            y_seg = seg_y_start - i * seg_spacing
            code  = str(row.get("Station Code", ""))
            fig_b.add_trace(go.Scatter(
                x=[start, end], y=[y_seg, y_seg],
                mode="lines+text",
                line=dict(color=color, width=2.5),
                opacity=0.8,
                text=["", f" <b>{code}</b>"],
                textposition="middle right",
                textfont=dict(color=color, size=11),
                name=f"{code} ({stype})",
                hovertemplate=f"Station: {code} ({stype})<br>"
                              f"Active: {start} – {end}<extra></extra>",
                showlegend=False,
            ))

    fig_b.update_yaxes(
        autorange="reversed",
        title="Depth (km, rel. to MSL)",
        gridcolor="#21262d",
        showgrid=True,
    )
    fig_b.update_xaxes(
        title="Date",
        gridcolor="#21262d",
        showgrid=True,
    )
    fig_b.update_layout(
        title=dict(
            text=(
                f"B — Time vs Depth  |  {area_name}  |  "
                f"{len(events)} events · {date_min} to {date_max}<br>"
                f"<span style='font-size:11px;color:#8b949e;'>{APP_CREDIT}</span>"
            ),
            font=dict(size=14, color="#e6edf3"),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.25,
            xanchor="left", x=0,
            font=dict(color="#c9d1d9"), bgcolor="rgba(22,27,34,0.8)",
        ),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9"),
        margin=dict(t=80, b=80, l=60, r=60),
        height=650,
        hovermode="closest",
    )

    st.plotly_chart(fig_b, use_container_width=True)

    st.markdown("---")

    # ---- Summary table ----
    st.subheader("📋 Event List")
    display_cols = ["Origin Date", "Origin Time", "Magnitude", "Depth_MSL_km",
                    "Latitude (WGS84)", "Longitude (WGS84)", "Evaluation Status",
                    "RegionName", "EventId"]
    display_cols = [c for c in display_cols if c in events.columns]
    st.dataframe(
        events[display_cols].rename(columns={
            "Depth_MSL_km": "Depth (km)", "Latitude (WGS84)": "Latitude",
            "Longitude (WGS84)": "Longitude", "RegionName": "Region",
        }).sort_values("Origin Date", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=350,
    )

    st.markdown("---")

    # ===========================================================
    # PANEL C — YEARLY DEPTH HISTOGRAMS
    # ===========================================================
    st.subheader("📊 Panel C — Yearly Depth Histograms")
    st.caption(
        "Horizontal depth distribution per year. Red lines show the mean (μ) "
        "and ±1σ / ±2σ bounds. "
        f"| *{APP_CREDIT}*"
    )

    from plotly.subplots import make_subplots

    events_c = events.dropna(subset=["Depth_MSL_km"]).copy()
    years_c  = sorted(events_c["Year"].unique())
    n_years  = len(years_c)

    if n_years > 0:
        d_min = float(events_c["Depth_MSL_km"].min())
        d_max = float(events_c["Depth_MSL_km"].max())
        if d_min == d_max:
            d_min -= 1.0
            d_max  += 1.0
        bins = np.linspace(d_min, d_max, 20)

        # Synchronize Y axes and define ticks
        y_ticks = [-6, -3, 0, 3, 6, 8, 10, 12, 14, 16]
        max_count = 0

        # Pre-calculate max count for uniform x-axis
        for year in years_c:
            yd = events_c.loc[events_c["Year"] == year, "Depth_MSL_km"]
            if len(yd) > 0:
                counts, _ = np.histogram(yd, bins=bins)
                max_count = max(max_count, counts.max())

        n_cols = 1
        n_rows = n_years

        subplot_titles = []
        for y in years_c:
            n_y     = int((events_c["Year"] == y).sum())
            n_nan_y = int((events["Year"] == y).sum()) - n_y
            tag = f" (+{n_nan_y} no depth)" if n_nan_y > 0 else ""
            subplot_titles.append(f"{y}  (n={n_y}{tag})")

        fig_c = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.12,
            vertical_spacing=max(0.02, 0.4 / n_rows) if n_rows > 1 else 0.1,
            shared_xaxes=True,
        )

        for idx, year in enumerate(years_c):
            r = idx + 1
            c = 1
            yd = events_c.loc[events_c["Year"] == year, "Depth_MSL_km"]
            if len(yd) == 0:
                continue
            counts, bin_edges = np.histogram(yd, bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            mu    = float(yd.mean())
            sigma = float(yd.std()) if len(yd) > 1 else 0.0

            fig_c.add_trace(
                go.Bar(
                    y=bin_centers, x=counts,
                    orientation="h",
                    marker=dict(color="#6366f1", opacity=0.65,
                                line=dict(color="#818cf8", width=0.5)),
                    name=str(year), showlegend=False,
                    hovertemplate="Depth: %{y:.1f} km<br>Count: %{x}<extra>" + str(year) + "</extra>",
                ),
                row=r, col=c,
            )

            for val, dash, lw, label_text in [
                (mu + 2 * sigma, "dot",  0.7, f"+2σ: {mu + 2 * sigma:.1f}"),
                (mu + sigma,     "dash", 1.0, f"+1σ: {mu + sigma:.1f}"),
                (mu,             "solid",1.6, f"μ: {mu:.1f}"),
                (mu - sigma,     "dash", 1.0, f"-1σ: {mu - sigma:.1f}"),
                (mu - 2 * sigma, "dot",  0.7, f"-2σ: {mu - 2 * sigma:.1f}"),
            ]:
                if d_min <= val <= d_max:
                    fig_c.add_shape(
                        type="line",
                        y0=val, y1=val, x0=0, x1=1,
                        xref="x domain", yref="y",
                        line=dict(color="#ff6b6b", dash=dash, width=lw),
                        row=r, col=c,
                    )
                    fig_c.add_annotation(
                        x=1.02, y=val,
                        xref="x domain", yref="y",
                        text=label_text,
                        showarrow=False,
                        xanchor="left",
                        yanchor="middle",
                        font=dict(color="#ff6b6b", size=11.0),
                        row=r, col=c,
                    )

            fig_c.update_yaxes(
                autorange="reversed", row=r, col=c,
                gridcolor="#21262d", showgrid=True,
                tickmode="array", tickvals=y_ticks,
                title_text="Depth (km)" if c == 1 else "",
            )
            fig_c.update_xaxes(
                row=r, col=c, gridcolor="#21262d", showgrid=True,
                range=[0, float(max_count) * 1.05],
                title_text="Count" if r == n_rows else "",
            )

        # Make plot taller to accommodate single column
        plot_height = int(max(350, 220 * n_rows))
        fig_c.update_layout(
            title=dict(
                text=(
                    f"C — Yearly Depth Histograms  |  {area_name}  |  "
                    f"{date_min} to {date_max}<br>"
                    f"<span style='font-size:11px;color:#8b949e;'>{APP_CREDIT}</span>"
                ),
                font=dict(size=14, color="#e6edf3"),
            ),
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="#c9d1d9"),
            margin=dict(t=90, b=50, l=70, r=60),
            height=plot_height,
        )
        
        # Constrain width using Option 2 (Streamlit columns)
        col_c_left, col_c_center, col_c_right = st.columns([0.15, 0.7, 0.15])
        with col_c_center:
            st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.info("No depth data available to generate histograms.")

# ===============================================================
# FOOTER
# ===============================================================
st.markdown("---")
st.markdown(
    f"<div style='text-align:center; color:#8b949e; font-size:0.78rem;'>"
    f"{APP_CREDIT} &nbsp;|&nbsp; "
    f"Data: <a href='https://catalog.texnet.beg.utexas.edu/' style='color:#c7d2fe;'>TexNet Catalog</a> "
    f"(BEG / UT Austin) &nbsp;|&nbsp; SRA: Texas RRC &nbsp;|&nbsp; Basins: BEG / UT Austin"
    f"</div>",
    unsafe_allow_html=True,
)
