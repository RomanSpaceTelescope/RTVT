"""Streamlit web app for the Roman Target Visibility Tool.

Run locally::

    streamlit run streamlit_app.py

Deploys to Streamlit Community Cloud by pointing at this file in the repo.
"""

from __future__ import annotations

import html
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from astropy import units as u
from astropy.coordinates import SkyCoord

from rtvt.analysis import check_cvz_status
from rtvt.coords import coordinate_labels, normalize_coordinate_system, skycoord_from_lon_lat
from rtvt.reports import figure_img_html
from rtvt.sky_grid import get_cached_all_sky_visibility_grid
from rtvt.utils import quantity_series_to_deg
from rtvt.visibility import VisibilityCalculator
from rtvt.visualization.gantt import (
    make_gantt_chart,
    make_separation_comparison,
    target_color,
)
from rtvt.visualization.timeseries import make_visibility_plot

DURATION_DAYS = 365
SAMPLING_DAYS = 1
GRID_STEP_DEG = 10
CLICK_STEP_DEG = 3
APP_BUILD_LABEL = "streamlit-app 3-degree click grid, 2026-05-21"


# --- Streamlit page setup ---------------------------------------------------

st.set_page_config(
    page_title="RTVT — Roman Target Visibility Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Session state init -----------------------------------------------------

if "selected_targets" not in st.session_state:
    st.session_state.selected_targets = []
if "last_click_key" not in st.session_state:
    st.session_state.last_click_key = None
if "coordinate_system" not in st.session_state:
    st.session_state.coordinate_system = "equatorial"


# --- Cached compute ---------------------------------------------------------


@st.cache_data(show_spinner="Computing all-sky visibility grid…")
def _cached_sky_grid(
    coordinate_system: str,
    duration_days: float,
    sampling_days: float,
    grid_step_deg: float,
):
    return get_cached_all_sky_visibility_grid(
        grid_step_deg=grid_step_deg,
        duration_days=duration_days,
        sampling_days=sampling_days,
        coordinate_system=coordinate_system,
    )


@st.cache_data(show_spinner="Computing target visibility…")
def _cached_target_visibility(
    ra_deg: float,
    dec_deg: float,
    duration_days: float,
    sampling_days: float,
):
    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    vis = VisibilityCalculator(
        target,
        report=False,
        fileout=None,
        interval_sampling_days=sampling_days,
        interval_start_time=None,
        interval_duration_days=duration_days,
    )
    vis.compute_and_display()
    label = vis.df_results.index.levels[0][0]
    df = vis.df_results.xs(label, level=0)
    dates = pd_doy_to_datetime(df)
    good = df["good_angles"].astype(bool).values
    separation = quantity_series_to_deg(df["separation"])
    return label, dates, good, separation


def pd_doy_to_datetime(df):
    import pandas as pd

    return pd.to_datetime(df.index.astype(str), format="%Y-%j.%f")


def mollweide_project(lon_deg, lat_deg):
    """Project longitude/latitude in degrees into Mollweide x/y coordinates."""
    lon_rad = np.deg2rad(np.asarray(lon_deg, dtype=float))
    lat_rad = np.deg2rad(np.asarray(lat_deg, dtype=float))
    theta = lat_rad.copy()
    pole_mask = np.isclose(np.abs(lat_rad), np.pi / 2)

    for _ in range(10):
        numerator = 2 * theta + np.sin(2 * theta) - np.pi * np.sin(lat_rad)
        denominator = 2 + 2 * np.cos(2 * theta)
        step = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(theta, dtype=float),
            where=np.abs(denominator) > 1e-12,
        )
        theta = theta - step

    theta = np.where(pole_mask, np.sign(lat_rad) * np.pi / 2, theta)
    x = (2 * np.sqrt(2) / np.pi) * lon_rad * np.cos(theta)
    y = np.sqrt(2) * np.sin(theta)
    return x, y


def mollweide_inverse_projected(x_value, y_value):
    """Convert Mollweide x/y coordinates back to lon/lat degrees."""
    x_array = np.asarray(x_value, dtype=float)
    y_array = np.asarray(y_value, dtype=float)
    theta = np.arcsin(np.clip(y_array / np.sqrt(2), -1, 1))
    lat_rad = np.arcsin(np.clip((2 * theta + np.sin(2 * theta)) / np.pi, -1, 1))
    cos_theta = np.cos(theta)
    lon_rad = np.divide(
        x_array * np.pi,
        2 * np.sqrt(2) * cos_theta,
        out=np.zeros_like(x_array, dtype=float),
        where=np.abs(cos_theta) > 1e-12,
    )
    return np.rad2deg(lon_rad), np.rad2deg(lat_rad)


def mollweide_inverse(x_value: float, y_value: float) -> tuple[float, float]:
    """Convert one Mollweide x/y point back to lon/lat degrees."""
    lon_deg, lat_deg = mollweide_inverse_projected(x_value, y_value)
    lon_deg = float(lon_deg)
    lat_deg = float(lat_deg)
    if lon_deg < 0:
        lon_deg += 360.0
    return lon_deg % 360.0, lat_deg


def customdata_value(customdata, index: int):
    """Return a Plotly customdata value from list-like or dict-like Streamlit payloads."""
    if customdata is None:
        return None
    if isinstance(customdata, dict):
        return customdata.get(index, customdata.get(str(index)))
    try:
        return customdata[index]
    except (IndexError, KeyError, TypeError):
        return None


def selection_point_lon_lat(point) -> tuple[float, float]:
    """Extract sky coordinates from a Streamlit Plotly selection point."""
    customdata = point.get("customdata") if hasattr(point, "get") else None
    custom_lon = customdata_value(customdata, 0)
    custom_lat = customdata_value(customdata, 1)
    if custom_lon is not None and custom_lat is not None:
        return float(custom_lon), float(custom_lat)

    return mollweide_inverse(
        float(point.get("x", 0.0)),
        float(point.get("y", 0.0)),
    )


def mollweide_visibility_raster(
    lon_grid_deg: np.ndarray,
    lat_grid_deg: np.ndarray,
    vis_frac_2d: np.ndarray,
    width: int = 361,
    height: int = 181,
):
    """Resample the coarse visibility grid onto a regular Mollweide image."""
    x_values = np.linspace(-2 * np.sqrt(2), 2 * np.sqrt(2), width)
    y_values = np.linspace(-np.sqrt(2), np.sqrt(2), height)
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    inside = (x_mesh / (2 * np.sqrt(2))) ** 2 + (y_mesh / np.sqrt(2)) ** 2 <= 1

    lon_plot_deg, lat_deg = mollweide_inverse_projected(x_mesh, y_mesh)
    lon_original_deg = np.where(lon_plot_deg < 0, lon_plot_deg + 360.0, lon_plot_deg) % 360.0

    lon_distance = np.abs(
        ((lon_original_deg[..., np.newaxis] - lon_grid_deg[np.newaxis, np.newaxis, :] + 180) % 360)
        - 180
    )
    lat_distance = np.abs(lat_deg[..., np.newaxis] - lat_grid_deg[np.newaxis, np.newaxis, :])
    lon_indices = np.argmin(lon_distance, axis=2)
    lat_indices = np.argmin(lat_distance, axis=2)

    raster = vis_frac_2d[lat_indices, lon_indices]
    raster = np.where(inside, raster, np.nan)
    return x_values, y_values, raster


def add_mollweide_grid(fig: go.Figure) -> None:
    """Draw notebook-like Mollweide grid lines and oval boundary."""
    line_style = dict(color="rgba(90, 96, 110, 0.28)", width=1)

    for lat in range(-60, 61, 30):
        lon_values = np.linspace(-180, 180, 361)
        lat_values = np.full_like(lon_values, lat, dtype=float)
        x_values, y_values = mollweide_project(lon_values, lat_values)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                line=line_style,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    for lon in range(-150, 180, 30):
        lat_values = np.linspace(-89.9, 89.9, 360)
        lon_values = np.full_like(lat_values, lon, dtype=float)
        x_values, y_values = mollweide_project(lon_values, lat_values)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                line=line_style,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    theta = np.linspace(0, 2 * np.pi, 361)
    fig.add_trace(
        go.Scatter(
            x=2 * np.sqrt(2) * np.cos(theta),
            y=np.sqrt(2) * np.sin(theta),
            mode="lines",
            line=dict(color="rgba(40, 45, 56, 0.45)", width=1),
            hoverinfo="skip",
            showlegend=False,
        )
    )


# --- Sidebar ----------------------------------------------------------------

with st.sidebar:
    st.title("RTVT")
    st.caption("Roman Target Visibility Tool")
    st.caption(f"Build: {APP_BUILD_LABEL}")

    coord_system = st.radio(
        "Coordinate system",
        options=("equatorial", "galactic"),
        format_func=lambda v: "Equatorial (RA/Dec)" if v == "equatorial" else "Galactic (l/b)",
        key="coordinate_system",
    )

    st.divider()
    st.caption(f"Interval: {DURATION_DAYS} days, sampling every {SAMPLING_DAYS} day(s)")
    st.caption(f"All-sky grid step: {GRID_STEP_DEG}°")
    st.caption(f"Map click coordinate step: {CLICK_STEP_DEG}°")

    st.divider()
    if st.button("Clear all targets", type="secondary", use_container_width=True):
        st.session_state.selected_targets = []
        st.session_state.last_click_key = None
        st.rerun()

    st.caption(f"Selected targets: **{len(st.session_state.selected_targets)}**")


# --- Top-of-page title ------------------------------------------------------

lon_label, lat_label, coord_title = coordinate_labels(coord_system)

st.title("Roman Target Visibility Tool")
st.caption(
    f"Click anywhere on the sky map below — or enter coordinates manually — "
    f"to compute visibility for that target. Currently using "
    f"**{coord_title}** coordinates."
)


# --- Helpers ----------------------------------------------------------------


def _add_target(lon_deg: float, lat_deg: float):
    """Compute visibility for (lon, lat) in the active coord system and append to state."""
    coord_system = normalize_coordinate_system(st.session_state.coordinate_system)
    sky = skycoord_from_lon_lat(lon_deg, lat_deg, coord_system)
    sky_icrs = sky.icrs
    ra_deg = float(sky_icrs.ra.deg)
    dec_deg = float(sky_icrs.dec.deg)

    label, dates, good, separation = _cached_target_visibility(
        ra_deg, dec_deg, DURATION_DAYS, SAMPLING_DAYS
    )

    # CVZ check needs a DataFrame slice; reconstruct minimal one.
    import pandas as pd

    df_for_cvz = pd.DataFrame(
        {"good_angles": good, "separation": separation},
    )
    is_cvz, vis_fraction, _ = check_cvz_status(df_for_cvz)

    lon_lat_label = (
        f"{lon_label}={lon_deg:.4f}, {lat_label}={lat_deg:.4f}"
    )

    color = target_color(len(st.session_state.selected_targets))
    st.session_state.selected_targets.append(
        dict(
            label=label,
            display_label=lon_lat_label,
            coord_system=coord_system,
            coord_lon_deg=float(lon_deg),
            coord_lat_deg=float(lat_deg),
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            dates=dates,
            good=good,
            separation=separation,
            is_cvz=bool(is_cvz),
            vis_fraction=float(vis_fraction),
            color=color,
        )
    )


# --- All-sky map ------------------------------------------------------------

ra_grid_deg, dec_grid_deg, vis_frac_2d = _cached_sky_grid(
    coord_system, DURATION_DAYS, SAMPLING_DAYS, GRID_STEP_DEG
)

# Project the grid into Mollweide coordinates. The heatmap trace gives the
# complete notebook-style oval map; the transparent marker trace gives
# Streamlit selectable points for click-to-add behavior at finer precision.
click_lon_grid_deg = np.arange(0, 360, CLICK_STEP_DEG)
click_lat_grid_deg = np.arange(-90, 90 + CLICK_STEP_DEG, CLICK_STEP_DEG)
click_lon_mesh, click_lat_mesh = np.meshgrid(click_lon_grid_deg, click_lat_grid_deg)
click_lon_flat = click_lon_mesh.ravel()
click_lat_flat = click_lat_mesh.ravel()
click_lon_plot = np.where(click_lon_flat > 180, click_lon_flat - 360, click_lon_flat)
click_x, click_y = mollweide_project(click_lon_plot, click_lat_flat)
raster_x, raster_y, raster_visibility = mollweide_visibility_raster(
    ra_grid_deg,
    dec_grid_deg,
    vis_frac_2d,
)

sky_fig = go.Figure()

sky_fig.add_trace(
    go.Heatmap(
        x=raster_x,
        y=raster_y,
        z=raster_visibility,
        colorscale="RdYlGn",
        zmin=0,
        zmax=1,
        colorbar=dict(title="Vis frac<br>(of year)", thickness=14, len=0.7),
        hoverinfo="skip",
        name="visibility fraction",
        showlegend=False,
    )
)

add_mollweide_grid(sky_fig)

sky_fig.add_trace(
    go.Scatter(
        x=click_x,
        y=click_y,
        mode="markers",
        marker=dict(
            color="rgba(0,0,0,0.01)",
            size=7,
            symbol="square",
            line=dict(width=0),
        ),
        customdata=np.column_stack([click_lon_flat, click_lat_flat]),  # original [0, 360) lon
        hovertemplate=(
            f"{lon_label}=%{{customdata[0]:.1f}}°"
            f"<br>{lat_label}=%{{customdata[1]:.1f}}°"
            "<extra></extra>"
        ),
        name="click target grid",
        showlegend=False,
    )
)

# Overlay markers for already-selected targets.
if st.session_state.selected_targets:
    sel_x = []
    sel_y = []
    sel_text = []
    sel_color = []
    sel_hover = []
    sel_customdata = []
    for i, item in enumerate(st.session_state.selected_targets):
        plot_lon = item["coord_lon_deg"]
        if plot_lon > 180:
            plot_lon -= 360
        projected_x, projected_y = mollweide_project(plot_lon, item["coord_lat_deg"])
        sel_x.append(float(projected_x))
        sel_y.append(float(projected_y))
        sel_text.append(str(i + 1))
        sel_color.append(item["color"])
        sel_customdata.append([item["coord_lon_deg"], item["coord_lat_deg"]])
        sel_hover.append(
            f"<b>{html.escape(item['display_label'])}</b>"
            f"<br>Vis: {item['vis_fraction']*100:.1f}%"
        )
    sky_fig.add_trace(
        go.Scatter(
            x=sel_x,
            y=sel_y,
            mode="markers+text",
            text=sel_text,
            textposition="top right",
            textfont=dict(color=sel_color, size=14),
            marker=dict(
                color=sel_color,
                size=16,
                symbol="x-thin",
                line=dict(width=3, color=sel_color),
            ),
            customdata=sel_customdata,
            hovertemplate=[f"{h}<extra></extra>" for h in sel_hover],
            name="selected",
            showlegend=False,
        )
    )

sky_fig.update_layout(
    title=dict(
        text=(
            f"All-Sky Visibility Fraction ({coord_title}) — "
            "click a cell to add a target"
        ),
        x=0.5,
        xanchor="center",
    ),
    xaxis=dict(
        title=f"{lon_label} (deg)",
        range=[-3.05, 3.05],
        showgrid=False,
        zeroline=False,
        tickmode="array",
        tickvals=[
            float(mollweide_project(lon, 0)[0])
            for lon in (-150, -100, -50, 0, 50, 100, 150)
        ],
        ticktext=["-150", "-100", "-50", "0", "50", "100", "150"],
    ),
    yaxis=dict(
        title=f"{lat_label} (deg)",
        range=[-1.58, 1.58],
        showgrid=False,
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
        tickmode="array",
        tickvals=[
            float(mollweide_project(0, lat)[1])
            for lat in (-60, -30, 0, 30, 60)
        ],
        ticktext=["-60", "-30", "0", "30", "60"],
    ),
    margin=dict(l=40, r=20, t=60, b=40),
    height=560,
    plot_bgcolor="white",
    clickmode="event+select",
    dragmode=False,
)

map_event = st.plotly_chart(
    sky_fig,
    use_container_width=True,
    on_select="rerun",
    selection_mode="points",
    key="sky_map",
)

# Click → add target (deduped against last processed click).
if map_event and map_event.selection and map_event.selection.points:
    pt = map_event.selection.points[0]
    click_lon, click_lat = selection_point_lon_lat(pt)
    click_key = (round(click_lon, 4), round(click_lat, 4))
    if st.session_state.last_click_key != click_key:
        st.session_state.last_click_key = click_key
        _add_target(click_lon, click_lat)
        st.rerun()


# --- Manual entry -----------------------------------------------------------

with st.expander("Enter exact coordinates", expanded=False):
    manual_cols = st.columns([1, 1, 1])
    with manual_cols[0]:
        manual_lon = st.number_input(
            f"{lon_label} (degrees)",
            min_value=0.0,
            max_value=360.0,
            value=0.0,
            step=1.0,
            format="%.4f",
            key="manual_lon",
        )
    with manual_cols[1]:
        manual_lat = st.number_input(
            f"{lat_label} (degrees)",
            min_value=-90.0,
            max_value=90.0,
            value=0.0,
            step=1.0,
            format="%.4f",
            key="manual_lat",
        )
    with manual_cols[2]:
        st.write("")  # vertical padding
        if st.button("Add target", type="primary", use_container_width=True):
            _add_target(float(manual_lon), float(manual_lat))
            st.rerun()


# --- Per-target visualizations ---------------------------------------------

selected = st.session_state.selected_targets

if not selected:
    st.info("No targets selected yet. Click the sky map above or use the manual entry to add one.")
    st.stop()


latest = selected[-1]
title_suffix = " [CVZ]" if latest["is_cvz"] else ""

st.subheader(f"Latest target: {latest['display_label']}{title_suffix}")
st.caption(
    f"Visibility fraction: **{latest['vis_fraction'] * 100:.1f}%** "
    f"of the {DURATION_DAYS}-day window."
)

latest_fig = make_visibility_plot(
    latest["dates"],
    latest["good"],
    latest["separation"],
    visibility_title=(
        f"Visibility Window — {latest['display_label']}{title_suffix} "
        f"(vis frac = {latest['vis_fraction'] * 100:.1f}%)"
    ),
    separation_title=f"Sun-Target Separation — {latest['display_label']} ({latest['label']})",
    color=latest["color"],
    line_width=1.8,
    fill_color="gray",
)
st.pyplot(latest_fig, use_container_width=True)


# --- Cumulative gantt + separation comparison ------------------------------

st.subheader("Selected-target Visibility Windows (Gantt; latest target highlighted)")
gantt_fig = make_gantt_chart(selected)
st.pyplot(gantt_fig, use_container_width=True)

st.subheader("Selected-target Sun-Target Separation Comparison")
sep_fig = make_separation_comparison(selected)
st.pyplot(sep_fig, use_container_width=True)


# --- Selected-target table -------------------------------------------------

st.subheader("Selected Targets")
table_rows = []
for i, item in enumerate(selected, start=1):
    icrs = SkyCoord(ra=item["ra_deg"] * u.deg, dec=item["dec_deg"] * u.deg, frame="icrs")
    table_rows.append(
        {
            "#": i,
            "Input": item["display_label"],
            "RA deg": round(item["ra_deg"], 6),
            "Dec deg": round(item["dec_deg"], 6),
            "Galactic l deg": round(float(icrs.galactic.l.deg), 6),
            "Galactic b deg": round(float(icrs.galactic.b.deg), 6),
            "Visible %": round(item["vis_fraction"] * 100, 1),
            "CVZ": "Yes" if item["is_cvz"] else "No",
        }
    )
st.dataframe(table_rows, hide_index=True, use_container_width=True)


# --- Report download -------------------------------------------------------


def _build_report_html() -> str:
    plot_sections = []
    plot_sections.append(
        (
            "Latest Target Visibility",
            figure_img_html(latest_fig, close=False),
        )
    )
    plot_sections.append(
        (
            "Selected-Target Visibility Gantt",
            figure_img_html(gantt_fig, close=False),
        )
    )
    plot_sections.append(
        (
            "All-Target Sun-Target Separation Comparison",
            figure_img_html(sep_fig, close=False),
        )
    )

    sections_html = "\n".join(
        f'<section><h3>{html.escape(title)}</h3>{img_html}</section>'
        for title, img_html in plot_sections
    )

    rows_html = []
    for i, item in enumerate(selected, start=1):
        icrs = SkyCoord(ra=item["ra_deg"] * u.deg, dec=item["dec_deg"] * u.deg, frame="icrs")
        rows_html.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td style='color:{item['color']}; font-weight: 700;'>"
            f"{html.escape(item['display_label'])}</td>"
            f"<td>{item['ra_deg']:.6f}</td>"
            f"<td>{item['dec_deg']:.6f}</td>"
            f"<td>{icrs.galactic.l.deg:.6f}</td>"
            f"<td>{icrs.galactic.b.deg:.6f}</td>"
            f"<td>{item['vis_fraction'] * 100:.1f}%</td>"
            f"<td>{'Yes' if item['is_cvz'] else 'No'}</td>"
            "</tr>"
        )

    table_html = (
        "<table><thead><tr>"
        "<th>#</th><th>Input</th><th>RA deg</th><th>Dec deg</th>"
        "<th>Galactic l deg</th><th>Galactic b deg</th>"
        "<th>Visible fraction</th><th>CVZ flag</th>"
        "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table>"
    )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>RTVT Visibility Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2, h3 {{ color: #1e334c; }}
    .meta {{ color: #53606d; margin-bottom: 18px; }}
    section {{ margin-top: 22px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px; text-align: left; }}
    th {{ background: #f4f6f8; }}
    img {{ max-width: 100%; height: auto; }}
  </style>
</head>
<body>
  <h1>Roman Target Visibility Tool Report</h1>
  <div class="meta">
    Generated {html.escape(datetime.now().isoformat(timespec="seconds"))}<br>
    Coordinate system: {html.escape(coord_title)}<br>
    Duration: {DURATION_DAYS} days; sampling: {SAMPLING_DAYS} day(s)
  </div>
  <section>
    <h2>Selected Targets</h2>
    {table_html}
  </section>
  {sections_html}
</body>
</html>
"""


st.divider()
report_html = _build_report_html()
st.download_button(
    label="Download HTML report",
    data=report_html,
    file_name=f"rtvt_visibility_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
    mime="text/html",
    type="primary",
)
