"""
roman_visibility_3d.py

Plotly 3D visualization of Sun-track relative to a fixed target on the celestial sphere,
colored by Roman visibility (good_angles) computed via tgt_vis.compute_visibility.

Notebook usage:
    from roman_visibility_3d import roman_suntrack_3d
    fig = roman_suntrack_3d("06h00m00s", "-01d00m00s", start_date="2024-01-01",
                            duration_days=365, sampling_days=1)
    fig.show()

Requires:
  - plotly
  - astropy
  - numpy, pandas
  - your tgt_vis.py (compute_visibility)
"""

from __future__ import annotations

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u

import plotly.graph_objects as go

from tgt_vis import compute_visibility


def _radec_unitvec(ra_rad: np.ndarray, dec_rad: np.ndarray):
    """
    Convert RA/Dec in radians to Cartesian unit vectors.
    Returns arrays x,y,z.
    """
    cosd = np.cos(dec_rad)
    x = cosd * np.cos(ra_rad)
    y = cosd * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return x, y, z


def _sphere_mesh(n_lon=60, n_lat=30):
    """
    Generate a unit sphere mesh.
    """
    lon = np.linspace(0, 2*np.pi, n_lon)
    lat = np.linspace(-np.pi/2, np.pi/2, n_lat)
    lon2, lat2 = np.meshgrid(lon, lat)
    x = np.cos(lat2) * np.cos(lon2)
    y = np.cos(lat2) * np.sin(lon2)
    z = np.sin(lat2)
    return x, y, z

def _angle_to_rad(x):
    if hasattr(x, "to_value"):
        return x.to_value(u.rad)
    else:
        # assume float is in degrees
        return np.deg2rad(float(x))

def roman_suntrack_3d(
    ra: str,
    dec: str,
    frame: str = "icrs",
    start_date: str | None = None,   # "YYYY-MM-DD"
    duration_days: float = 365,
    sampling_days: float = 1,
    show_sphere: bool = True,
    sphere_opacity: float = 0.12,
    marker_size: int = 5,
):
    """
    Returns a Plotly Figure.

    Parameters
    ----------
    ra, dec : str
        Target coordinates (sexagesimal or degrees recognized by SkyCoord)
    start_date : str or None
        "YYYY-MM-DD" start date for sampling. If None, uses your class default (2024-01-01).
    duration_days : float
        Total time span to sample
    sampling_days : float
        Cadence in days
    """

    tgt_input = SkyCoord(ra, dec, frame=frame)
    tgt = tgt_input.icrs

    t0 = None
    if start_date:
        t0 = Time(f"{start_date}T00:00:00.0", format="isot", scale="utc")

    vis = compute_visibility(
        tgt,
        report=False,
        fileout=None,
        interval_sampling_days=float(sampling_days),
        interval_start_time=t0,
        interval_duration_days=float(duration_days),
    )
    vis.compute_and_display()

    # Slice single target results
    label = vis.df_results.index.levels[0][0]
    df = vis.df_results.xs(label, level=0)

    # Sun track vectors
    sun_ra = np.array([_angle_to_rad(x) for x in df["Sun_RA"].values], dtype=float)
    sun_dec = np.array([_angle_to_rad(x) for x in df["Sun_Dec"].values], dtype=float)

    sx, sy, sz = _radec_unitvec(sun_ra, sun_dec)

    # Target vector
    tx, ty, tz = _radec_unitvec(
        np.array([tgt.ra.to_value(u.rad)]),
        np.array([tgt.dec.to_value(u.rad)]),
    )

    # Visibility mask and time labels
    good = df["good_angles"].astype(bool).values
    times = vis.sampled_times
    time_str = np.array([t.isot for t in times], dtype=object)

    # Break into segments so we can color good vs not-good cleanly
    sx_good = np.where(good, sx, np.nan)
    sy_good = np.where(good, sy, np.nan)
    sz_good = np.where(good, sz, np.nan)

    sx_bad = np.where(~good, sx, np.nan)
    sy_bad = np.where(~good, sy, np.nan)
    sz_bad = np.where(~good, sz, np.nan)

    fig = go.Figure()

    if show_sphere:
        mx, my, mz = _sphere_mesh()
        fig.add_trace(go.Surface(
            x=mx, y=my, z=mz,
            opacity=sphere_opacity,
            showscale=False,
            hoverinfo="skip",
        ))

    # Sun track (good)
    fig.add_trace(go.Scatter3d(
        x=sx_good, y=sy_good, z=sz_good,
        mode="lines+markers",
        marker=dict(size=marker_size),
        line=dict(width=5),
        name="Sun track (IN FOR)",
        text=time_str,
        hovertemplate="Time: %{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
    ))

    # Sun track (bad)
    fig.add_trace(go.Scatter3d(
        x=sx_bad, y=sy_bad, z=sz_bad,
        mode="lines+markers",
        marker=dict(size=marker_size),
        line=dict(width=5, dash="dash"),
        name="Sun track (OUT of FOR)",
        text=time_str,
        hovertemplate="Time: %{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
    ))

    # Target marker
    fig.add_trace(go.Scatter3d(
        x=tx, y=ty, z=tz,
        mode="markers+text",
        marker=dict(size=10),
        text=["Target"],
        textposition="top center",
        name="Target",
        hovertemplate=(
            f"Target<br>RA={tgt.ra.to_string(u.hour)}"
            f"<br>Dec={tgt.dec.to_string(u.deg, alwayssign=True)}"
            f"<br>l={tgt.galactic.l.deg:.3f} deg"
            f"<br>b={tgt.galactic.b.deg:.3f} deg"
            "<extra></extra>"
        ),
    ))

    # A tiny line from origin to target to make it obvious
    fig.add_trace(go.Scatter3d(
        x=[0, float(tx[0])], y=[0, float(ty[0])], z=[0, float(tz[0])],
        mode="lines",
        line=dict(width=6),
        name="LOS (origin→target)",
        hoverinfo="skip",
    ))

    fig.update_layout(
        title=f"Roman Visibility Geometry: Sun track vs Target on Unit Sphere<br><sub>{label}</sub>",
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=70, b=0),
    )

    return fig
