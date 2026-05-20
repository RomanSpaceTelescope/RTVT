"""
Interactive Roman visibility picker with cumulative Gantt chart.

This module contains the code needed by the lightweight notebook
Run_viz_tool_interactive_gantt.ipynb. It can also be imported from another
notebook after enabling an interactive matplotlib backend, for example:

    %matplotlib widget
    from interactive_visibility_gantt import launch_interactive_sky_gantt
    launch_interactive_sky_gantt()
"""

from __future__ import annotations

import io
import os
import tempfile
import warnings
from pathlib import Path

import ipywidgets as widgets
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning
from IPython.display import Image as IPyImage
from IPython.display import clear_output, display

from tgt_vis import compute_visibility


def _prepare_matplotlib_cache():
    """Use a writable cache in locked-down environments."""
    cache_root = Path(tempfile.gettempdir()) / "rtvt-matplotlib"
    xdg_cache_root = Path(tempfile.gettempdir()) / "rtvt-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    xdg_cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_root))


_prepare_matplotlib_cache()

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator, date2num


def to_deg(series):
    """Convert a pandas Series of astropy Quantities or floats to degrees."""
    return np.array(
        [
            x.to_value(u.deg) if hasattr(x, "to_value") else float(x)
            for x in series.values
        ],
        dtype=float,
    )


def get_dates(df):
    """Convert DOY index strings like '2024-001.00000' to a DatetimeIndex."""
    return pd.to_datetime(df.index.astype(str), format="%Y-%j.%f")


def get_observable_windows(good):
    """Return contiguous True runs as (start_index, run_length)."""
    good = np.asarray(good, dtype=bool)
    if not np.any(good):
        return []

    padded = np.concatenate(([False], good, [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return list(zip(starts, ends - starts))


def check_cvz_status(df, max_sep_deg=126.0, good_angle_threshold=0.99):
    """
    Check whether a target meets the notebook's CVZ criterion.

    This preserves the criterion used in Run_viz_tool_dev.ipynb:
    a target is flagged as CVZ if it is visible for at least the threshold
    fraction of sampled times, or if the separation is always above max_sep_deg.
    """
    good = df["good_angles"].astype(bool).values
    vis_frac = float(np.mean(good))
    separation = to_deg(df["separation"])

    always_above_max = np.all(separation > max_sep_deg)
    mostly_good = vis_frac >= good_angle_threshold
    is_cvz = bool(always_above_max or mostly_good)

    if always_above_max:
        info_str = f"IN CVZ: Sun-target separation always > {max_sep_deg} deg"
    elif mostly_good:
        info_str = (
            f"IN CVZ: Good visibility {vis_frac * 100:.1f}% of the year "
            f"(>={good_angle_threshold * 100:.0f}%)"
        )
    else:
        info_str = f"NOT in CVZ: Good visibility only {vis_frac * 100:.1f}% of the year"

    return is_cvz, vis_frac, info_str


def compute_all_sky_visibility_grid(
    ra_grid=None,
    dec_grid=None,
    grid_step_deg=10,
    start_time=None,
    duration_days=365,
    sampling_days=1,
):
    """
    Compute a coarse all-sky visibility-fraction grid for the background map.

    Parameters are intentionally the same style as compute_visibility. If
    ra_grid/dec_grid are omitted, a coarse 10 degree grid is used by default.
    """
    if ra_grid is None:
        ra_grid = np.arange(0, 360, grid_step_deg)
    if dec_grid is None:
        dec_grid = np.arange(-90, 91, grid_step_deg)

    ra_mesh, dec_mesh = np.meshgrid(ra_grid, dec_grid)
    ra_flat = ra_mesh.ravel()
    dec_flat = dec_mesh.ravel()

    sky_points = [
        SkyCoord(ra=r * u.deg, dec=d * u.deg, frame="icrs")
        for r, d in zip(ra_flat, dec_flat)
    ]

    t0 = _normalize_start_time(start_time)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        vis_sky = compute_visibility(
            sky_points,
            report=False,
            fileout=None,
            interval_sampling_days=sampling_days,
            interval_start_time=t0,
            interval_duration_days=duration_days,
        )
        vis_sky.get_good_angles()

    vis_frac = np.zeros(len(sky_points))
    for i, label in enumerate(vis_sky.df_results.index.levels[0]):
        df_pt = vis_sky.df_results.xs(label, level=0)
        vis_frac[i] = df_pt["good_angles"].astype(bool).mean()

    return np.asarray(ra_grid), np.asarray(dec_grid), vis_frac.reshape(ra_mesh.shape)


def launch_interactive_sky_gantt(
    ra_grid=None,
    dec_grid=None,
    vis_frac_2d=None,
    test_targets=None,
    grid_step_deg=10,
    start_time=None,
    duration_days=365,
    sampling_days=1,
    good_angle_threshold=0.99,
):
    """
    Launch the interactive Mollweide selector and cumulative Gantt display.

    If ra_grid, dec_grid, and vis_frac_2d are supplied, they are reused. If any
    are omitted, a coarse all-sky visibility map is computed first.
    """
    if ra_grid is None or dec_grid is None or vis_frac_2d is None:
        ra_grid, dec_grid, vis_frac_2d = compute_all_sky_visibility_grid(
            ra_grid=ra_grid,
            dec_grid=dec_grid,
            grid_step_deg=grid_step_deg,
            start_time=start_time,
            duration_days=duration_days,
            sampling_days=sampling_days,
        )

    if test_targets is None:
        test_targets = [
            SkyCoord("06h00m00s", "-01d00m00s", frame="icrs"),
            SkyCoord("08h00m00s", "60d00m00s", frame="icrs"),
            SkyCoord("08h00m00s", "-60d00m00s", frame="icrs"),
            SkyCoord("17h45m40s", "-29d00m28s", frame="icrs"),
            SkyCoord("09h00m00s", "89d00m00s", frame="icrs"),
            SkyCoord("09h00m00s", "-89d00m00s", frame="icrs"),
        ]

    ra_shifted = np.where(ra_grid > 180, ra_grid - 360, ra_grid)
    sort_idx = np.argsort(ra_shifted)
    ra_sorted_deg = ra_shifted[sort_idx]
    vis_frac_sorted = vis_frac_2d[:, sort_idx]

    ra_plot = np.deg2rad(ra_sorted_deg)
    dec_plot = np.deg2rad(dec_grid)
    ra_plot_mesh, dec_plot_mesh = np.meshgrid(ra_plot, dec_plot)

    fig_sky, ax_sky = plt.subplots(
        figsize=(12, 6),
        subplot_kw=dict(projection="mollweide"),
    )

    pcm = ax_sky.pcolormesh(
        ra_plot_mesh,
        dec_plot_mesh,
        vis_frac_sorted,
        cmap="RdYlGn",
        shading="auto",
        vmin=0,
        vmax=1,
    )
    cbar = fig_sky.colorbar(
        pcm,
        ax=ax_sky,
        orientation="horizontal",
        pad=0.05,
        shrink=0.7,
    )
    cbar.set_label("Visibility Fraction (of year)")

    for tgt in test_targets:
        tgt_ra = tgt.ra.deg
        tgt_dec = tgt.dec.deg
        tgt_ra_plot = np.deg2rad(tgt_ra - 360 if tgt_ra > 180 else tgt_ra)
        tgt_dec_plot = np.deg2rad(tgt_dec)
        ax_sky.plot(
            tgt_ra_plot,
            tgt_dec_plot,
            "w*",
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )

    ax_sky.set_title(
        "All-Sky Visibility Fraction -- Click to Select Targets",
        fontsize=12,
        pad=20,
    )
    ax_sky.grid(True, alpha=0.3)
    fig_sky.tight_layout()

    status_label = widgets.Label(value="Click on the sky map to select targets.")
    vis_output = widgets.Output()
    gantt_output = widgets.Output()

    selected_targets = []
    selected_markers = []
    selected_number_labels = []

    def _render_latest_visibility(radec_label, dates_vis, good, separation, vis_fraction, is_cvz):
        fig_v, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

        title_suffix = " [CVZ]" if is_cvz else ""

        ax1.step(dates_vis, good.astype(int), where="mid", lw=1.5, color="#1f77b4")
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(["Not in FOR", "In FOR"])
        ax1.set_ylim(-0.1, 1.1)
        ax1.set_ylabel("Visibility")
        ax1.set_title(
            f"Visibility Window -- {radec_label}{title_suffix} "
            f"(vis frac = {vis_fraction * 100:.1f}%)"
        )
        ax1.grid(alpha=0.3)

        ax2.plot(dates_vis, separation, lw=1.5, color="#1f77b4")
        ax2.axhline(54, ls="--", color="green", lw=1, label="Min (54 deg)")
        ax2.axhline(126, ls="--", color="orange", lw=1, label="Max (126 deg)")
        ax2.fill_between(
            dates_vis,
            54,
            126,
            alpha=0.12,
            color="blue",
            label="Observable range",
        )
        ax2.set_ylabel("Separation (deg)")
        ax2.set_xlabel("Date")
        ax2.set_title(f"Sun-Target Separation -- {radec_label}")
        ax2.legend(fontsize=8, loc="upper right")
        ax2.grid(alpha=0.3)

        ax2.xaxis.set_major_locator(MonthLocator())
        ax2.xaxis.set_major_formatter(DateFormatter("%b %d"))
        for tick in ax2.get_xticklabels():
            tick.set_rotation(45)
        fig_v.tight_layout()

        buf = io.BytesIO()
        fig_v.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig_v)
        buf.seek(0)

        with vis_output:
            clear_output(wait=True)
            display(IPyImage(data=buf.read()))

    def _render_selected_gantt():
        if not selected_targets:
            with gantt_output:
                clear_output(wait=True)
            return

        fig_g, ax = plt.subplots(figsize=(13, 0.7 * len(selected_targets) + 1.8))
        colors = plt.cm.Set2(np.linspace(0, 1, max(len(selected_targets), 1)))

        for i, item in enumerate(selected_targets):
            windows = get_observable_windows(item["good"])
            bars = []
            for start_idx, length in windows:
                d_start = item["dates"][start_idx]
                d_end = item["dates"][min(start_idx + length - 1, len(item["dates"]) - 1)]
                width = max(date2num(d_end) - date2num(d_start), 0.5)
                bars.append((date2num(d_start), width))

            facecolor = "#2ca02c" if item["is_cvz"] else colors[i]
            if bars:
                ax.broken_barh(
                    bars,
                    (i - 0.35, 0.7),
                    facecolors=facecolor,
                    edgecolors="black",
                    linewidth=0.5,
                )
            else:
                ax.text(
                    0.5,
                    i,
                    "No observable days",
                    transform=ax.get_yaxis_transform(),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="crimson",
                )

        labels = [
            f"{i + 1}: RA={item['ra_deg']:.1f}, Dec={item['dec_deg']:.1f}"
            f"{' [CVZ]' if item['is_cvz'] else ''}"
            for i, item in enumerate(selected_targets)
        ]
        ax.set_yticks(range(len(selected_targets)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_ylim(-0.6, len(selected_targets) - 0.4)
        ax.xaxis.set_major_locator(MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%b %d"))
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        ax.set_xlabel("Date")
        ax.set_title("Selected-target Visibility Windows (Gantt)")
        ax.grid(axis="x", alpha=0.3)
        fig_g.tight_layout()

        buf = io.BytesIO()
        fig_g.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig_g)
        buf.seek(0)

        with gantt_output:
            clear_output(wait=True)
            display(IPyImage(data=buf.read()))

    def on_sky_click(event):
        if event.inaxes is not ax_sky or event.xdata is None:
            return

        lon_rad = event.xdata
        lat_rad = event.ydata

        ra_deg = np.degrees(lon_rad)
        if ra_deg < 0:
            ra_deg += 360.0
        dec_deg = np.degrees(lat_rad)

        ra_deg = np.clip(ra_deg, 0, 360)
        dec_deg = np.clip(dec_deg, -90, 90)

        status_label.value = f"Computing visibility for RA={ra_deg:.1f}, Dec={dec_deg:.1f}..."

        t0 = _normalize_start_time(start_time)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            tgt_click = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
            vis = compute_visibility(
                tgt_click,
                report=False,
                fileout=None,
                interval_sampling_days=sampling_days,
                interval_start_time=t0,
                interval_duration_days=duration_days,
            )
            vis.compute_and_display()

        radec_label = vis.df_results.index.levels[0][0]
        df_one = vis.df_results.xs(radec_label, level=0)

        good = df_one["good_angles"].astype(bool).values
        dates_vis = get_dates(df_one)
        separation = to_deg(df_one["separation"])
        is_cvz, vis_fraction, _ = check_cvz_status(
            df_one,
            good_angle_threshold=good_angle_threshold,
        )

        target_number = len(selected_targets) + 1
        marker_color = "limegreen" if is_cvz else "red"
        marker, = ax_sky.plot(
            [lon_rad],
            [lat_rad],
            marker="x",
            linestyle="None",
            color=marker_color,
            markersize=15,
            markeredgewidth=3,
            zorder=10,
        )
        number_label = ax_sky.text(
            lon_rad,
            lat_rad,
            f" {target_number}",
            color=marker_color,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
            zorder=11,
        )
        selected_markers.append(marker)
        selected_number_labels.append(number_label)

        selected_targets.append(
            dict(
                label=radec_label,
                ra_deg=float(ra_deg),
                dec_deg=float(dec_deg),
                dates=dates_vis,
                good=good,
                separation=separation,
                is_cvz=bool(is_cvz),
                vis_fraction=float(vis_fraction),
            )
        )

        fig_sky.canvas.draw_idle()
        _render_latest_visibility(
            radec_label,
            dates_vis,
            good,
            separation,
            vis_fraction,
            is_cvz,
        )
        _render_selected_gantt()

        status_label.value = (
            f"Selected {len(selected_targets)} target(s). Latest: "
            f"RA={ra_deg:.1f}, Dec={dec_deg:.1f} -- {radec_label} -- "
            f"Vis: {vis_fraction * 100:.1f}% -- "
            f"{'CVZ' if is_cvz else 'not CVZ'}"
        )

    fig_sky.canvas.mpl_connect("button_press_event", on_sky_click)

    plt.show()
    display(status_label)
    display(vis_output)
    display(gantt_output)

    return {
        "fig_sky": fig_sky,
        "ax_sky": ax_sky,
        "status_label": status_label,
        "vis_output": vis_output,
        "gantt_output": gantt_output,
        "selected_targets": selected_targets,
        "selected_markers": selected_markers,
        "selected_number_labels": selected_number_labels,
    }


def _normalize_start_time(start_time):
    if start_time is None:
        return None
    if isinstance(start_time, Time):
        return start_time
    if isinstance(start_time, str):
        if "T" in start_time:
            return Time(start_time, format="isot", scale="utc")
        return Time(f"{start_time}T00:00:00.0", format="isot", scale="utc")
    raise TypeError("start_time must be None, an astropy Time, or an ISO date string")
