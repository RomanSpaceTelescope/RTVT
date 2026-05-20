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

import base64
from datetime import datetime
import html
import io
import os
import tempfile
import warnings
from pathlib import Path

import ipywidgets as widgets
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, get_body
from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning
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


TARGET_COLORS = [
    "#1b9e77",  # teal
    "#d95f02",  # orange
    "#7570b3",  # purple
    "#e7298a",  # magenta
    "#66a61e",  # green
    "#e6ab02",  # gold
    "#a6761d",  # brown
    "#1f78b4",  # blue
    "#b2df8a",  # light green
    "#666666",  # gray
]

_SKY_GRID_CACHE = {}


def target_color(index):
    return TARGET_COLORS[index % len(TARGET_COLORS)]


def figure_png_bytes(fig, dpi=120, close=True):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return buf.getvalue()


def figure_img_html(fig, dpi=120, close=True):
    png = figure_png_bytes(fig, dpi=dpi, close=close)
    encoded = base64.b64encode(png).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" style="max-width: 100%; height: auto;">'


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


def normalize_coordinate_system(value):
    normalized = str(value).strip().lower()
    aliases = {
        "equatorial": "equatorial",
        "eq": "equatorial",
        "icrs": "equatorial",
        "radec": "equatorial",
        "ra/dec": "equatorial",
        "galactic": "galactic",
        "gal": "galactic",
        "lb": "galactic",
        "l/b": "galactic",
    }
    if normalized not in aliases:
        raise ValueError("coordinate_system must be 'equatorial' or 'galactic'")
    return aliases[normalized]


def coordinate_labels(coordinate_system):
    if normalize_coordinate_system(coordinate_system) == "galactic":
        return "l", "b", "Galactic"
    return "RA", "Dec", "Equatorial"


def skycoord_from_lon_lat(lon_deg, lat_deg, coordinate_system):
    if normalize_coordinate_system(coordinate_system) == "galactic":
        return SkyCoord(l=lon_deg * u.deg, b=lat_deg * u.deg, frame="galactic")
    return SkyCoord(ra=lon_deg * u.deg, dec=lat_deg * u.deg, frame="icrs")


def display_lon_lat(coord, coordinate_system):
    if normalize_coordinate_system(coordinate_system) == "galactic":
        gal = coord.galactic
        return float(gal.l.deg), float(gal.b.deg)
    icrs = coord.icrs
    return float(icrs.ra.deg), float(icrs.dec.deg)


def format_display_coordinate(lon_deg, lat_deg, coordinate_system, precision=1):
    lon_label, lat_label, _ = coordinate_labels(coordinate_system)
    return f"{lon_label}={lon_deg:.{precision}f}, {lat_label}={lat_deg:.{precision}f}"


def sampled_times_for_interval(start_time=None, duration_days=365, sampling_days=1):
    t_start = _normalize_start_time(start_time)
    if t_start is None:
        t_start = Time(["2024-01-01T00:00:00.0"], format="isot", scale="utc")
    return t_start + np.arange(0.0, duration_days, sampling_days) * u.d


def compute_all_sky_visibility_grid(
    ra_grid=None,
    dec_grid=None,
    grid_step_deg=10,
    start_time=None,
    duration_days=365,
    sampling_days=1,
    coordinate_system="equatorial",
):
    """
    Compute a coarse all-sky visibility-fraction grid for the background map.

    Parameters are intentionally the same style as compute_visibility. If
    ra_grid/dec_grid are omitted, a coarse 10 degree grid is used by default.
    """
    coordinate_system = normalize_coordinate_system(coordinate_system)

    if ra_grid is None:
        ra_grid = np.arange(0, 360, grid_step_deg)
    if dec_grid is None:
        dec_grid = np.arange(-90, 91, grid_step_deg)

    ra_mesh, dec_mesh = np.meshgrid(ra_grid, dec_grid)
    ra_flat = ra_mesh.ravel()
    dec_flat = dec_mesh.ravel()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        sky_points = skycoord_from_lon_lat(ra_flat, dec_flat, coordinate_system).icrs
        sampled_times = sampled_times_for_interval(
            start_time=start_time,
            duration_days=duration_days,
            sampling_days=sampling_days,
        )
        sun_coord = get_body("Sun", sampled_times)
        separation = sun_coord[:, np.newaxis].separation(sky_points[np.newaxis, :])

    good_angles = (separation >= 54 * u.deg) & (separation <= 126 * u.deg)
    vis_frac = np.mean(good_angles, axis=0)

    return np.asarray(ra_grid), np.asarray(dec_grid), vis_frac.reshape(ra_mesh.shape)


def get_cached_all_sky_visibility_grid(
    grid_step_deg=10,
    start_time=None,
    duration_days=365,
    sampling_days=1,
    coordinate_system="equatorial",
):
    """
    Cache default notebook sky maps inside the active Python kernel.

    The first map still performs the visibility calculation; rerunning the cell
    or switching back to a frame reuses the cached grid for the same settings.
    """
    coordinate_system = normalize_coordinate_system(coordinate_system)
    t0 = _normalize_start_time(start_time)
    start_key = "default" if t0 is None else t0.isot
    key = (
        coordinate_system,
        float(grid_step_deg),
        start_key,
        float(duration_days),
        float(sampling_days),
    )
    if key not in _SKY_GRID_CACHE:
        _SKY_GRID_CACHE[key] = compute_all_sky_visibility_grid(
            grid_step_deg=grid_step_deg,
            start_time=t0,
            duration_days=duration_days,
            sampling_days=sampling_days,
            coordinate_system=coordinate_system,
        )
    return _SKY_GRID_CACHE[key]


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
    coordinate_system="equatorial",
):
    """
    Launch the interactive Mollweide selector and cumulative Gantt display.

    If ra_grid, dec_grid, and vis_frac_2d are supplied, they are reused. If any
    are omitted, a coarse all-sky visibility map is computed first.
    """
    coordinate_system = normalize_coordinate_system(coordinate_system)
    lon_label, lat_label, coordinate_title = coordinate_labels(coordinate_system)

    if ra_grid is None and dec_grid is None and vis_frac_2d is None:
        ra_grid, dec_grid, vis_frac_2d = get_cached_all_sky_visibility_grid(
            grid_step_deg=grid_step_deg,
            start_time=start_time,
            duration_days=duration_days,
            sampling_days=sampling_days,
            coordinate_system=coordinate_system,
        )
    elif ra_grid is None or dec_grid is None or vis_frac_2d is None:
        ra_grid, dec_grid, vis_frac_2d = compute_all_sky_visibility_grid(
            ra_grid=ra_grid,
            dec_grid=dec_grid,
            grid_step_deg=grid_step_deg,
            start_time=start_time,
            duration_days=duration_days,
            sampling_days=sampling_days,
            coordinate_system=coordinate_system,
        )

    ra_shifted = np.where(ra_grid > 180, ra_grid - 360, ra_grid)
    sort_idx = np.argsort(ra_shifted)
    ra_sorted_deg = ra_shifted[sort_idx]
    vis_frac_sorted = vis_frac_2d[:, sort_idx]

    ra_plot = np.deg2rad(ra_sorted_deg)
    dec_plot = np.deg2rad(dec_grid)
    ra_plot_mesh, dec_plot_mesh = np.meshgrid(ra_plot, dec_plot)

    with plt.ioff():
        fig_sky, ax_sky = plt.subplots(
            figsize=(12, 6),
            subplot_kw=dict(projection="mollweide"),
        )
    if hasattr(fig_sky.canvas, "toolbar_position"):
        fig_sky.canvas.toolbar_position = "bottom"

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

    if test_targets:
        for tgt in test_targets:
            tgt_lon, tgt_lat = display_lon_lat(tgt, coordinate_system)
            tgt_ra_plot = np.deg2rad(tgt_lon - 360 if tgt_lon > 180 else tgt_lon)
            tgt_dec_plot = np.deg2rad(tgt_lat)
            ax_sky.plot(
                tgt_ra_plot,
                tgt_dec_plot,
                marker="x",
                linestyle="None",
                color="#222222",
                markersize=8,
                markeredgewidth=1.5,
                alpha=0.75,
            )

    ax_sky.set_title(
        f"All-Sky Visibility Fraction ({coordinate_title}) -- Click to Select Targets",
        fontsize=12,
        pad=20,
    )
    ax_sky.grid(True, alpha=0.3)
    fig_sky.tight_layout()

    status_label = widgets.Label(
        value=f"Click on the sky map to select targets in {coordinate_title} coordinates."
    )
    plots_output = widgets.HTML()
    report_button = widgets.Button(
        description="Create report",
        button_style="success",
        tooltip="Write an HTML report with the selected targets and current plots.",
        layout=widgets.Layout(width="160px"),
    )
    report_status = widgets.HTML()
    rendered_plots = {"latest": "", "gantt": "", "comparison": ""}

    selected_targets = []
    selected_markers = []
    selected_number_labels = []

    manual_lon = widgets.Text(
        description=f"{lon_label}:",
        placeholder="decimal degrees",
        layout=widgets.Layout(width="220px"),
        style={"description_width": "42px"},
    )
    manual_lat = widgets.Text(
        description=f"{lat_label}:",
        placeholder="decimal degrees",
        layout=widgets.Layout(width="220px"),
        style={"description_width": "42px"},
    )
    manual_button = widgets.Button(
        description="Compute",
        button_style="primary",
        tooltip="Compute visibility for the typed coordinates.",
        layout=widgets.Layout(width="220px"),
    )
    manual_status = widgets.HTML()
    manual_panel = widgets.VBox(
        [
            widgets.HTML(
                f"<b>Exact {html.escape(coordinate_title)} coordinates</b>"
                "<br><span style='font-size: 12px; color: #666;'>"
                "Enter decimal-degree coordinates, or click the map."
                "</span>"
            ),
            manual_lon,
            manual_lat,
            manual_button,
            manual_status,
        ],
        layout=widgets.Layout(
            width="255px",
            border="1px solid #d6d6d6",
            padding="10px",
            margin="0 12px 0 0",
        ),
    )

    def _plot_section(title, image_html):
        return (
            '<section style="margin-top: 14px;">'
            f'<h3 style="margin: 0 0 6px 0; font-size: 16px;">{title}</h3>'
            f"{image_html}"
            "</section>"
        )

    def _update_plots_output():
        plots_output.value = "\n".join(
            rendered_plots[key]
            for key in ("latest", "gantt", "comparison")
            if rendered_plots[key]
        )

    def _map_radians(lon_deg, lat_deg):
        plot_lon = lon_deg - 360.0 if lon_deg > 180.0 else lon_deg
        return np.deg2rad(plot_lon), np.deg2rad(lat_deg)

    def _target_rows_html():
        if not selected_targets:
            return "<p>No selected targets yet.</p>"

        rows = []
        for i, item in enumerate(selected_targets, start=1):
            target_icrs = SkyCoord(
                ra=item["ra_deg"] * u.deg,
                dec=item["dec_deg"] * u.deg,
                frame="icrs",
            )
            rows.append(
                "<tr>"
                f"<td>{i}</td>"
                f"<td><span style='color:{item['color']}; font-weight: 700;'>"
                f"{html.escape(item['display_label'])}</span></td>"
                f"<td>{item['ra_deg']:.6f}</td>"
                f"<td>{item['dec_deg']:.6f}</td>"
                f"<td>{target_icrs.galactic.l.deg:.6f}</td>"
                f"<td>{target_icrs.galactic.b.deg:.6f}</td>"
                f"<td>{item['vis_fraction'] * 100:.1f}%</td>"
                f"<td>{'Yes' if item['is_cvz'] else 'No'}</td>"
                "</tr>"
            )

        return (
            "<table style='border-collapse: collapse; width: 100%;'>"
            "<thead><tr>"
            "<th>#</th><th>Input coordinates</th><th>RA deg</th><th>Dec deg</th>"
            "<th>Galactic l deg</th><th>Galactic b deg</th>"
            "<th>Visible fraction</th><th>CVZ flag</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
            "<style>th,td{border:1px solid #ddd;padding:6px;text-align:left;}th{background:#f4f6f8;}</style>"
        )

    def _create_report(_event=None):
        if not selected_targets:
            report_status.value = "<span style='color:#b00020;'>Select at least one target first.</span>"
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path.cwd() / f"rtvt_visibility_report_{timestamp}.html"
        plot_sections = "\n".join(
            rendered_plots[key]
            for key in ("latest", "gantt", "comparison")
            if rendered_plots[key]
        )
        sky_snapshot = _plot_section("All-Sky Selection Map", figure_img_html(fig_sky, close=False))
        report_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>RTVT Visibility Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2, h3 {{ color: #1e334c; }}
    .meta {{ color: #53606d; margin-bottom: 18px; }}
    section {{ margin-top: 22px; }}
  </style>
</head>
<body>
  <h1>Roman Target Visibility Tool Report</h1>
  <div class="meta">
    Generated {html.escape(datetime.now().isoformat(timespec="seconds"))}<br>
    Coordinate system: {html.escape(coordinate_title)}<br>
    Duration: {duration_days:g} days; sampling: {sampling_days:g} day(s)
  </div>
  <section>
    <h2>Selected Targets</h2>
    {_target_rows_html()}
  </section>
  {sky_snapshot}
  {plot_sections}
</body>
</html>
"""
        report_path.write_text(report_html, encoding="utf-8")
        report_status.value = f"Report saved: <code>{html.escape(str(report_path))}</code>"

    report_button.on_click(_create_report)

    def _render_latest_visibility(
        display_label,
        radec_label,
        dates_vis,
        good,
        separation,
        vis_fraction,
        is_cvz,
        color,
    ):
        with plt.ioff():
            fig_v, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

        title_suffix = " [CVZ]" if is_cvz else ""

        ax1.step(dates_vis, good.astype(int), where="mid", lw=1.8, color=color)
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(["Not in FOR", "In FOR"])
        ax1.set_ylim(-0.1, 1.1)
        ax1.set_ylabel("Visibility")
        ax1.set_title(
            f"Visibility Window -- {display_label}{title_suffix} "
            f"(vis frac = {vis_fraction * 100:.1f}%)"
        )
        ax1.grid(alpha=0.3)

        ax2.plot(dates_vis, separation, lw=1.8, color=color)
        ax2.axhline(54, ls="--", color="green", lw=1, label="Min (54 deg)")
        ax2.axhline(126, ls="--", color="orange", lw=1, label="Max (126 deg)")
        ax2.fill_between(
            dates_vis,
            54,
            126,
            alpha=0.12,
            color="gray",
            label="Observable range",
        )
        ax2.set_ylabel("Separation (deg)")
        ax2.set_xlabel("Date")
        ax2.set_title(f"Sun-Target Separation -- {display_label} ({radec_label})")
        ax2.legend(fontsize=8, loc="upper right")
        ax2.grid(alpha=0.3)

        ax2.xaxis.set_major_locator(MonthLocator())
        ax2.xaxis.set_major_formatter(DateFormatter("%b %d"))
        for tick in ax2.get_xticklabels():
            tick.set_rotation(45)
        fig_v.tight_layout()

        rendered_plots["latest"] = _plot_section(
            "Latest Target Visibility",
            figure_img_html(fig_v),
        )
        _update_plots_output()

    def _render_selected_gantt():
        if not selected_targets:
            rendered_plots["gantt"] = ""
            rendered_plots["comparison"] = ""
            _update_plots_output()
            return

        with plt.ioff():
            fig_g, ax = plt.subplots(figsize=(13, 0.7 * len(selected_targets) + 1.8))
        latest_index = len(selected_targets) - 1

        for i, item in enumerate(selected_targets):
            if i == latest_index:
                ax.axhspan(
                    i - 0.43,
                    i + 0.43,
                    color=item["color"],
                    alpha=0.16,
                    zorder=0,
                )
            windows = get_observable_windows(item["good"])
            bars = []
            for start_idx, length in windows:
                d_start = item["dates"][start_idx]
                d_end = item["dates"][min(start_idx + length - 1, len(item["dates"]) - 1)]
                width = max(date2num(d_end) - date2num(d_start), 0.5)
                bars.append((date2num(d_start), width))

            facecolor = item["color"]
            if bars:
                ax.broken_barh(
                    bars,
                    (i - 0.35, 0.7),
                    facecolors=facecolor,
                    edgecolors="black",
                    linewidth=0.5,
                    zorder=2,
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
                    color=item["color"],
                )

        labels = [
            f"{i + 1}: {item['display_label']}"
            f"{' [CVZ]' if item['is_cvz'] else ''}"
            for i, item in enumerate(selected_targets)
        ]
        ax.set_yticks(range(len(selected_targets)))
        ax.set_yticklabels(labels, fontsize=8)
        for tick, item in zip(ax.get_yticklabels(), selected_targets):
            tick.set_color(item["color"])
        ax.set_ylim(-0.6, len(selected_targets) - 0.4)
        ax.xaxis.set_major_locator(MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%b %d"))
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        ax.set_xlabel("Date")
        ax.set_title("Selected-target Visibility Windows (Gantt; latest target highlighted)")
        ax.grid(axis="x", alpha=0.3)
        fig_g.tight_layout()

        rendered_plots["gantt"] = _plot_section(
            "Selected-Target Visibility Gantt",
            figure_img_html(fig_g),
        )
        _update_plots_output()

        _render_selected_separation_comparison()

    def _render_selected_separation_comparison():
        if not selected_targets:
            rendered_plots["comparison"] = ""
            _update_plots_output()
            return

        with plt.ioff():
            fig_c, ax = plt.subplots(figsize=(13, 5.2))
        latest_index = len(selected_targets) - 1

        for i, item in enumerate(selected_targets):
            is_latest = i == latest_index
            label = f"{i + 1}: {item['display_label']}"
            ax.plot(
                item["dates"],
                item["separation"],
                lw=3.0 if is_latest else 1.3,
                alpha=1.0 if is_latest else 0.72,
                color=item["color"],
                label=label + (" (latest)" if is_latest else ""),
                zorder=4 if is_latest else 2,
            )

        ax.axhline(54, ls="--", color="green", lw=1, label="Min (54 deg)")
        ax.axhline(126, ls="--", color="orange", lw=1, label="Max (126 deg)")
        ax.fill_between(
            selected_targets[-1]["dates"],
            54,
            126,
            alpha=0.12,
            color="gray",
            label="Observable range",
        )
        ax.set_ylabel("Sun-target separation (deg)")
        ax.set_xlabel("Date")
        ax.set_title("Selected-target Sun-Target Separation Comparison")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
        ax.xaxis.set_major_locator(MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%b %d"))
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        fig_c.tight_layout()

        rendered_plots["comparison"] = _plot_section(
            "All-Target Sun-Target Separation Comparison",
            figure_img_html(fig_c),
        )
        _update_plots_output()

    def _add_target(tgt_click, lon_deg, lat_deg, map_lon_rad, map_lat_rad):
        display_label = format_display_coordinate(lon_deg, lat_deg, coordinate_system, precision=4)
        status_label.value = f"Computing visibility for {display_label}..."
        manual_status.value = ""
        report_status.value = ""
        t0 = _normalize_start_time(start_time)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
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
        marker_color = target_color(target_number - 1)
        marker, = ax_sky.plot(
            [map_lon_rad],
            [map_lat_rad],
            marker="x",
            linestyle="None",
            color=marker_color,
            markersize=15,
            markeredgewidth=3,
            zorder=10,
        )
        number_label = ax_sky.text(
            map_lon_rad,
            map_lat_rad,
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
                display_label=display_label,
                coord_system=coordinate_system,
                coord_lon_deg=float(lon_deg),
                coord_lat_deg=float(lat_deg),
                ra_deg=float(tgt_click.icrs.ra.deg),
                dec_deg=float(tgt_click.icrs.dec.deg),
                dates=dates_vis,
                good=good,
                separation=separation,
                is_cvz=bool(is_cvz),
                vis_fraction=float(vis_fraction),
                color=marker_color,
            )
        )

        fig_sky.canvas.draw_idle()
        _render_latest_visibility(
            display_label,
            radec_label,
            dates_vis,
            good,
            separation,
            vis_fraction,
            is_cvz,
            marker_color,
        )
        _render_selected_gantt()

        status_label.value = (
            f"Selected {len(selected_targets)} target(s). Latest: "
            f"{display_label} -- {radec_label} -- "
            f"Vis: {vis_fraction * 100:.1f}% -- "
            f"{'CVZ' if is_cvz else 'not CVZ'}"
        )

    def _compute_manual_target(_event=None):
        try:
            lon_deg = float(manual_lon.value)
            lat_deg = float(manual_lat.value)
        except ValueError:
            manual_status.value = "<span style='color:#b00020;'>Enter numeric decimal-degree coordinates.</span>"
            return

        if not -90.0 <= lat_deg <= 90.0:
            manual_status.value = "<span style='color:#b00020;'>Latitude must be between -90 and +90 degrees.</span>"
            return

        lon_deg = lon_deg % 360.0
        map_lon_rad, map_lat_rad = _map_radians(lon_deg, lat_deg)
        tgt_manual = skycoord_from_lon_lat(lon_deg, lat_deg, coordinate_system)
        _add_target(tgt_manual, lon_deg, lat_deg, map_lon_rad, map_lat_rad)

    def on_sky_click(event):
        if event.inaxes is not ax_sky or event.xdata is None:
            return

        lon_rad = event.xdata
        lat_rad = event.ydata

        lon_deg = np.degrees(lon_rad)
        if lon_deg < 0:
            lon_deg += 360.0
        lat_deg = np.degrees(lat_rad)

        lon_deg = float(np.clip(lon_deg, 0, 360))
        lat_deg = float(np.clip(lat_deg, -90, 90))
        tgt_click = skycoord_from_lon_lat(lon_deg, lat_deg, coordinate_system)
        _add_target(tgt_click, lon_deg, lat_deg, lon_rad, lat_rad)

    manual_button.on_click(_compute_manual_target)
    fig_sky.canvas.mpl_connect("button_press_event", on_sky_click)

    map_box = widgets.HBox(
        [manual_panel, fig_sky.canvas],
        layout=widgets.Layout(align_items="flex-start"),
    )
    display(map_box)
    display(status_label)
    display(plots_output)
    display(widgets.HBox([report_button, report_status], layout=widgets.Layout(margin="12px 0 0 0")))

    return {
        "fig_sky": fig_sky,
        "ax_sky": ax_sky,
        "status_label": status_label,
        "manual_panel": manual_panel,
        "plots_output": plots_output,
        "report_button": report_button,
        "report_status": report_status,
        "selected_targets": selected_targets,
        "selected_markers": selected_markers,
        "selected_number_labels": selected_number_labels,
    }


def launch_interactive_sky_gantt_with_controls(
    grid_step_deg=10,
    start_time=None,
    duration_days=365,
    sampling_days=1,
    good_angle_threshold=0.99,
):
    """
    Display notebook controls for choosing Equatorial or Galactic coordinates.

    Click Launch/Refresh after changing the coordinate system. The selected
    frame controls the all-sky map axes and the coordinates reported for each
    click; the underlying Roman visibility calculation is performed in ICRS.
    """
    coordinate_selector = widgets.ToggleButtons(
        options=[
            ("Equatorial (RA/Dec)", "equatorial"),
            ("Galactic (l/b)", "galactic"),
        ],
        value="equatorial",
        description="Coords:",
        button_style="",
    )
    launch_button = widgets.Button(
        description="Launch / Refresh",
        button_style="primary",
        tooltip="Create the visibility map with the selected coordinate system.",
    )
    output = widgets.Output()
    state = {"viewer": None}

    def _launch(_event=None):
        with output:
            clear_output(wait=True)
            previous_viewer = state.get("viewer")
            if previous_viewer is not None:
                plt.close(previous_viewer["fig_sky"])
            print(f"Preparing {coordinate_selector.value} all-sky map...")
            state["viewer"] = launch_interactive_sky_gantt(
                grid_step_deg=grid_step_deg,
                start_time=start_time,
                duration_days=duration_days,
                sampling_days=sampling_days,
                good_angle_threshold=good_angle_threshold,
                coordinate_system=coordinate_selector.value,
            )

    coordinate_selector.observe(_launch, names="value")
    launch_button.on_click(_launch)
    display(widgets.VBox([widgets.HBox([coordinate_selector, launch_button]), output]))
    _launch()
    return {
        "coordinate_selector": coordinate_selector,
        "launch_button": launch_button,
        "output": output,
        "state": state,
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
