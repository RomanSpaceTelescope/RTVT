"""ipywidgets interactive picker — the entry point used by the demo notebook."""

from __future__ import annotations

import html
import warnings
from datetime import datetime
from pathlib import Path

import ipywidgets as widgets
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning
from IPython.display import clear_output, display

from rtvt.analysis import check_cvz_status
from rtvt.coords import (
    coordinate_labels,
    format_display_coordinate,
    normalize_coordinate_system,
    skycoord_from_lon_lat,
)
from rtvt.reports import figure_img_html
from rtvt.sky_grid import (
    compute_all_sky_visibility_grid,
    get_cached_all_sky_visibility_grid,
)
from rtvt.utils import prepare_matplotlib_cache, quantity_series_to_deg
from rtvt.visibility import VisibilityCalculator, normalize_start_time
from rtvt.visualization.gantt import (
    make_gantt_chart,
    make_separation_comparison,
    target_color,
)
from rtvt.visualization.sky import make_all_sky_mollweide
from rtvt.visualization.timeseries import make_visibility_plot

prepare_matplotlib_cache()

import matplotlib.pyplot as plt


def _dataframe_doy_to_datetime(df) -> "pd.DatetimeIndex":
    import pandas as pd

    return pd.to_datetime(df.index.astype(str), format="%Y-%j.%f")


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
    """Launch the Mollweide picker plus cumulative Gantt + separation comparison.

    If ``ra_grid``, ``dec_grid``, and ``vis_frac_2d`` are all supplied, they
    are reused. If any are omitted, a coarse all-sky visibility map is
    computed (using the cached result when available).
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

    fig_sky, ax_sky, _pcm = make_all_sky_mollweide(
        ra_grid,
        dec_grid,
        vis_frac_2d,
        coordinate_system=coordinate_system,
        test_targets=test_targets,
    )

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

    selected_targets: list[dict] = []
    selected_markers: list = []
    selected_number_labels: list = []

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
        sky_snapshot = _plot_section(
            "All-Sky Selection Map",
            figure_img_html(fig_sky, close=False),
        )
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

    def _render_latest_visibility(display_label, radec_label, item):
        title_suffix = " [CVZ]" if item["is_cvz"] else ""
        fig_v = make_visibility_plot(
            item["dates"],
            item["good"],
            item["separation"],
            visibility_title=(
                f"Visibility Window -- {display_label}{title_suffix} "
                f"(vis frac = {item['vis_fraction'] * 100:.1f}%)"
            ),
            separation_title=f"Sun-Target Separation -- {display_label} ({radec_label})",
            color=item["color"],
            line_width=1.8,
            fill_color="gray",
        )
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

        fig_g = make_gantt_chart(selected_targets)
        rendered_plots["gantt"] = _plot_section(
            "Selected-Target Visibility Gantt",
            figure_img_html(fig_g),
        )
        _update_plots_output()

        fig_c = make_separation_comparison(selected_targets)
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
        t0 = normalize_start_time(start_time)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            vis = VisibilityCalculator(
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
        dates_vis = _dataframe_doy_to_datetime(df_one)
        separation = quantity_series_to_deg(df_one["separation"])
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

        item = dict(
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
        selected_targets.append(item)

        fig_sky.canvas.draw_idle()
        _render_latest_visibility(display_label, radec_label, item)
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
    """Wrap `launch_interactive_sky_gantt` with an Equatorial/Galactic coord-system toggle.

    Click *Launch / Refresh* after changing the coordinate system. The frame
    controls the map axes and the per-target display coordinates; the
    underlying Roman visibility math always runs in ICRS.
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
