"""CSV and HTML report writers for `VisibilityResult` outputs."""

from __future__ import annotations

import base64
import html
from pathlib import Path

from rtvt.analysis import VisibilityResult, format_summary, summarize_windows
from rtvt.utils import figure_to_png_bytes, prepare_matplotlib_cache
from rtvt.visualization.timeseries import plot_visibility_result


def write_csv(result: VisibilityResult, path: str) -> None:
    """Write the sampled visibility table to a CSV file, prepending the ISO timestamp column."""
    output = result.table.copy()
    output.insert(0, "time_isot", [time.isot for time in result.sampled_times])
    output.to_csv(path, index=True)


def figure_img_html(fig, dpi: int = 120, close: bool = True) -> str:
    """Encode a matplotlib figure as a self-contained inline ``<img>`` tag."""
    png = figure_to_png_bytes(fig, dpi=dpi, close=close)
    encoded = base64.b64encode(png).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" style="max-width: 100%; height: auto;">'


def make_html_report(
    result: VisibilityResult,
    target_name: str | None = None,
    plot_png: bytes | None = None,
) -> str:
    """Build a single-target HTML report (summary + plot + windows + data table)."""
    title = target_name or result.label
    windows = summarize_windows(result)
    sampled = result.table.copy()
    sampled.insert(0, "time_isot", [time.isot for time in result.sampled_times])
    if plot_png is None:
        prepare_matplotlib_cache()
        fig = plot_visibility_result(result, target_name=target_name)
        plot_png = figure_to_png_bytes(fig, close=True)

    plot_encoded = base64.b64encode(plot_png).decode("ascii")
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>RTVT Report - {html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ color: #1e334c; }}
    pre {{ background: #f4f6f8; padding: 14px; white-space: pre-wrap; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 5px; text-align: left; }}
    th {{ background: #f4f6f8; }}
    img {{ max-width: 100%; height: auto; }}
  </style>
</head>
<body>
  <h1>Roman Target Visibility Tool Report</h1>
  <h2>Summary</h2>
  <pre>{html.escape(format_summary(result, target_name=target_name))}</pre>
  <h2>Visibility Plot</h2>
  <img src="data:image/png;base64,{plot_encoded}">
  <h2>Observable Windows</h2>
  {windows.to_html(index=False) if not windows.empty else "<p>No observable windows found.</p>"}
  <h2>Sampled Data</h2>
  {sampled.to_html(index=True)}
</body>
</html>
"""


def write_report(result: VisibilityResult, path: str, target_name: str | None = None) -> None:
    """Write the single-target HTML report to ``path``."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(make_html_report(result, target_name=target_name), encoding="utf-8")
