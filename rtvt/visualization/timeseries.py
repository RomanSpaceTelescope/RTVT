"""2-panel matplotlib figure: visibility step plot + Sun-target separation."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from rtvt.analysis import VisibilityResult
from rtvt.utils import quantity_series_to_deg


def make_visibility_plot(
    dates: Sequence,
    good_mask: np.ndarray,
    separation_deg: np.ndarray,
    *,
    visibility_title: str,
    separation_title: str | None = None,
    color: str = "#1f77b4",
    line_width: float = 1.5,
    fill_color: str = "blue",
):
    """Build a 2-panel figure (in-FOR step + Sun-target separation) and return it.

    The caller owns the figure (closing, saving, displaying). Both panels share
    the X axis. Pass ``separation_title`` to label the lower panel; the CLI
    leaves it blank, the notebook uses a per-target string.
    """
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter, MonthLocator

    fig, (ax_vis, ax_sep) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    ax_vis.step(dates, np.asarray(good_mask, dtype=int), where="mid", lw=line_width, color=color)
    ax_vis.set_yticks([0, 1])
    ax_vis.set_yticklabels(["Not in FOR", "In FOR"])
    ax_vis.set_ylim(-0.1, 1.1)
    ax_vis.set_ylabel("Visibility")
    ax_vis.set_title(visibility_title)
    ax_vis.grid(alpha=0.3)

    ax_sep.plot(dates, separation_deg, lw=line_width, color=color)
    ax_sep.axhline(54, ls="--", color="green", lw=1, label="Min (54 deg)")
    ax_sep.axhline(126, ls="--", color="orange", lw=1, label="Max (126 deg)")
    ax_sep.fill_between(dates, 54, 126, alpha=0.12, color=fill_color, label="Observable range")
    ax_sep.set_ylabel("Separation (deg)")
    ax_sep.set_xlabel("Date")
    if separation_title:
        ax_sep.set_title(separation_title)
    ax_sep.legend(fontsize=8, loc="upper right")
    ax_sep.grid(alpha=0.3)
    ax_sep.xaxis.set_major_locator(MonthLocator())
    ax_sep.xaxis.set_major_formatter(DateFormatter("%b %d"))
    for tick in ax_sep.get_xticklabels():
        tick.set_rotation(45)

    fig.tight_layout()
    return fig


def plot_visibility_result(result: VisibilityResult, target_name: str | None = None):
    """CLI-style 2-panel plot for a VisibilityResult."""
    title = target_name or result.label
    dates = [time.datetime for time in result.sampled_times]
    good = result.table["good_angles"].astype(bool).to_numpy()
    separation = quantity_series_to_deg(result.table["separation"])
    return make_visibility_plot(
        dates,
        good,
        separation,
        visibility_title=f"Roman visibility window: {title}",
    )
