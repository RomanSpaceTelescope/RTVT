"""Figure builders for the Roman Target Visibility Tool.

All public figure builders are flat-imported here so callers can write
``from rtvt.visualization import make_visibility_plot`` etc.
"""

from rtvt.visualization.gantt import (
    TARGET_COLOR_PALETTE,
    make_gantt_chart,
    make_separation_comparison,
    target_color,
)
from rtvt.visualization.sky import make_all_sky_mollweide
from rtvt.visualization.suntrack3d import roman_suntrack_3d
from rtvt.visualization.timeseries import make_visibility_plot, plot_visibility_result

__all__ = [
    "TARGET_COLOR_PALETTE",
    "make_all_sky_mollweide",
    "make_gantt_chart",
    "make_separation_comparison",
    "make_visibility_plot",
    "plot_visibility_result",
    "roman_suntrack_3d",
    "target_color",
]
