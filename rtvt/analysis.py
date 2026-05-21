"""Analysis of `VisibilityCalculator` outputs: observable windows, summary text, CVZ check."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time

from rtvt.utils import quantity_series_to_deg


@dataclass
class VisibilityResult:
    """Single-target view of a `VisibilityCalculator` run."""

    target: SkyCoord
    label: str
    table: pd.DataFrame
    sampled_times: Time
    sampling_days: float
    coordinate_system: str = "equatorial"


def find_observable_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous True runs in ``mask`` as ``(start_index, length)`` tuples."""
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return []

    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return list(zip(starts.tolist(), (ends - starts).tolist()))


def summarize_windows(result: VisibilityResult) -> pd.DataFrame:
    """Return a DataFrame of observable windows with start/end/duration/roll bookends."""
    good = result.table["good_angles"].astype(bool).to_numpy()
    columns = [
        "window_start",
        "window_end",
        "duration_days",
        "nominal_roll_start",
        "nominal_roll_end",
    ]
    if not np.any(good):
        return pd.DataFrame(columns=columns)

    rolls = quantity_series_to_deg(result.table["nominal_roll"])
    rows = []
    for start, length in find_observable_runs(good):
        end_exclusive = start + length
        last = end_exclusive - 1
        rows.append(
            {
                "window_start": result.sampled_times[start].isot,
                "window_end": result.sampled_times[last].isot,
                "duration_days": length * result.sampling_days,
                "nominal_roll_start": rolls[start],
                "nominal_roll_end": rolls[last],
            }
        )
    return pd.DataFrame(rows)


def format_summary(result: VisibilityResult, target_name: str | None = None) -> str:
    """Render the terminal summary block (target metadata + windows table)."""
    good = result.table["good_angles"].astype(bool).to_numpy()
    windows = summarize_windows(result)
    vis_fraction = float(np.mean(good)) if len(good) else 0.0
    start = result.sampled_times[0].isot
    end = result.sampled_times[-1].isot
    title = target_name or result.label

    lines = [
        "Roman Target Visibility Tool",
        f"Target: {title}",
        f"RA: {result.target.ra.deg:.8f} deg",
        f"Dec: {result.target.dec.deg:.8f} deg",
        f"Galactic l: {result.target.galactic.l.deg:.8f} deg",
        f"Galactic b: {result.target.galactic.b.deg:.8f} deg",
        f"Input coordinate system: {result.coordinate_system}",
        f"Checked interval: {start} to {end}",
        f"Sampling cadence: {result.sampling_days:g} day(s)",
        f"Visible samples: {int(np.sum(good))}/{len(good)} ({vis_fraction * 100:.1f}%)",
        "",
    ]

    if windows.empty:
        lines.append("No observable windows found.")
    else:
        lines.append("Observable windows:")
        lines.append(windows.to_string(index=False))

    return "\n".join(lines)


def check_cvz_status(
    df: pd.DataFrame,
    max_sep_deg: float = 126.0,
    good_angle_threshold: float = 0.99,
) -> tuple[bool, float, str]:
    """Notebook CVZ heuristic: target is in CVZ if visible most of the year.

    Returns ``(is_cvz, visible_fraction, info_string)``. A target qualifies when
    visibility fraction meets ``good_angle_threshold``, or when the Sun-target
    separation is always strictly above ``max_sep_deg``.
    """
    good = df["good_angles"].astype(bool).values
    vis_frac = float(np.mean(good))
    separation = quantity_series_to_deg(df["separation"])

    always_above_max = bool(np.all(separation > max_sep_deg))
    mostly_good = vis_frac >= good_angle_threshold
    is_cvz = always_above_max or mostly_good

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
