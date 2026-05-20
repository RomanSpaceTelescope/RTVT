"""Command-line interface for the Roman Target Visibility Tool."""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from tgt_vis import compute_visibility


@dataclass
class VisibilityResult:
    target: SkyCoord
    label: str
    table: pd.DataFrame
    sampled_times: Time
    sampling_days: float


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = run_visibility(args)

    if args.write_csv:
        write_csv(result, args.write_csv)

    if not args.quiet:
        print(format_summary(result, target_name=args.target_name))

    if args.write_plot:
        write_plot(result, args.write_plot, target_name=args.target_name)

    if args.show_plot:
        import matplotlib.pyplot as plt

        fig = make_plot(result, target_name=args.target_name)
        plt.show()
        plt.close(fig)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rtvt",
        description="Roman Target Visibility Tool fixed-target calculator.",
    )
    parser.add_argument("--ra", required=True, help="Right ascension in degrees or sexagesimal hour angle.")
    parser.add_argument("--dec", required=True, help="Declination in degrees or sexagesimal degrees.")
    parser.add_argument("--start-date", help="Start date in YYYY-MM-DD format. Defaults to the calculator default.")
    parser.add_argument("--duration-days", type=float, default=365.0, help="Total duration to sample in days.")
    parser.add_argument("--sampling-days", type=float, default=1.0, help="Sampling cadence in days.")
    parser.add_argument("--target-name", help="Optional display name for the target.")
    parser.add_argument("--write-csv", help="Write the sampled visibility table to this CSV path.")
    parser.add_argument("--write-plot", help="Write a static visibility plot to this image path.")
    parser.add_argument("--show-plot", action="store_true", help="Display the static visibility plot.")
    parser.add_argument("--quiet", action="store_true", help="Suppress terminal summary output.")
    parser.add_argument("--version", action="version", version="rtvt 0.1.0")
    return parser


def run_visibility(args: argparse.Namespace) -> VisibilityResult:
    target = parse_target(args.ra, args.dec)
    start_time = parse_start_date(args.start_date)

    vis = compute_visibility(
        target,
        report=False,
        fileout=None,
        interval_sampling_days=args.sampling_days,
        interval_start_time=start_time,
        interval_duration_days=args.duration_days,
    )
    vis.compute_and_display()

    label = vis.df_results.index.levels[0][0]
    table = vis.df_results.xs(label, level=0).copy()
    return VisibilityResult(
        target=target,
        label=label,
        table=table,
        sampled_times=vis.sampled_times,
        sampling_days=float(args.sampling_days),
    )


def parse_target(ra: str, dec: str) -> SkyCoord:
    ra_unit = u.hourangle if _looks_sexagesimal_ra(ra) else u.deg
    return SkyCoord(ra, dec, unit=(ra_unit, u.deg), frame="icrs")


def parse_start_date(start_date: str | None) -> Time | None:
    if start_date is None:
        return None
    return Time(f"{start_date}T00:00:00.0", format="isot", scale="utc")


def format_summary(result: VisibilityResult, target_name: str | None = None) -> str:
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


def summarize_windows(result: VisibilityResult) -> pd.DataFrame:
    good = result.table["good_angles"].astype(bool).to_numpy()
    if not np.any(good):
        return pd.DataFrame(
            columns=[
                "window_start",
                "window_end",
                "duration_days",
                "nominal_roll_start",
                "nominal_roll_end",
            ]
        )

    starts, ends = _true_runs(good)
    rows = []
    rolls = _to_float_array(result.table["nominal_roll"])
    for start, end in zip(starts, ends):
        last = end - 1
        rows.append(
            {
                "window_start": result.sampled_times[start].isot,
                "window_end": result.sampled_times[last].isot,
                "duration_days": (end - start) * result.sampling_days,
                "nominal_roll_start": rolls[start],
                "nominal_roll_end": rolls[last],
            }
        )
    return pd.DataFrame(rows)


def write_csv(result: VisibilityResult, path: str) -> None:
    output = result.table.copy()
    output.insert(0, "time_isot", [time.isot for time in result.sampled_times])
    output.to_csv(path, index=True)


def write_plot(result: VisibilityResult, path: str, target_name: str | None = None) -> None:
    prepare_matplotlib_file_output()

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig = make_plot(result, target_name=target_name)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_plot(result: VisibilityResult, target_name: str | None = None):
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter, MonthLocator

    dates = [time.datetime for time in result.sampled_times]
    good = result.table["good_angles"].astype(bool).to_numpy()
    separation = _to_float_array(result.table["separation"])
    title = target_name or result.label

    fig, (ax_vis, ax_sep) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    ax_vis.step(dates, good.astype(int), where="mid", lw=1.5)
    ax_vis.set_yticks([0, 1])
    ax_vis.set_yticklabels(["Not in FOR", "In FOR"])
    ax_vis.set_ylim(-0.1, 1.1)
    ax_vis.set_ylabel("Visibility")
    ax_vis.set_title(f"Roman visibility window: {title}")
    ax_vis.grid(alpha=0.3)

    ax_sep.plot(dates, separation, lw=1.5)
    ax_sep.axhline(54, ls="--", color="green", lw=1, label="Min (54 deg)")
    ax_sep.axhline(126, ls="--", color="orange", lw=1, label="Max (126 deg)")
    ax_sep.fill_between(dates, 54, 126, alpha=0.12, color="blue", label="Observable range")
    ax_sep.set_ylabel("Separation (deg)")
    ax_sep.set_xlabel("Date")
    ax_sep.legend(fontsize=8, loc="upper right")
    ax_sep.grid(alpha=0.3)
    ax_sep.xaxis.set_major_locator(MonthLocator())
    ax_sep.xaxis.set_major_formatter(DateFormatter("%b %d"))
    for tick in ax_sep.get_xticklabels():
        tick.set_rotation(45)

    fig.tight_layout()
    return fig


def _true_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return starts, ends


def _to_float_array(series: pd.Series) -> np.ndarray:
    return np.array(
        [
            value.to_value(u.deg) if hasattr(value, "to_value") else float(value)
            for value in series.to_numpy()
        ],
        dtype=float,
    )


def _looks_sexagesimal_ra(value: str) -> bool:
    lowered = value.lower()
    return ":" in value or "h" in lowered or "m" in lowered or "s" in lowered


def prepare_matplotlib_file_output() -> None:
    cache_root = Path(tempfile.gettempdir()) / "rtvt-matplotlib"
    xdg_cache_root = Path(tempfile.gettempdir()) / "rtvt-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    xdg_cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_root))


if __name__ == "__main__":
    raise SystemExit(main())
