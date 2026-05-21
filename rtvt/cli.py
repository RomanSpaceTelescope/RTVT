"""Command-line interface for the Roman Target Visibility Tool."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from astropy.time import Time

from rtvt.analysis import VisibilityResult, format_summary
from rtvt.coords import normalize_coordinate_system, parse_target
from rtvt.reports import write_csv, write_report
from rtvt.utils import prepare_matplotlib_cache
from rtvt.visibility import VisibilityCalculator
from rtvt.visualization.timeseries import plot_visibility_result

__all__ = ["build_parser", "main", "run_visibility"]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = run_visibility(args)

    if args.write_csv:
        write_csv(result, args.write_csv)

    if not args.quiet:
        print(format_summary(result, target_name=args.target_name))

    if args.write_plot or args.show_plot:
        _handle_plot(
            result,
            write_path=args.write_plot,
            show=args.show_plot,
            target_name=args.target_name,
        )

    if args.write_report:
        write_report(result, args.write_report, target_name=args.target_name)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rtvt",
        description="Roman Target Visibility Tool fixed-target calculator.",
    )
    parser.add_argument(
        "--ra",
        "--lon",
        dest="ra",
        required=True,
        help="Input longitude: RA for equatorial coordinates, or Galactic l for galactic coordinates.",
    )
    parser.add_argument(
        "--dec",
        "--lat",
        dest="dec",
        required=True,
        help="Input latitude: Dec for equatorial coordinates, or Galactic b for galactic coordinates.",
    )
    parser.add_argument("--start-date", help="Start date in YYYY-MM-DD format. Defaults to the calculator default.")
    parser.add_argument("--duration-days", type=float, default=365.0, help="Total duration to sample in days.")
    parser.add_argument("--sampling-days", type=float, default=1.0, help="Sampling cadence in days.")
    parser.add_argument(
        "--coordinate-system",
        "--coords",
        choices=["equatorial", "galactic"],
        default="equatorial",
        help="Coordinate system for the input pair. Equatorial uses RA/Dec; Galactic uses l/b.",
    )
    parser.add_argument("--target-name", help="Optional display name for the target.")
    parser.add_argument("--write-csv", help="Write the sampled visibility table to this CSV path.")
    parser.add_argument("--write-plot", help="Write a static visibility plot to this image path.")
    parser.add_argument("--write-report", help="Write an HTML visibility report to this path.")
    parser.add_argument("--show-plot", action="store_true", help="Display the static visibility plot.")
    parser.add_argument("--quiet", action="store_true", help="Suppress terminal summary output.")
    parser.add_argument("--version", action="version", version="rtvt 0.1.0")
    return parser


def run_visibility(args: argparse.Namespace) -> VisibilityResult:
    """Translate parsed CLI args into a single-target ``VisibilityResult``."""
    coordinate_system = normalize_coordinate_system(
        getattr(args, "coordinate_system", "equatorial")
    )
    target = parse_target(args.ra, args.dec, coordinate_system=coordinate_system)
    target_icrs = target.icrs
    start_time = _parse_start_date(args.start_date)

    vis = VisibilityCalculator(
        target_icrs,
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
        target=target_icrs,
        label=label,
        table=table,
        sampled_times=vis.sampled_times,
        sampling_days=float(args.sampling_days),
        coordinate_system=coordinate_system,
    )


def _parse_start_date(start_date: str | None) -> Time | None:
    if start_date is None:
        return None
    return Time(f"{start_date}T00:00:00.0", format="isot", scale="utc")


def _handle_plot(
    result: VisibilityResult,
    write_path: str | None = None,
    show: bool = False,
    target_name: str | None = None,
) -> None:
    if not show and write_path:
        _write_plot_file(result, write_path, target_name=target_name)
        return

    prepare_matplotlib_cache()

    import matplotlib
    import matplotlib.pyplot as plt

    fig = plot_visibility_result(result, target_name=target_name)

    if write_path:
        Path(write_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(write_path, dpi=150, bbox_inches="tight")

    if show:
        backend = matplotlib.get_backend()
        if _matplotlib_backend_can_show(backend):
            plt.show()
        else:
            print(
                "Plot display skipped because Matplotlib is using the "
                f"non-interactive '{backend}' backend. "
                "Use --write-plot to save the figure, or run from a terminal "
                "with an interactive Matplotlib backend.",
                file=sys.stderr,
            )

    plt.close(fig)


def _write_plot_file(result: VisibilityResult, path: str, target_name: str | None = None) -> None:
    prepare_matplotlib_cache()

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig = plot_visibility_result(result, target_name=target_name)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _matplotlib_backend_can_show(backend: str) -> bool:
    non_interactive = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
    return backend.lower() not in non_interactive


if __name__ == "__main__":
    raise SystemExit(main())
