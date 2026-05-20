import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from rtvt import compute_visibility
from rtvt.cli import build_parser, format_summary, parse_target, run_visibility, summarize_windows


def test_compute_visibility_returns_expected_columns():
    target = SkyCoord("06h00m00s", "-01d00m00s", frame="icrs")
    vis = compute_visibility(
        target,
        report=False,
        fileout=None,
        interval_sampling_days=1,
        interval_start_time=Time("2024-01-01T00:00:00.0", format="isot", scale="utc"),
        interval_duration_days=5,
    )
    vis.compute_and_display()

    label = vis.df_results.index.levels[0][0]
    table = vis.df_results.xs(label, level=0)

    assert len(table) == 5
    assert "good_angles" in table.columns
    assert "nominal_roll" in table.columns
    assert float(table["separation"].iloc[0]) > 0


def test_cli_summary_and_windows_smoke():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--ra",
            "90.0",
            "--dec",
            "-1.0",
            "--start-date",
            "2024-01-01",
            "--duration-days",
            "5",
        ]
    )
    result = run_visibility(args)
    summary = format_summary(result, target_name="smoke")
    windows = summarize_windows(result)

    assert "Roman Target Visibility Tool" in summary
    assert "smoke" in summary
    assert len(result.table) == 5
    assert set(windows.columns) == {
        "window_start",
        "window_end",
        "duration_days",
        "nominal_roll_start",
        "nominal_roll_end",
    }


def test_galactic_coordinate_input_smoke():
    target = parse_target("0.0", "0.0", coordinate_system="galactic")
    assert target.frame.name == "galactic"

    parser = build_parser()
    args = parser.parse_args(
        [
            "--coordinate-system",
            "galactic",
            "--lon",
            "0.0",
            "--lat",
            "0.0",
            "--start-date",
            "2024-01-01",
            "--duration-days",
            "5",
        ]
    )
    result = run_visibility(args)
    summary = format_summary(result, target_name="galactic smoke")

    assert result.coordinate_system == "galactic"
    assert len(result.table) == 5
    assert "Galactic l:" in summary


def test_duplicate_transformed_target_labels_are_unique():
    first = SkyCoord(l=0 * u.deg, b=90 * u.deg, frame="galactic")
    second = SkyCoord(l=90 * u.deg, b=90 * u.deg, frame="galactic")

    vis = compute_visibility(
        [first, second],
        report=False,
        interval_start_time=Time("2024-01-01T00:00:00"),
        interval_duration_days=2,
        interval_sampling_days=1,
    )
    vis.get_good_angles()

    labels = list(vis.df_results.index.levels[0])
    assert len(labels) == 2
    assert labels[0] != labels[1]
