"""Cumulative Gantt + multi-target Sun-target separation comparison figures."""

from __future__ import annotations

from typing import Sequence

from rtvt.analysis import find_observable_runs


TARGET_COLOR_PALETTE = [
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


def target_color(index: int) -> str:
    """Pick a deterministic color for the Nth selected target."""
    return TARGET_COLOR_PALETTE[index % len(TARGET_COLOR_PALETTE)]


def make_gantt_chart(selected_targets: Sequence[dict]):
    """Render a cumulative Gantt of observable windows; highlight the latest target.

    Each item in ``selected_targets`` must have keys ``dates`` (datetime sequence),
    ``good`` (boolean array, same length), ``color``, ``display_label``, and
    ``is_cvz`` (bool). Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter, MonthLocator, date2num

    with plt.ioff():
        fig, ax = plt.subplots(figsize=(13, 0.7 * len(selected_targets) + 1.8))
    latest_index = len(selected_targets) - 1

    for i, item in enumerate(selected_targets):
        if i == latest_index:
            ax.axhspan(i - 0.43, i + 0.43, color=item["color"], alpha=0.16, zorder=0)
        windows = find_observable_runs(item["good"])
        bars = []
        for start_idx, length in windows:
            d_start = item["dates"][start_idx]
            d_end = item["dates"][min(start_idx + length - 1, len(item["dates"]) - 1)]
            width = max(date2num(d_end) - date2num(d_start), 0.5)
            bars.append((date2num(d_start), width))

        if bars:
            ax.broken_barh(
                bars,
                (i - 0.35, 0.7),
                facecolors=item["color"],
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
        f"{i + 1}: {item['display_label']}" + (" [CVZ]" if item["is_cvz"] else "")
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
    fig.tight_layout()

    return fig


def make_separation_comparison(selected_targets: Sequence[dict]):
    """Render the multi-target Sun-target separation overlay; highlight the latest target.

    Each item must have keys ``dates``, ``separation`` (deg array, same length),
    ``color``, and ``display_label``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter, MonthLocator

    with plt.ioff():
        fig, ax = plt.subplots(figsize=(13, 5.2))
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
    fig.tight_layout()

    return fig
