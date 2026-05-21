"""Mollweide all-sky visibility-fraction figure used by the interactive notebook picker."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from rtvt.coords import coordinate_labels, display_lon_lat


def make_all_sky_mollweide(
    ra_grid_deg: np.ndarray,
    dec_grid_deg: np.ndarray,
    vis_frac_2d: np.ndarray,
    *,
    coordinate_system: str = "equatorial",
    test_targets: Iterable | None = None,
):
    """Render a Mollweide projection of the visibility fraction over the sky.

    Returns ``(fig, ax, pcm)``. The caller is responsible for displaying,
    closing, or wiring up click handlers — this function only builds the
    figure. ``test_targets`` are optional SkyCoord-like objects drawn as
    preview markers.
    """
    import matplotlib.pyplot as plt

    _, _, coordinate_title = coordinate_labels(coordinate_system)

    # Shift to a [-180, 180] longitude range so Mollweide centers on lon=0.
    ra_shifted = np.where(ra_grid_deg > 180, ra_grid_deg - 360, ra_grid_deg)
    sort_idx = np.argsort(ra_shifted)
    ra_sorted_deg = ra_shifted[sort_idx]
    vis_frac_sorted = vis_frac_2d[:, sort_idx]

    ra_plot = np.deg2rad(ra_sorted_deg)
    dec_plot = np.deg2rad(dec_grid_deg)
    ra_plot_mesh, dec_plot_mesh = np.meshgrid(ra_plot, dec_plot)

    with plt.ioff():
        fig, ax = plt.subplots(figsize=(12, 6), subplot_kw=dict(projection="mollweide"))
    if hasattr(fig.canvas, "toolbar_position"):
        fig.canvas.toolbar_position = "bottom"

    pcm = ax.pcolormesh(
        ra_plot_mesh,
        dec_plot_mesh,
        vis_frac_sorted,
        cmap="RdYlGn",
        shading="auto",
        vmin=0,
        vmax=1,
    )
    cbar = fig.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
    cbar.set_label("Visibility Fraction (of year)")

    if test_targets:
        for tgt in test_targets:
            tgt_lon, tgt_lat = display_lon_lat(tgt, coordinate_system)
            tgt_ra_plot = np.deg2rad(tgt_lon - 360 if tgt_lon > 180 else tgt_lon)
            tgt_dec_plot = np.deg2rad(tgt_lat)
            ax.plot(
                tgt_ra_plot,
                tgt_dec_plot,
                marker="x",
                linestyle="None",
                color="#222222",
                markersize=8,
                markeredgewidth=1.5,
                alpha=0.75,
            )

    ax.set_title(
        f"All-Sky Visibility Fraction ({coordinate_title}) -- Click to Select Targets",
        fontsize=12,
        pad=20,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, ax, pcm
