"""Shared plumbing utilities used across the RTVT package."""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u


def prepare_matplotlib_cache() -> None:
    """Point matplotlib + XDG caches at a writable temp directory.

    Several install targets (locked-down kernels, the tkinter GUI on macOS,
    binder-style notebook environments) refuse to create the default
    ``~/.matplotlib`` cache. Routing both caches through ``tempfile.gettempdir()``
    keeps every entry point happy without needing per-frontend setup.
    """
    cache_root = Path(tempfile.gettempdir()) / "rtvt-matplotlib"
    xdg_cache_root = Path(tempfile.gettempdir()) / "rtvt-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    xdg_cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_root))


def figure_to_png_bytes(fig, dpi: int = 150, close: bool = False) -> bytes:
    """Render a matplotlib figure to PNG bytes, optionally closing the figure."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return buffer.getvalue()


def quantity_series_to_deg(series: pd.Series) -> np.ndarray:
    """Convert a pandas Series of astropy Quantities or floats to a float ndarray in degrees."""
    return np.array(
        [
            value.to_value(u.deg) if hasattr(value, "to_value") else float(value)
            for value in series.to_numpy()
        ],
        dtype=float,
    )
