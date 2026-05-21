"""Coarse all-sky visibility-fraction grid used by the notebook's Mollweide map."""

from __future__ import annotations

import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import get_body
from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning

from rtvt.coords import normalize_coordinate_system, skycoord_from_lon_lat
from rtvt.visibility import normalize_start_time, sampled_times_for_interval

_SKY_GRID_CACHE: dict = {}


def compute_all_sky_visibility_grid(
    ra_grid: np.ndarray | None = None,
    dec_grid: np.ndarray | None = None,
    grid_step_deg: float = 10,
    start_time: Time | str | None = None,
    duration_days: float = 365,
    sampling_days: float = 1,
    coordinate_system: str = "equatorial",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the visibility fraction over a coarse all-sky grid.

    If ``ra_grid`` / ``dec_grid`` are omitted, a ``grid_step_deg``-spaced grid
    is used. Returned arrays are ``(lon_grid_deg, lat_grid_deg, vis_frac_2d)``;
    ``vis_frac_2d`` has shape ``(len(lat_grid), len(lon_grid))``.
    """
    coordinate_system = normalize_coordinate_system(coordinate_system)

    if ra_grid is None:
        ra_grid = np.arange(0, 360, grid_step_deg)
    if dec_grid is None:
        dec_grid = np.arange(-90, 91, grid_step_deg)

    ra_mesh, dec_mesh = np.meshgrid(ra_grid, dec_grid)
    ra_flat = ra_mesh.ravel()
    dec_flat = dec_mesh.ravel()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        sky_points = skycoord_from_lon_lat(ra_flat, dec_flat, coordinate_system).icrs
        sampled_times = sampled_times_for_interval(
            start_time=start_time,
            duration_days=duration_days,
            sampling_days=sampling_days,
        )
        sun_coord = get_body("Sun", sampled_times)
        separation = sun_coord[:, np.newaxis].separation(sky_points[np.newaxis, :])

    good_angles = (separation >= 54 * u.deg) & (separation <= 126 * u.deg)
    vis_frac = np.mean(good_angles, axis=0)

    return np.asarray(ra_grid), np.asarray(dec_grid), vis_frac.reshape(ra_mesh.shape)


def get_cached_all_sky_visibility_grid(
    grid_step_deg: float = 10,
    start_time: Time | str | None = None,
    duration_days: float = 365,
    sampling_days: float = 1,
    coordinate_system: str = "equatorial",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process-local cache of `compute_all_sky_visibility_grid` keyed on its inputs."""
    coordinate_system = normalize_coordinate_system(coordinate_system)
    t0 = normalize_start_time(start_time)
    start_key = "default" if t0 is None else t0.isot
    key = (
        coordinate_system,
        float(grid_step_deg),
        start_key,
        float(duration_days),
        float(sampling_days),
    )
    if key not in _SKY_GRID_CACHE:
        _SKY_GRID_CACHE[key] = compute_all_sky_visibility_grid(
            grid_step_deg=grid_step_deg,
            start_time=t0,
            duration_days=duration_days,
            sampling_days=sampling_days,
            coordinate_system=coordinate_system,
        )
    return _SKY_GRID_CACHE[key]
