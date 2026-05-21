"""Roman target-visibility calculator.

Adapted from
https://github.com/spacetelescope/roman-technical-information/blob/main/data/Observatory/Visibility/test_tgt_vis2.py
(original author Jeff Kruk, 2021-09-04).

Coordinate conventions
----------------------
Calculations are performed in ecliptic coordinates: ``+X`` to the vernal
equinox, ``+Z`` to the north ecliptic pole, ``+Y`` chosen to make a
right-handed system. The S/C local frame is obtained by rotating first about
``+Z`` to align ``X'`` with the LOS longitude, then about ``Y'`` to align
``X''`` with the LOS itself. Observatory axes: ``+X_obs`` along the
boresight, ``+Z_obs`` normal to the solar array, ``+Y_obs`` completes the
right-handed system. At nominal roll the Sun lies in the ``X_obs``-``Z_obs``
plane with ``+Z_obs`` as close to the Sun as the pitch allows.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import get_body
from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning


def normalize_start_time(start_time: Time | str | None) -> Time | None:
    """Coerce ``None`` / ISO string / ``Time`` into an astropy ``Time`` or ``None``."""
    if start_time is None:
        return None
    if isinstance(start_time, Time):
        return start_time
    if isinstance(start_time, str):
        if "T" in start_time:
            return Time(start_time, format="isot", scale="utc")
        return Time(f"{start_time}T00:00:00.0", format="isot", scale="utc")
    raise TypeError("start_time must be None, an astropy Time, or an ISO date string")


def sampled_times_for_interval(
    start_time: Time | str | None = None,
    duration_days: float = 365.0,
    sampling_days: float = 1.0,
) -> Time:
    """Build the array of sample times used by the visibility computation."""
    t_start = normalize_start_time(start_time)
    if t_start is None:
        t_start = Time(["2024-01-01T00:00:00.0"], format="isot", scale="utc")
    if sampling_days > duration_days:
        raise ValueError("sampling interval cannot exceed total duration")
    return t_start + np.arange(0.0, duration_days, sampling_days) * u.d


class VisibilityCalculator:
    """Compute Roman field-of-regard visibility for one or more fixed targets."""

    def __init__(
        self,
        targets_coordinates,
        fileout=None,
        report=False,
        interval_sampling_days=None,
        interval_start_time=None,
        interval_duration_days=None,
    ):
        if isinstance(targets_coordinates, list):
            targets = targets_coordinates
        else:
            targets = [targets_coordinates]
        self.targets_coordinates = [coords.icrs for coords in targets]
        self.target_labels = self._make_unique_target_labels(self.targets_coordinates)

        self.fileout = fileout
        self.report = report
        self.interval = {
            "sampling_days": interval_sampling_days,
            "start_time": interval_start_time,
            "duration_days": interval_duration_days,
        }
        self.sampled_times = sampled_times_for_interval(
            start_time=interval_start_time,
            duration_days=365.0 if interval_duration_days is None else interval_duration_days,
            sampling_days=1.0 if interval_sampling_days is None else interval_sampling_days,
        )
        self.sun_coord = get_body("Sun", self.sampled_times)
        self.min_sun_angle = (90.0 - 36.0) * u.deg
        self.max_sun_angle = (90.0 + 36.0) * u.deg
        self.df_results = self.initialize_dataframe()

    def compute_and_display(self):
        self.get_good_angles()
        self.get_roll_pa_sunang()
        if self.report:
            self.printout(self.get_preamble())
            self.printout(self.df_results.to_string())

    def initialize_dataframe(self):
        radec_string = self.target_labels
        time_string = [self.format_time(time) for time in self.sampled_times]
        index_levels = [radec_string, time_string]
        index_names = ["(RA, Dec)", "DOY"]
        multi_index = pd.MultiIndex.from_product(index_levels, names=index_names)
        column_names = [
            "Sun_RA",
            "Sun_Dec",
            "separation",
            "good_angles",
            "nominal_roll",
            "pa_obs_y",
            "pa_fpa_local_x",
            "pa_fpa_local_y",
            "sunang_x",
            "sunang_y",
            "sunang_z",
        ]

        # The level reindex below works around a pandas bug with sorting
        # categorical multi-index levels:
        # https://stackoverflow.com/questions/71837659
        return (pd.DataFrame(index=multi_index, columns=column_names)).reindex(
            radec_string, level=0
        )

    def _make_unique_target_labels(self, targets_coordinates):
        labels = [
            "({}, {})".format(
                coords.ra.to_string(u.hour),
                coords.dec.to_string(u.degree, alwayssign=True),
            )
            for coords in targets_coordinates
        ]
        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1

        seen = {}
        unique_labels = []
        for label in labels:
            seen[label] = seen.get(label, 0) + 1
            if counts[label] == 1:
                unique_labels.append(label)
            else:
                unique_labels.append(f"{label} #{seen[label]}")
        return unique_labels

    def format_time(self, time_object):
        """Convert an astropy ``Time`` to the ``YYYY-DOY.fraction`` index string."""
        datetime = time_object.datetime
        decimal_hours = datetime.hour + datetime.minute / 60.0 + datetime.second / 3600
        return (
            datetime.strftime("%Y")
            + "-"
            + datetime.strftime("%j")
            + "."
            + "{:7.5f}".format(decimal_hours / 24)[2:]
        )

    def printout(self, text):
        if self.fileout is not None:
            try:
                with open(self.fileout, "w") as ofile:
                    print(text, file=ofile)
            except Exception as e:
                print(f"Error writing to file: {e}", file=sys.stderr)
                print(text)
        else:
            print(text)

    def get_preamble(self):
        preamble = "Xang, Yang, Zang are angles of Sun vector in Observatory coordinate frame \n"
        preamble += "X is the boresight, valid angles are 54-126 degrees \n"
        preamble += "Z is normal to solar array, as close to zero as the pitch allows (<36 at nominal roll when pitch is OK) \n"
        preamble += "Y is perpendicular to X-Z plane; Yang should always be 90 at nominal roll \n"
        return preamble

    def get_good_angles(self):
        for i, target_coordinates in enumerate(self.targets_coordinates):
            target_label = self.target_labels[i]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AstropyWarning)
                sun_angle = self.sun_coord.separation(target_coordinates)
                good_angles = (sun_angle >= self.min_sun_angle) & (sun_angle <= self.max_sun_angle)
            self.df_results.loc[pd.IndexSlice[target_label, :], "good_angles"] = good_angles
            self.df_results.loc[pd.IndexSlice[target_label, :], "separation"] = sun_angle
            self.df_results.loc[pd.IndexSlice[target_label, :], "Sun_RA"] = self.sun_coord.ra
            self.df_results.loc[pd.IndexSlice[target_label, :], "Sun_Dec"] = self.sun_coord.dec

    def get_roll_pa_sunang(self):
        for i, target_coordinates in enumerate(self.targets_coordinates):
            target_label = self.target_labels[i]

            cos_ra_t = np.cos(target_coordinates.ra.radian)
            cos_dec_t = np.cos(target_coordinates.dec.radian)
            sin_ra_t = np.sin(target_coordinates.ra.radian)
            sin_dec_t = np.sin(target_coordinates.dec.radian)
            cos_ra_s = np.cos(self.sun_coord.ra.radian)
            cos_dec_s = np.cos(self.sun_coord.dec.radian)
            sin_ra_s = np.sin(self.sun_coord.ra.radian)
            sin_dec_s = np.sin(self.sun_coord.dec.radian)

            cc_s = cos_ra_s * cos_dec_s
            sc_s = sin_ra_s * cos_dec_s
            cs_s = cos_ra_s * sin_dec_s
            ss_s = sin_ra_s * sin_dec_s

            cc_t = cos_ra_t * cos_dec_t
            sc_t = sin_ra_t * cos_dec_t
            cs_t = cos_ra_t * sin_dec_t
            ss_t = sin_ra_t * sin_dec_t

            # Rotation matrix to convert a celestial vector to body-frame
            # coordinates for the S/C aligned to (RA=a, DEC=d, roll=r):
            #   /                  cosd cosa,                   cosd sina,       sind \
            #   | -cosr sina -sinr sind cosa,    cosr cosa - sinr sind sina,  sinr cosd |
            #   \  sinr sina -cosr sind cosa,   -sinr cosa - cosr sind sina,  cosr cosd /
            # Roll is found by rotating the Sun vector into the body frame and
            # minimizing the Z component.
            arg1 = sin_ra_t * cc_s - cos_ra_t * sc_s
            arg2 = cos_dec_t * sin_dec_s - sin_dec_t * (cos_ra_t * cc_s + sin_ra_t * sc_s)
            phi = np.arctan2(arg1, arg2)

            # Two solutions exist; confirm the projection onto +Z is positive.
            sinr = np.sin(phi)
            cosr = np.cos(phi)

            z_sun1 = (sinr * sin_ra_t - cosr * cs_t) * cc_s
            z_sun2 = (-sinr * cos_ra_t - cosr * ss_t) * sc_s
            z_sun3 = cosr * cos_dec_t * sin_dec_s
            z_sun = z_sun1 + z_sun2 + z_sun3
            sunang_z = np.arccos(z_sun) * 180.0 / np.pi

            phi = np.where(z_sun < 0.0, phi + np.pi, phi)

            nominal_roll = phi * 180.0 / np.pi
            nominal_roll = np.where(nominal_roll >= 0.0, nominal_roll, nominal_roll + 360.0)

            pa_obs_y = nominal_roll - 90.0
            pa_fpa_local_x = nominal_roll - 30.0
            pa_fpa_local_y = nominal_roll - 120.0
            pa_obs_y = np.where(pa_obs_y >= 0.0, pa_obs_y, pa_obs_y + 360.0)
            pa_fpa_local_x = np.where(pa_fpa_local_x >= 0.0, pa_fpa_local_x, pa_fpa_local_x + 360.0)
            pa_fpa_local_y = np.where(pa_fpa_local_y >= 0.0, pa_fpa_local_y, pa_fpa_local_y + 360.0)

            # Sanity-check projections onto the other body axes.
            x_sun = (cc_t * cc_s) + (sc_t * sc_s) + (sin_dec_t * sin_dec_s)
            sunang_x = np.arccos(x_sun) * 180.0 / np.pi

            sinr = np.sin(phi)
            cosr = np.cos(phi)
            y_sun1 = (-cosr * sin_ra_t - sinr * cs_t) * cc_s
            y_sun2 = (cosr * cos_ra_t - sinr * ss_t) * sc_s
            y_sun3 = sinr * cos_dec_t * sin_dec_s
            y_sun = y_sun1 + y_sun2 + y_sun3
            sunang_y = np.arccos(y_sun) * 180.0 / np.pi

            self.df_results.loc[pd.IndexSlice[target_label, :], "nominal_roll"] = nominal_roll
            self.df_results.loc[pd.IndexSlice[target_label, :], "pa_obs_y"] = pa_obs_y
            self.df_results.loc[pd.IndexSlice[target_label, :], "pa_fpa_local_x"] = pa_fpa_local_x
            self.df_results.loc[pd.IndexSlice[target_label, :], "pa_fpa_local_y"] = pa_fpa_local_y
            self.df_results.loc[pd.IndexSlice[target_label, :], "sunang_x"] = sunang_x
            self.df_results.loc[pd.IndexSlice[target_label, :], "sunang_y"] = sunang_y
            self.df_results.loc[pd.IndexSlice[target_label, :], "sunang_z"] = sunang_z
