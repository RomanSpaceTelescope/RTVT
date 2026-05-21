"""Coordinate-system parsing and display helpers."""

from __future__ import annotations

from astropy import units as u
from astropy.coordinates import SkyCoord

_COORDINATE_SYSTEM_ALIASES = {
    "equatorial": "equatorial",
    "eq": "equatorial",
    "icrs": "equatorial",
    "radec": "equatorial",
    "ra/dec": "equatorial",
    "galactic": "galactic",
    "gal": "galactic",
    "lb": "galactic",
    "l/b": "galactic",
}


def normalize_coordinate_system(value: str) -> str:
    """Resolve an input coord-system label to ``"equatorial"`` or ``"galactic"``."""
    normalized = str(value).strip().lower()
    if normalized not in _COORDINATE_SYSTEM_ALIASES:
        raise ValueError("coordinate_system must be 'equatorial' or 'galactic'")
    return _COORDINATE_SYSTEM_ALIASES[normalized]


def coordinate_labels(coordinate_system: str) -> tuple[str, str, str]:
    """Return ``(lon_label, lat_label, frame_title)`` for the resolved coord system."""
    if normalize_coordinate_system(coordinate_system) == "galactic":
        return "l", "b", "Galactic"
    return "RA", "Dec", "Equatorial"


def skycoord_from_lon_lat(lon_deg: float, lat_deg: float, coordinate_system: str) -> SkyCoord:
    """Build a SkyCoord from decimal-degree lon/lat in the requested frame."""
    if normalize_coordinate_system(coordinate_system) == "galactic":
        return SkyCoord(l=lon_deg * u.deg, b=lat_deg * u.deg, frame="galactic")
    return SkyCoord(ra=lon_deg * u.deg, dec=lat_deg * u.deg, frame="icrs")


def display_lon_lat(coord: SkyCoord, coordinate_system: str) -> tuple[float, float]:
    """Project a SkyCoord to the display lon/lat (degrees) for the requested frame."""
    if normalize_coordinate_system(coordinate_system) == "galactic":
        gal = coord.galactic
        return float(gal.l.deg), float(gal.b.deg)
    icrs = coord.icrs
    return float(icrs.ra.deg), float(icrs.dec.deg)


def format_display_coordinate(
    lon_deg: float,
    lat_deg: float,
    coordinate_system: str,
    precision: int = 1,
) -> str:
    """Format a lon/lat pair as ``"RA=…, Dec=…"`` or ``"l=…, b=…"``."""
    lon_label, lat_label, _ = coordinate_labels(coordinate_system)
    return f"{lon_label}={lon_deg:.{precision}f}, {lat_label}={lat_deg:.{precision}f}"


def parse_target(ra: str, dec: str, coordinate_system: str = "equatorial") -> SkyCoord:
    """Parse CLI/GUI-style string inputs into a SkyCoord.

    Equatorial RA accepts decimal degrees or sexagesimal hour-angle strings
    (``"06:00:00"``, ``"6h0m0s"``); Dec is parsed as decimal degrees.
    Galactic inputs are decimal-degree ``l`` and ``b``.
    """
    coordinate_system = normalize_coordinate_system(coordinate_system)
    if coordinate_system == "galactic":
        return SkyCoord(l=float(ra) * u.deg, b=float(dec) * u.deg, frame="galactic")

    ra_unit = u.hourangle if _looks_sexagesimal_ra(ra) else u.deg
    return SkyCoord(ra, dec, unit=(ra_unit, u.deg), frame="icrs")


def _looks_sexagesimal_ra(value: str) -> bool:
    lowered = value.lower()
    return ":" in value or "h" in lowered or "m" in lowered or "s" in lowered
