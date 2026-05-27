# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
This module provides vectorized coordinate transformations for terrestrial and celestial
reference frames commonly used in SAR and orbital geometry computations. It leverages `pyproj`
for geodetic conversions and `Astropy` for precise celestial frame transformations.

### Coordinate Systems Supported

- **ECEF (EPSG:4978)**: Earth-Centered Earth-Fixed cartesian coordinates (X, Y, Z) in meters
- **LLH (EPSG:4326)**: Geodetic coordinates, Latitude, Longitude in radians/degrees, Height in meters
- **UTM (EPSG:326xx/327xx)**: Universal Transverse Mercator coordinates (Easting, Northing, Height) in meters
- **ECI (GCRS)**: Earth-Centered Inertial (Geocentric Celestial Reference System)

All functions support both single point (shape (3,)) and batch operations on arrays
of coordinates (shape (N, 3)). The ECEF<->ECI transformations properly account for
Earth rotation requiring a precise UTC timestamp via `PreciseDateTime` and velocities.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from astropy import units
from astropy.coordinates import GCRS, ITRS, CartesianDifferential, CartesianRepresentation
from astropy.time import Time
from pyproj import CRS, Transformer

from perseo_core.timing.precise_datetime import PreciseDateTime

LLH_CRS = CRS.from_epsg(4326)
ECEF_CRS = CRS.from_epsg(4978)
UTM_EPSG_NORTH_BASE = 32600
UTM_EPSG_SOUTH_BASE = 32700
xyz2llh_transformer = Transformer.from_crs(ECEF_CRS, LLH_CRS)
llh2xyz_transformer = Transformer.from_crs(LLH_CRS, ECEF_CRS)


def _get_utm_epsg_code(zone: str) -> int:
    """Convert UTM zone string in format ZZH (e.g., '33N', '33S') to EPSG codes for transformations.

    Parameters
    ----------
    zone : str
        UTM zone in format 'ZZH' where ZZ is zone number (1-60) and H is hemisphere ('N' or 'S')

    Returns
    -------
    int
        UTM epsg code, 326XX for northern hemisphere, 327XX for southern

    Raises
    ------
    ValueError
        if zone format is invalid or zone number is out of range
    """
    if len(zone) not in (2, 3):
        raise ValueError(f"Zone must be string format like '1N' or '33S', not {zone}")

    hemisphere = zone[-1].upper()
    try:
        zone_num = int(zone[:-1])
    except ValueError as e:
        raise ValueError(f"Invalid zone format '{zone}'. Expected format like '33N' or '33S'") from e

    if zone_num < 1 or zone_num > 60:
        raise ValueError(f"UTM zone number must be between 1 and 60, not {zone_num}")

    if hemisphere not in ("N", "S"):
        raise ValueError(f"Hemisphere must be 'N' or 'S', got '{hemisphere}'")

    return UTM_EPSG_NORTH_BASE + zone_num if hemisphere == "N" else UTM_EPSG_SOUTH_BASE + zone_num


def xyz2llh(coordinates: npt.NDArray[np.floating], radians: bool = True) -> npt.NDArray[np.floating]:
    """Conversion from XYZ ECEF coordinates (epsg:4978) in [m] to LLH (latitude [rad/deg], longitude [rad/deg],
    height [m]) geodetic coordinates (epsg:4326). Output latitude and longitude may be returned in [deg] if *radians*
    input flag is set to False.

    Parameters
    ----------
    coordinates : npt.NDArray[np.floating]
        XYZ EEF coordinates (epsg:4978), with shape (3,), (1, 3) or (N, 3), with 3 being X, Y and Z in meters
    radians : bool, optional
        if output latitude and longitude must be expressed in radians, otherwise they are provided in deg,
        by default True

    Returns
    -------
    npt.NDArray[np.floating]
        LLH geodetic coordinates (epsg:4326), with shape (3,), (1, 3) or (N, 3), with 3 being Lat [rad/deg],
        Lon [rad/deg] and H [m]
    """
    coordinates = np.atleast_2d(coordinates)
    return np.c_[
        xyz2llh_transformer.transform(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], radians=radians)
    ].squeeze()


def llh2xyz(coordinates: npt.NDArray[np.floating], radians: bool = True) -> npt.NDArray[np.floating]:
    """Conversion from LLH geodetic coordinates (epsg:4326) in [rad/deg, rad/deg, m] to XYZ ECEF coordinates (epsg:4978)
    XYZ in [m]. Input latitude and longitude may be provided in [deg] if radians input flag is set to False.

    Parameters
    ----------
    coordinates : npt.NDArray[np.floating]
        llh geodetic coordinates (epsg:4326), with shape (3,), (1, 3) or (N, 3), with 3 being Lat [rad/deg],
        Lon [rad/deg] and H [m]
    radians : bool, optional
        if input latitude and longitude are expressed in radians, otherwise they can be provided in deg,
        by default True

    Returns
    -------
    npt.NDArray[np.floating]
        XYZ EEF coordinates (epsg:4978), with shape (3,), (1, 3) or (N, 3), with 3 being X, Y and Z in meters
    """
    coordinates = np.atleast_2d(coordinates)
    return np.c_[
        llh2xyz_transformer.transform(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], radians=radians)
    ].squeeze()


def ecef2eci(
    positions: npt.NDArray[np.floating],
    velocities: npt.NDArray[np.floating],
    times: PreciseDateTime | npt.NDArray,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Conversion from XYZ ECEF (epsg:4978 ITRS) positions [m] and velocities [m/s] to XYZ ECI (GCRS) coordinates
    keeping the same unit of measurement.

    Parameters
    ----------
    positions : npt.NDArray[np.floating]
        positions in ECEF coordinates expressed in [m], with shape (3,) or (N, 3)
    velocities : npt.NDArray[np.floating]
        velocities in ECEF coordinates expressed in [m/s], with shape (3,) or (N, 3)
    times : times: PreciseDateTime | npt.NDArray
        observation times [UTC] associated to the input positions and velocities, scalar or with shape (N,)

    Returns
    -------
    npt.NDArray[np.floating]
        XYZ ECI position coordinates expressed in [m], with shape (3,) or (N, 3)
    npt.NDArray[np.floating]
        XYZ ECI velocities coordinates expressed in [m/s], with shape (3,) or (N, 3)
    """

    is_scalar = positions.ndim == 1
    times = np.atleast_1d(times)
    observation_times = Time([t.isoformat() for t in times], scale="utc")

    # Build representation with velocity
    cartesian_representation_3d = CartesianRepresentation(
        x=positions[..., 0] * units.m,
        y=positions[..., 1] * units.m,
        z=positions[..., 2] * units.m,
        differentials=CartesianDifferential(
            d_x=velocities[..., 0] * units.m / units.s,
            d_y=velocities[..., 1] * units.m / units.s,
            d_z=velocities[..., 2] * units.m / units.s,
        ),
    )

    # coordinates in ITRS frame
    ecef = ITRS(cartesian_representation_3d, obstime=observation_times)

    # transform to ECI (GCRS)
    eci = ecef.transform_to(GCRS(obstime=observation_times))

    positions_eci = np.moveaxis(eci.cartesian.xyz.to_value("m"), 0, -1)
    velocities_eci = np.moveaxis(eci.cartesian.differentials["s"].d_xyz.to_value("m/s"), 0, -1)
    if is_scalar:
        return positions_eci[0], velocities_eci[0]
    return positions_eci, velocities_eci


def eci2ecef(
    positions: npt.NDArray[np.floating],
    velocities: npt.NDArray[np.floating],
    times: PreciseDateTime | npt.NDArray,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Conversion from XYZ ECI (GCRS) positions [m] and velocities [m/s] to XYZ ECEF (epsg:4978 ITRS) coordinates
    keeping the same unit of measurement.

    Parameters
    ----------
    positions : npt.NDArray[np.floating]
        positions in ECI coordinates expressed in [m], with shape (3,) or (N, 3)
    velocities : npt.NDArray[np.floating]
        velocities in ECI coordinates expressed in [m/s], with shape (3,) or (N, 3)
    times : times: PreciseDateTime | npt.NDArray
        observation times [UTC] associated to the input positions and velocities, scalar or with shape (N,)

    Returns
    -------
    npt.NDArray[np.floating]
        XYZ ECI position coordinates expressed in [m], with shape (3,) or (N, 3)
    npt.NDArray[np.floating]
        XYZ ECI velocities coordinates expressed in [m/s], with shape (3,) or (N, 3)
    """

    is_scalar = positions.ndim == 1
    times = np.atleast_1d(times)
    observation_times = Time([t.isoformat() for t in times], scale="utc")

    # build representation with velocity
    cartesian_representation_3d = CartesianRepresentation(
        x=positions[..., 0] * units.m,
        y=positions[..., 1] * units.m,
        z=positions[..., 2] * units.m,
        differentials=CartesianDifferential(
            d_x=velocities[..., 0] * units.m / units.s,
            d_y=velocities[..., 1] * units.m / units.s,
            d_z=velocities[..., 2] * units.m / units.s,
        ),
    )

    # coordinates in GCRS frame
    eci = GCRS(cartesian_representation_3d, obstime=observation_times)

    # transform to ECEF (ITRS)
    ecef = eci.transform_to(ITRS(obstime=observation_times))

    positions_ecef = np.moveaxis(ecef.cartesian.xyz.to_value("m"), 0, -1)
    velocities_ecef = np.moveaxis(ecef.cartesian.differentials["s"].d_xyz.to_value("m/s"), 0, -1)
    if is_scalar:
        return positions_ecef[0], velocities_ecef[0]
    return positions_ecef, velocities_ecef


def utm2llh(coordinates: npt.NDArray[np.floating], zone: str, radians: bool = True) -> npt.NDArray[np.floating]:
    """Conversion from UTM (Easting[m], Northing [m], Height [m]) coordinates (epsg:326xx or 327xx) to
    LLH (latitude [rad/deg], longitude [rad/deg], height [m]) geodetic coordinates (epsg:4326).

    Parameters
    ----------
    coordinates : npt.NDArray[np.floating]
        UTM coordinates with shape (3,), (1, 3) or (N, 3), with 3 being Easting, Northing, and Height in meters
    zone : str
        UTM zone in format 'ZZH' where ZZ is zone number (1-60) and H is hemisphere ('N' for northern, 'S' for southern)
        Example: '33N', '33S', '1N', '60S'
    radians : bool, optional
        if output latitude and longitude must be expressed in radians, otherwise they are provided in deg,
        by default True

    Returns
    -------
    npt.NDArray[np.floating]
        LLH geodetic coordinates (epsg:4326), with shape (3,), (1, 3) or (N, 3), with 3 being Lat [rad/deg],
        Lon [rad/deg] and H [m]

    Raises
    ------
    ValueError
        if zone format is invalid or zone number is out of range (1-60)
    """
    utm_epsg = _get_utm_epsg_code(zone)
    transformer = Transformer.from_crs(CRS.from_epsg(utm_epsg), LLH_CRS)

    coordinates = np.atleast_2d(coordinates)
    return np.c_[
        transformer.transform(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], radians=radians)
    ].squeeze()


def llh2utm(coordinates: npt.NDArray[np.floating], zone: str, radians: bool = True) -> npt.NDArray[np.floating]:
    """Conversion from LLH (latitude [rad/deg], longitude [rad/deg], height [m]) geodetic coordinates (epsg:4326) to
    UTM (Easting[m], Northing [m], Height [m]) coordinates (epsg:326xx or 327xx).

    Parameters
    ----------
    coordinates : npt.NDArray[np.floating]
        LLH geodetic coordinates (epsg:4326), with shape (3,), (1, 3) or (N, 3), with 3 being Lat [rad/deg],
        Lon [rad/deg] and H [m]
    zone : str
        UTM zone in format 'ZZH' where ZZ is zone number (1-60) and H is hemisphere ('N' for northern, 'S' for southern)
        Example: '33N', '33S', '1N', '60S'
    radians : bool, optional
        if input latitude and longitude are expressed in radians, otherwise they can be provided in deg,
        by default True

    Returns
    -------
    npt.NDArray[np.floating]
        UTM coordinates, with shape (3,), (1, 3) or (N, 3), with 3 being Easting [m], Northing [m], and Height [m]

    Raises
    ------
    ValueError
        if zone format is invalid or zone number is out of range (1-60)
    """
    utm_epsg = _get_utm_epsg_code(zone)
    transformer = Transformer.from_crs(LLH_CRS, CRS.from_epsg(utm_epsg))

    coordinates = np.atleast_2d(coordinates)
    return np.c_[
        transformer.transform(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], radians=radians)
    ].squeeze()
