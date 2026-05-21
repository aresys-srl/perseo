# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""IGRF: coordinates conversion functions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pymap3d as pm


def geodetic_to_geocentric(
    geodetic_lat: npt.NDArray[np.floating],
    height: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Convert geodetic coordinates (latitude, height) to geocentric (colatitude, radius).

    Parameters
    ----------
    geodetic_lat : npt.NDArray[np.floating]
        geodetic latitude [degrees], with shape (N,)
    height : npt.NDArray[np.floating]
        height above ellipsoid [km], with shape (N,)

    Returns
    -------
    npt.NDArray[np.floating]
        geocentric colatitude [degrees], with shape (N,)
    npt.NDArray[np.floating]
        geocentric radial distance [km], with shape (N,)
    """
    spherical_lat, _, radius_m = pm.geodetic2spherical(geodetic_lat, 0.0, height * 1000.0)
    return 90.0 - spherical_lat, radius_m / 1000.0


def rotate_field_to_geocentric(
    b_north: npt.NDArray[np.floating],
    b_up: npt.NDArray[np.floating],
    geodetic_lat: npt.NDArray[np.floating],
    theta: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Rotate magnetic field from geodetic (north/up) to geocentric (theta/radial) frame.

    Parameters
    ----------
    b_north : npt.NDArray[np.floating]
        magnetic field component in northward direction (geodetic), with shape (N,)
    b_up : npt.NDArray[np.floating]
        magnetic field component in upward direction (geodetic), with shape (N,)
    geodetic_lat : npt.NDArray[np.floating]
        geodetic latitude [degrees], with shape (N,)
    theta : npt.NDArray[np.floating]
        geocentric colatitude [degrees], with shape (N,)

    Returns
    -------
    npt.NDArray[np.floating]
        magnetic field component in the theta (southward) direction (geocentric)
    npt.NDArray[np.floating]
        magnetic field component in the radial (upward) direction (geocentric)
    """
    geodetic_lat_rad = np.radians(geodetic_lat)
    theta_rad = np.radians(theta)

    psi = np.sin(geodetic_lat_rad) * np.sin(theta_rad) - np.cos(geodetic_lat_rad) * np.cos(theta_rad)

    B_r = -np.sin(psi) * b_north + np.cos(psi) * b_up
    B_th = -np.cos(psi) * b_north - np.sin(psi) * b_up

    return B_th, B_r


def geocentric_to_geodetic(
    theta: npt.NDArray[np.floating],
    radius: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Convert geocentric coordinates (colatitude, radius) to geodetic (latitude, height).

    Parameters
    ----------
    theta : npt.NDArray[np.floating]
        geocentric colatitude [degrees], with shape (N,)
    radius : npt.NDArray[np.floating]
        geocentric radial distance [km], with shape (N,)

    Returns
    -------
    npt.NDArray[np.floating]
        geodetic latitude [degrees], with shape (N,)
    npt.NDArray[np.floating]
        height above ellipsoid [km], with shape (N,)
    """
    geodetic_lat, _, altitude_m = pm.spherical2geodetic(90.0 - theta, 0.0, radius * 1000.0)
    return geodetic_lat, altitude_m / 1000.0


def rotate_field_to_geodetic(
    b_theta: npt.NDArray[np.floating],
    b_r: npt.NDArray[np.floating],
    geodetic_lat: npt.NDArray[np.floating],
    theta: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Rotate magnetic field from geocentric (theta/radial) to geodetic (north/up) frame.

    Parameters
    ----------
    b_theta : npt.NDArray[np.floating]
        magnetic field component in the theta (southward) direction (geocentric), with shape (N,)
    b_r : npt.NDArray[np.floating]
        magnetic field component in the radial (upward) direction (geocentric), with shape (N,)
    geodetic_lat : npt.NDArray[np.floating]
        geodetic latitude [degrees], with shape (N,)
    theta : npt.NDArray[np.floating]
        geocentric colatitude [degrees], with shape (N,)

    Returns
    -------
    npt.NDArray[np.floating]
        magnetic field component in northward direction (geodetic), with shape (N,)
    npt.NDArray[np.floating]
        magnetic field component in upward direction (geodetic), with shape (N,)
    """
    geodetic_lat_rad = np.radians(geodetic_lat)
    theta_rad = np.radians(theta)

    psi = np.sin(geodetic_lat_rad) * np.sin(theta_rad) - np.cos(geodetic_lat_rad) * np.cos(theta_rad)
    b_north = -np.cos(psi) * b_theta - np.sin(psi) * b_r
    b_up = -np.sin(psi) * b_theta + np.cos(psi) * b_r

    return b_north, b_up
