# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Geomagnetic Model: International Geomagnetic Reference Field (IGRF) v14
-----------------------------------------------------------------------
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import numpy.typing as npt
import xarray as xr

from perseo_perturbations import IGRF_EARTH_RADIUS_KM, igrf_14_coeff_path
from perseo_perturbations.geomagnetic.core.conversions import (
    geocentric_to_geodetic,
    geodetic_to_geocentric,
    rotate_field_to_geodetic,
)
from perseo_perturbations.geomagnetic.core.shc_reader import read_shc
from perseo_perturbations.geomagnetic.core.utils import get_legendre


def _validate_degree_range(min_degree: int, max_degree: int) -> None:
    """Validate input degree range, raising error if not valid.

    Parameters
    ----------
    min_degree : int
        lowest spherical harmonic degree
    max_degree : int
        highest spherical harmonic degree

    Raises
    ------
    ValueError
        if min_degree > max_degree
    ValueError
        if min_degree < 1
    ValueError
        if max_degree > 13
    """
    if min_degree > max_degree:
        raise ValueError(
            f"Highest degree of expansion {max_degree} must be larger or equal to lowest degree {min_degree}."
        )
    if min_degree < 1:
        raise ValueError(f"Minimum degree of expansion {min_degree} must be larger or equal to 1.")
    if max_degree > 13:
        raise ValueError(f"Maximum degree of expansion {max_degree} must be smaller or equal to 13.")


def _prepare_coefficients(date: datetime) -> xr.Dataset:
    """Read the shipped coefficient file and interpolate to the requested date.

    Parameters
    ----------
    date : datetime
        a single date for evaluation

    Returns
    -------
    xr.Dataset
        interpolated coefficients with a single time entry, variables ``g``, ``h`` and coordinates ``n``, ``m`` along
        the ``coeff`` dimension.

    Raises
    ------
    ValueError
        if the date provided is outside the coefficient file date window
    """
    ds = read_shc(filename=igrf_14_coeff_path)
    d = np.datetime64(date).astype(ds.time.dtype)
    if d > ds.time.values[-1] or d < ds.time.values[0]:
        raise ValueError(f"Date provided not covered by coefficient file ({ds.time.values[0]} to {ds.time.values[-1]})")
    d = np.clip(d, ds.time.values[0], ds.time.values[-1])
    return ds.interp(time=d)


def get_geocentric_igrf(
    coordinates: npt.NDArray[np.floating],
    date: datetime,
    min_degree: int = 1,
    max_degree: int = 13,
) -> npt.NDArray[np.floating]:
    """Evaluate the IGRF magnetic field in geocentric coordinates.

    Parameters
    ----------
    coordinates : npt.NDArray[np.floating]
        geocentric coordinates with shape (N, 3) or (3,), where 3 corresponds to ``radius [km]``,
        ``colatitude [degrees]`` and ``longitude [degrees]``
    date : datetime
        the date of the evaluation
    min_degree : int, optional
        lowest spherical harmonic degree, by default 1
    max_degree : int, optional
        highest spherical harmonic degree, by default 13

    Returns
    -------
    npt.NDArray[np.floating]
        magnetic field components [nT] at the given coordinates with shape (N, 3) or (3,), where 3 corresponds to
        ``Br (radial)``, ``B_theta (southward)`` and ``B_phi (eastward)``
    """
    squeeze_output = coordinates.ndim == 1
    coordinates = np.atleast_2d(np.asarray(coordinates))

    r, theta, phi = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
    _validate_degree_range(min_degree, max_degree)

    shape = coordinates.shape[:-1]
    r = r.reshape((-1, 1))
    theta = theta.reshape((-1, 1))
    phi = phi.reshape((-1, 1))

    ds = _prepare_coefficients(date)
    m = ds["m"].values.reshape((1, -1))
    keys = list(zip(ds["n"].values, ds["m"].values, strict=True))

    legendre_functions, delta_p = get_legendre(theta, keys)

    phi_rad = np.radians(phi)
    cos_m_phi = np.cos(phi_rad * m)
    sin_m_phi = np.sin(phi_rad * m)

    nn, mm = np.tile(ds["n"].values.reshape((1, -1)), 2), np.tile(m, 2)
    n_map = ((nn >= min_degree) & (nn <= max_degree)).astype(int)

    gh = np.hstack((ds["g"].values, ds["h"].values))
    rho_n1 = (IGRF_EARTH_RADIUS_KM / r) ** (nn + 1)
    rho_n2 = (IGRF_EARTH_RADIUS_KM / r) ** (nn + 2)

    g_field = n_map * rho_n2 * (nn + 1) * np.hstack((legendre_functions * cos_m_phi, legendre_functions * sin_m_phi))
    b_r = g_field.dot(gh.T).T.reshape(shape)

    g_field = -n_map * rho_n1 * np.hstack((delta_p * cos_m_phi, delta_p * sin_m_phi)) * IGRF_EARTH_RADIUS_KM / r
    b_theta = g_field.dot(gh.T).T.reshape(shape)

    sin_theta = np.sin(np.radians(theta))
    sin_theta = np.where(np.abs(sin_theta) < 1e-12, np.nan, sin_theta)

    g_field = (
        -n_map
        * rho_n1
        * mm
        * np.hstack((-legendre_functions * sin_m_phi, legendre_functions * cos_m_phi))
        * IGRF_EARTH_RADIUS_KM
        / r
        / sin_theta
    )
    b_phi = g_field.dot(gh.T).T.reshape(shape)
    b_phi = np.nan_to_num(b_phi, nan=0.0, posinf=0.0, neginf=0.0)

    result = np.stack([b_r, b_theta, b_phi], axis=-1)
    return result.squeeze(0) if squeeze_output else result


def get_geodetic_igrf(
    coordinates: npt.NDArray[np.floating],
    date: datetime,
    min_degree: int = 1,
    max_degree: int = 13,
) -> npt.NDArray[np.floating]:
    """Evaluate the IGRF magnetic field in geodetic coordinates.

    Geodetic coordinates account for the ellipsoidal shape of the Earth using WGS84 model. The northward component is
    tangential to the ellipsoid, and the upward component is perpendicular to it.

    Parameters
    ----------
    coordinates : npt.NDArray[np.floating]
        geodetic coordinates with shape (N, 3) or (3,), where 3 corresponds to ``longitude [degrees]``,
        ``latitude [degrees]`` and ``height [km above WGS84 ellipsoid]``
    date : datetime
        the date of the evaluation
    min_degree : int, optional
        lowest spherical harmonic degree, by default 1
    max_degree : int, optional
        highest spherical harmonic degree, by default 13

    Returns
    -------
    npt.NDArray[np.floating]
        magnetic field components [nT] at the given coordinates with shape (N, 3) or (3,), where 3 corresponds to
        ``Be (eastward)``, ``Bn (northward, tangential to the ellipsoid)`` and ``Bu (upward, perpendicular to the
        ellipsoid)``
    """
    squeeze_output = coordinates.ndim == 1
    coordinates = np.atleast_2d(np.asarray(coordinates))

    lon, lat, h = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
    shape = coordinates.shape[:-1]

    theta, r = geodetic_to_geocentric(lat.ravel(), h.ravel())

    result_gc = get_geocentric_igrf(
        np.stack([r, theta, lon.ravel()], axis=-1),
        date,
        min_degree=min_degree,
        max_degree=max_degree,
    )

    b_r = result_gc[..., 0].ravel()
    b_theta = result_gc[..., 1].ravel()
    b_phi = result_gc[..., 2].ravel()

    geodetic_lat, _ = geocentric_to_geodetic(theta, r)
    b_north, b_up = rotate_field_to_geodetic(b_theta, b_r, geodetic_lat, theta)

    result = np.stack([b_phi.reshape(shape), b_north.reshape(shape), b_up.reshape(shape)], axis=-1)
    return result.squeeze(0) if squeeze_output else result


def get_geocentric_igrf_potential(
    coordinates: npt.NDArray[np.floating],
    date: datetime,
    min_degree: int = 1,
    max_degree: int = 13,
) -> float | npt.NDArray[np.floating]:
    """Evaluate the IGRF magnetic potential in geocentric coordinates.

    The magnetic potential $V$ satisfies $\\mathbf{B} = -\\nabla V$ and is expanded in spherical harmonics
    up to degree 13.

    Parameters
    ----------
    coordinates : npt.NDArray[np.floating]
        geocentric coordinates with shape (N, 3) or (3,), where 3 corresponds to ``radius [km]``,
        ``colatitude [degrees]`` and ``longitude [degrees]``
    date : datetime
        the date of the evaluation
    min_degree : int, optional
        lowest spherical harmonic degree, by default 1
    max_degree : int, optional
        highest spherical harmonic degree, by default 13

    Returns
    -------
    float | npt.NDArray[np.floating]
        magnetic potential [nT km], with shape (N,) or scalar
    """
    squeeze_output = coordinates.ndim == 1
    coordinates = np.atleast_2d(np.asarray(coordinates))

    r, theta, phi = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
    _validate_degree_range(min_degree, max_degree)

    shape = coordinates.shape[:-1]
    r = r.reshape((-1, 1))
    theta = theta.reshape((-1, 1))
    phi = phi.reshape((-1, 1))

    ds = _prepare_coefficients(date)

    m = ds["m"].values.reshape((1, -1))
    keys = list(zip(ds["n"].values, ds["m"].values, strict=True))

    legendre_functions, _ = get_legendre(theta, keys)

    phi_rad = np.radians(phi)
    cos_m_phi = np.cos(phi_rad * m)
    sin_m_phi = np.sin(phi_rad * m)

    nn = np.tile(ds["n"].values.reshape((1, -1)), 2)
    n_map = ((nn >= min_degree) & (nn <= max_degree)).astype(int)

    field = (
        n_map
        * IGRF_EARTH_RADIUS_KM
        * (IGRF_EARTH_RADIUS_KM / r) ** (nn + 1)
        * np.hstack((legendre_functions * cos_m_phi, legendre_functions * sin_m_phi))
    )
    result = field.dot(np.hstack((ds["g"].values, ds["h"].values)).T).T.reshape(shape)

    return result.item() if squeeze_output else result
