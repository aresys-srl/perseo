# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""IGRF: utilities functions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def get_legendre(
    theta: npt.NDArray[np.floating], keys: list[tuple[int, int]]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Calculate Schmidt semi-normalized associated Legendre functions.

    Calculations based on recursive algorithm referring to "Spacecraft attitude determination and control" by
    Wertz, James R., 1978, https://doi.org/10.1007/978-94-009-9907-7

    Parameters
    ----------
    theta : npt.NDArray[np.floating]
        colatitudes in degrees, with shape (N,)
    keys : list[tuple[int, int]]
        list of spherical harmonic degree[0] and order[1], tuple (n, m) for each term in the expansion

    Returns
    -------
    npt.NDArray[np.floating]
        Legendre functions (P), with shape (N, len(keys))
    npt.NDArray[np.floating]
        dP/d_theta, with shape (N, len(keys))
    """

    # get maximum N and maximum M:
    n, m = np.array([k for k in keys]).T
    max_degree, max_order = np.max(n), np.max(m)

    theta = theta.flatten()[:, np.newaxis]
    theta_rad = np.radians(theta)

    P = {}
    dP = {}
    sin_theta = np.sin(theta_rad)
    cos_theta = np.cos(theta_rad)

    # Initialize Schmidt normalization
    S = {}
    S[0, 0] = 1.0

    # initialize the functions:
    for n in range(max_degree + 1):
        for m in range(max_degree + 1):
            P[n, m] = np.zeros_like(theta, dtype=np.float64)
            dP[n, m] = np.zeros_like(theta, dtype=np.float64)

    P[0, 0] = np.ones_like(theta, dtype=np.float64)
    for n in range(1, max_degree + 1):
        for m in range(0, min([n + 1, max_order + 1])):
            # do the legendre polynomials and derivatives
            if n == m:
                P[n, n] = sin_theta * P[n - 1, m - 1]
                dP[n, n] = sin_theta * dP[n - 1, m - 1] + cos_theta * P[n - 1, n - 1]
            else:
                if n == 1:
                    Knm = 0.0
                    P[n, m] = cos_theta * P[n - 1, m]
                    dP[n, m] = cos_theta * dP[n - 1, m] - sin_theta * P[n - 1, m]

                elif n > 1:
                    Knm = ((n - 1) ** 2 - m**2) / ((2 * n - 1) * (2 * n - 3))
                    P[n, m] = cos_theta * P[n - 1, m] - Knm * P[n - 2, m]
                    dP[n, m] = cos_theta * dP[n - 1, m] - sin_theta * P[n - 1, m] - Knm * dP[n - 2, m]

            # compute Schmidt normalization
            if m == 0:
                S[n, 0] = S[n - 1, 0] * (2.0 * n - 1) / n
            else:
                delta = 2.0 if m == 1 else 1.0
                S[n, m] = S[n, m - 1] * np.sqrt((n - m + 1) * delta / (n + m))

    # now apply Schmidt normalization
    for n in range(1, max_degree + 1):
        for m in range(0, min([n + 1, max_order + 1])):
            P[n, m] *= S[n, m]
            dP[n, m] *= S[n, m]

    return np.hstack(tuple(P[key] for key in keys)), np.hstack(tuple(dP[key] for key in keys))


def get_magnetic_inclination(
    geodetic_magnetic_field: npt.NDArray[np.floating],
    degrees: bool = True,
) -> npt.NDArray[np.floating]:
    """Compute the magnetic inclination angles of the IGRF from magnetic field components in geodetic coordinates.

    ``Inclination angle`` is defined as the angle between the magnetic field vector and the horizontal plane:

    $$
    I = \\arctan \\left(\\frac{-B_u}{\\sqrt{B_e^2 + B_n^2}}\\right)
    $$

    Parameters
    ----------
    geodetic_magnetic_field : npt.NDArray[np.floating]
        magnetic field components [nT] with shape (N, 3) or (3,), where 3 corresponds to ``Be (eastward)``,
        ``Bn (northward, tangential to the ellipsoid)`` and ``Bu (upward, perpendicular to the
        ellipsoid)``
    degrees : bool, optional
        flag to return the angles in degrees, by default True

    Returns
    -------
    npt.NDArray[np.floating]
        inclination angle of the IGRF magnetic vector [degrees] or [radians], with shape (N,) or scalar
    """
    # compute H, the horizontal component of B
    horizontal_component = np.sqrt(geodetic_magnetic_field[..., 0] ** 2 + geodetic_magnetic_field[..., 1] ** 2)

    mask = horizontal_component != 0

    # compute inclination
    safe_division = np.divide(
        -geodetic_magnetic_field[..., 2],
        horizontal_component,
        out=np.zeros_like(horizontal_component, dtype=float),
        where=mask,
    )

    inclination = np.where(
        mask,
        np.arctan(safe_division),
        -np.sign(geodetic_magnetic_field[..., 2]) * np.pi / 2,
    )

    return np.degrees(inclination) if degrees else inclination


def get_magnetic_declination(
    geodetic_magnetic_field: npt.NDArray[np.floating],
    degrees: bool = True,
) -> npt.NDArray[np.floating]:
    """Compute the magnetic declination angles of the IGRF from magnetic field components in geodetic coordinates.

    ``Declination angle`` is defined as the azimuth of the projection of the magnetic field vector onto the horizontal
    plane (starting from the northing direction, positive to the east and negative to the west):

    $$
    D = \\arctan2\\left(B_e, B_n\\right)
    $$

    Parameters
    ----------
    geodetic_magnetic_field : npt.NDArray[np.floating]
        magnetic field components [nT] with shape (N, 3) or (3,), where 3 corresponds to ``Be (eastward)``,
        ``Bn (northward, tangential to the ellipsoid)`` and ``Bu (upward, perpendicular to the
        ellipsoid)``
    degrees : bool, optional
        flag to return the angles in degrees, by default True

    Returns
    -------
    npt.NDArray[np.floating]
        declination angle of the IGRF magnetic vector [degrees] or [radians], with shape (N,) or scalar
    """
    declination = np.arctan2(geodetic_magnetic_field[..., 0], geodetic_magnetic_field[..., 1])

    return np.degrees(declination) if degrees else declination
