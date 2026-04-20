# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Geometry - Coordinates Conversions
----------------------------------
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pyproj import Transformer

xyz2llh_transformer = Transformer.from_proj("epsg:4978", "epsg:4326")
llh2xyz_transformer = Transformer.from_proj("epsg:4326", "epsg:4978")


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
