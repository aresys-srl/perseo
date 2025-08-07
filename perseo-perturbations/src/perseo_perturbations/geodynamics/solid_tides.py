# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Geodynamics Earth Crustal Displacement: Solid Tides submodule
-------------------------------------------------------------

Accounting for solid tides displacement using the IERS Conventions model.
"""

from __future__ import annotations

import datetime

import numpy as np
from arepytools.geometry.conversions import xyz2llh
from arepytools.io.metadata import PreciseDateTime

from perseo_perturbations.geodynamics.solid import solid_core

SECONDS_IN_MINUTES = 60
SECONDS_IN_HOUR = 3600
MINUTES_IN_DAY = 24 * 60


def _compute_displacement_unit_vectors(lat_geo_rad: np.ndarray, lon_rad: np.ndarray) -> np.ndarray:
    """Computing the unit vectors for each displacement direction, namely north, east and up.

    Parameters
    ----------
    lat_geo_rad : np.ndarray
        array of latitude coordinates geocentric in radians, shape (N,)
    lon_rad : np.ndarray
        array of longitude coordinates in radians, shape (N,)

    Returns
    -------
    np.ndarray
        array of shape (3, N, 3) where first dimension represent the displacement direction [north, east, up],
        second dimension is the size of the input arrays, third dimension is the unit vector coordinate [x, y, z]
    """

    # filling up an array of shape (3, N, 3):
    # first dimension represent the displacement unit vector, namely north [0], east [1] and up [2]
    # second dimension is the number of points in the input arrays
    # third dimension is the number of components of each unit vector (x, y, z)
    displacement_unit_vectors = np.zeros(shape=(3, lat_geo_rad.size, 3))

    # north unit vector
    displacement_unit_vectors[0] = np.array(
        [
            -np.sin(lat_geo_rad) * np.cos(lon_rad),
            -np.sin(lat_geo_rad) * np.sin(lon_rad),
            np.cos(lat_geo_rad),
        ]
    ).T

    # east unit vector
    displacement_unit_vectors[1] = np.array([-np.sin(lon_rad), np.cos(lon_rad), np.zeros_like(lon_rad)]).T

    # up unit vector
    displacement_unit_vectors[2] = np.array(
        [
            np.cos(lat_geo_rad) * np.cos(lon_rad),
            np.cos(lat_geo_rad) * np.sin(lon_rad),
            np.sin(lat_geo_rad),
        ]
    ).T

    return displacement_unit_vectors


def compute_solid_earth_tides(
    year: int, month: int, day_of_month: int, lat_deg: float, lon_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Solid Earth Tides wrapper of original Fortran code. This function uses the numpy.f2py conversion of fortran code
    library to extract solid earth tides displacements.

    Earth tides are estimated using an external tool (executable) called Solid, based on a fortran script.

    Solid Earth Tide (SET) displacement estimator class based on Solid fortran executable by Dennis Milbert.

    Program Solid is based on an edited version of the dehanttideinelMJD.f source code provided by Professor V. Dehant.
    This code is an implementation of the Solid Earth Tide computation found in section 7.1.2 of the
    IERS Conventions (2003) , IERS Technical Note No. 32

    Program inputs: date (year, month of the year, day of the month), latitude (deg), longitude (deg).
    Program output: .txt file with solid earth tide (body tide) components [north, east, up] for each minute of the
    input date.

    Solid is driven by a pair of routines that compute low-precision geocentric coordinates for the Moon and the Sun.
    These routines were coded from the equations in "Satellite Orbits: Models, Methods, Applications" by
    Montenbruck & Gill (2000), section 3.3.2, pp.70-73

    Solid does not contain ocean loading, atmospheric loading, or deformation due to polar motion.

    Ref: 'http://geodesyworld.github.io/SOFTS/solid.htm'

    Parameters
    ----------
    year : int
        year of the date at which the displacement must be estimated, between 1901 and 2099
    month : int
        month of the date at which the displacement must be estimated, between 1 and 12
    day_of_month : int
        day of the month of the date at which the displacement must be estimated, between 1 and 31
    lat_deg : float
        latitude where the displacement should be evaluated, in deg
    lon_deg : float
        longitude where the displacement should be evaluated, in deg

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        time axis corresponding to a whole day with 60 seconds steps,
        displacement array, of shape (N, 3), with columns being north, east and up components

    Raises
    ------
    ValueError
        if the input date is not valid
    """

    # check year, month and day validity
    assert isinstance(year, int), f"year {year} is not of type int"
    assert 1900 < year < 2100, f"year {year} exceeds boundaries [1901-2099]"
    assert isinstance(month, int), f"month {month} is not of type int"
    assert 1 <= month <= 12, f"{month} is not a valid month"
    assert isinstance(day_of_month, int), f"day {day_of_month} is not of type int"
    assert 1 <= day_of_month <= 31, f"{day_of_month} is not a valid day of the month"

    # check datetime validity
    try:
        datetime.datetime(year=year, month=month, day=day_of_month)
    except Exception as err:
        raise ValueError(f"{year}-{month}-{day_of_month} [yy-mm-dd] is not a valid date") from err

    # check validity of latitude and longitude
    assert -90 <= lat_deg <= 90
    assert -360 <= lon_deg <= 360

    # calling function from dll
    time_array, north_array, east_array, up_array = solid_core(
        lat_deg,
        lon_deg,
        year,
        month,
        day_of_month,
    )

    return time_array, np.stack([north_array, east_array, up_array], axis=1)


def compute_displacement(target_xyz_coords: np.ndarray, acquisition_time: PreciseDateTime) -> np.ndarray:
    """Estimate the input coordinates displacement due to earth tides based on acquisition time using the Fortran code
    wrapper.

    Parameters
    ----------
    target_xyz_coords : PreciseDateTime
        the input coordinates on scene, xyz format, shape Nx3
    acquisition_time : PreciseDateTime
        sensor acquisition time of the input coordinates on scene

    Returns
    -------
    np.ndarray
        updated coordinates, same input coordinate but with displacement added
    """

    # calculating the acquisition time in seconds relative to that day
    acq_time_sec = (
        acquisition_time.hour_of_day * SECONDS_IN_HOUR
        + acquisition_time.minute_of_hour * SECONDS_IN_MINUTES
        + acquisition_time.second_of_minute
    )

    # coordinates conversion: geodetic to geocentric
    # TODO: change this
    llh_coordinates = xyz2llh(target_xyz_coords.T).T
    lat_geocentric = np.arctan((1 - 1 / 298.25642) ** 2 * np.tan(llh_coordinates[:, 0]))

    # compute displacement unit vectors along north, east and up
    displacement_unit_vectors = _compute_displacement_unit_vectors(
        lat_geo_rad=lat_geocentric, lon_rad=llh_coordinates[:, 1]
    )

    # creating an empty array of shape (N, 3), columns are: north, east and up
    displacement_interp = np.zeros_like(target_xyz_coords)

    # compute displacement values for each point
    for point_id, _ in enumerate(target_xyz_coords):
        # converting lat and lon to deg and call the SOLID executable
        lat_deg = np.rad2deg(llh_coordinates[point_id, 0])
        lon_deg = np.rad2deg(llh_coordinates[point_id, 1])
        time_axis, displacements = compute_solid_earth_tides(
            year=acquisition_time.year,
            month=acquisition_time.month,
            day_of_month=acquisition_time.day_of_the_month,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
        )

        # evaluate the interpolated displacement at the right time (in seconds relative to midnight of that day)
        # north displacement interpolated value
        displacement_interp[point_id, 0] = np.interp(acq_time_sec, time_axis, displacements[:, 0])
        # east displacement interpolated value
        displacement_interp[point_id, 1] = np.interp(acq_time_sec, time_axis, displacements[:, 1])
        # up displacement interpolated value
        displacement_interp[point_id, 2] = np.interp(acq_time_sec, time_axis, displacements[:, 2])

    # compute displacement vectors
    north_total_displacement = displacement_interp[:, 0].reshape(-1, 1) * displacement_unit_vectors[0]
    east_total_displacement = displacement_interp[:, 1].reshape(-1, 1) * displacement_unit_vectors[1]
    up_total_displacement = displacement_interp[:, 2].reshape(-1, 1) * displacement_unit_vectors[2]

    return north_total_displacement + east_total_displacement + up_total_displacement
