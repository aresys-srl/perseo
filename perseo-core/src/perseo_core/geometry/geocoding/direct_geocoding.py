# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Direct geocoding main functionalities"""

from __future__ import annotations

import numpy as np
from scipy.constants import speed_of_light

from perseo_core.geometry.coords_conversions import llh2xyz, xyz2llh
from perseo_core.geometry.geocoding.direct_geocoding_core import (
    direct_geocoding_bistatic_core,
    direct_geocoding_monostatic_core,
)
from perseo_core.geometry.utilities import ReferenceFrameLike
from perseo_core.geometry.utilities.ellipsoid import (
    compute_line_ellipsoid_intersections,
    create_inflated_WGS84_ellipsoid,
)
from perseo_core.geometry.utilities.reference_frames import (
    compute_sensor_local_axis,
)
from perseo_core.geometry.utilities.rotations import euler_angles_to_rotation
from perseo_core.models.enums import SensorLookDirection
from perseo_core.models.types import CoordinatesArrayType, FloatArrayType


def direct_geocoding_with_looking_direction(
    sensor_positions: CoordinatesArrayType,
    looking_direction: CoordinatesArrayType,
    geodetic_altitude: float = 0.0,
) -> CoordinatesArrayType:
    """Computes the ground points seen with the given looking directions.

    The looking direction defines a line: its norm and sign do not matter.

    Based on :meth:`perseo.geometry.utilities.ellipsoid.compute_line_ellipsoid_intersections`

    Parameters
    ----------
    sensor_positions : CoordinatesArrayType
        sensor positions, with shape (3,) or (N, 3)
    looking_direction : CoordinatesArrayType
        vectors aligned with a looking direction, with shape (3,) or (N, 3)
    geodetic_altitude : float, optional
        altitude over the WGS84 ellipsoid, by default 0.0

    Returns
    -------
    CoordinatesArrayType
        ground points, with shape (3,) or (N, 3), np.nan is a place-holder in case of impossible geocoding
    """
    inflated_ellipsoid = create_inflated_WGS84_ellipsoid(geodetic_altitude)

    intersections = compute_line_ellipsoid_intersections(
        line_directions=looking_direction,
        line_origins=sensor_positions,
        ellipsoid=inflated_ellipsoid,
    )

    points = np.empty(np.broadcast_shapes(np.shape(looking_direction), np.shape(sensor_positions)))

    if points.ndim == 1:
        intersections = (intersections,)

    for intersections_pair, point in zip(intersections, points.reshape((-1, 3)), strict=False):
        point[:] = intersections_pair[0] if len(intersections_pair) > 0 else np.nan

    return points


def direct_geocoding_with_look_angles(
    sensor_positions: CoordinatesArrayType,
    sensor_velocities: CoordinatesArrayType,
    reference_frame: ReferenceFrameLike,
    look_angles: float | FloatArrayType,
    geodetic_altitude: float = 0.0,
) -> CoordinatesArrayType:
    """Computes the points at a given altitude over WGS84 ellipsoid seen with the given look angles.

    Based on :meth:`perseo.geometry.geocoding.direct_geocoding.direct_geocoding_with_looking_direction`

    Parameters
    ----------
    sensor_positions : CoordinatesArrayType
        sensor positions, with shape (3,) or (N, 3)
    sensor_velocities : CoordinatesArrayType
        sensor velocity, with shape (3,) or (N, 3)
    reference_frame : ReferenceFrameLike
        which reference frame to assume
    look_angles : float | FloatArrayType
        look angles in radians, scalar or (N,)
    geodetic_altitude : float, optional
        altitude over the WGS84 ellipsoid, by default 0.0

    Returns
    -------
    CoordinatesArrayType
        ground points, with shape (3,) or (N, 3), np.nan is a place-holder in case of impossible geocoding
    """
    local_axis = compute_sensor_local_axis(
        sensor_positions=sensor_positions, sensor_velocities=sensor_velocities, reference_frame=reference_frame
    )

    look_angles = np.atleast_1d(look_angles)
    assert look_angles.ndim == 1
    rotation = euler_angles_to_rotation(
        euler_angles_rad=np.stack(
            [np.zeros_like(look_angles), np.zeros_like(look_angles), -np.asarray(look_angles)], axis=-1
        ),
        order="YPR",
    )

    pointing = (local_axis * rotation).as_matrix().squeeze()[..., 2]

    return direct_geocoding_with_looking_direction(
        sensor_positions=sensor_positions, looking_direction=pointing, geodetic_altitude=geodetic_altitude
    )


def direct_geocoding_monostatic(
    sensor_positions: CoordinatesArrayType,
    sensor_velocities: CoordinatesArrayType,
    range_times: float | FloatArrayType,
    frequencies_doppler_centroid: float | FloatArrayType,
    wavelength: float,
    look_direction: str | SensorLookDirection,
    geodetic_altitude: float,
    initial_guesses: CoordinatesArrayType | None = None,
) -> CoordinatesArrayType:
    """Perform direct geocoding for monostatic sensor.

    Parameters
    ----------
    sensor_positions : CoordinatesArrayType
        position of the sensor, with shape (3,) or (N, 3)
    sensor_velocities : CoordinatesArrayType
        velocity of the sensor, with shape (3,) or (N, 3)
    range_times : float | FloatArrayType
        range times, float or (M,)
    frequencies_doppler_centroid : float | FloatArrayType
        frequency_doppler_centroid value, single value or array (M,), if a single value is passed and there is more
        than 1 range times, it is broadcasted to all of them
    wavelength : float
        carrier signal wavelength
    look_direction : str | SensorLookDirection
        side where to perform geocoding
    geodetic_altitude : float
        the altitude over wgs84
    initial_guesses : CoordinatesArrayType | None, optional
        initial guess for Newton method. If not provided a guess will be computed, by default None

    Returns
    -------
    CoordinatesArrayType
        geocoded position for each input time and position value

    Raises
    ------
    AmbiguousInputCorrelation
        if inputs shapes are ambiguous to match, this error is raised
    """

    look_direction = SensorLookDirection(look_direction)

    # computation of initial guesses, if not provided
    if initial_guesses is None:
        # computing mid range distance
        average_input_range = np.median(range_times) * speed_of_light / 2
        initial_guesses = direct_geocoding_init(
            sensor_positions=sensor_positions,
            sensor_velocities=sensor_velocities,
            range_distance=average_input_range,
            look_direction=look_direction,
        )

    # direct geocoding monostatic core
    ground_points = direct_geocoding_monostatic_core(
        initial_guesses=initial_guesses,
        sensor_positions=sensor_positions,
        sensor_velocities=sensor_velocities,
        range_times=range_times,
        frequencies_doppler_centroid=frequencies_doppler_centroid,
        wavelength=wavelength,
        geodetic_altitude=geodetic_altitude,
    )

    return ground_points


def direct_geocoding_bistatic(
    sensor_positions_rx: CoordinatesArrayType,
    sensor_velocities_rx: CoordinatesArrayType,
    sensor_positions_tx: CoordinatesArrayType,
    sensor_velocities_tx: CoordinatesArrayType,
    range_times: float | FloatArrayType,
    frequencies_doppler_centroid: float | FloatArrayType,
    wavelength: float,
    look_direction: str | SensorLookDirection,
    geodetic_altitude: float,
    initial_guesses: CoordinatesArrayType | None = None,
) -> CoordinatesArrayType:
    """Perform direct geocoding for bistatic sensors.

    Parameters
    ----------
    sensor_positions_rx : CoordinatesArrayType
        position of the sensor rx, with shape (3,) or (N, 3)
    sensor_velocities_rx : CoordinatesArrayType
        velocity of the sensor rx, with shape (3,) or (N, 3)
    sensor_positions_tx : CoordinatesArrayType
        position of the sensor tx, with shape (3,) or (M, 3), where M is the number of range times
    sensor_velocities_tx : CoordinatesArrayType
        velocity of the sensor tx, with shape (3,) or (M, 3), where M is the number of range times
    range_times : float | FloatArrayType
        range times where to evaluate the direct geocoding, with shape float or (M,)
    frequencies_doppler_centroid : float | FloatArrayType
        frequency_doppler_centroid value, single value or array (M,), if a single value is passed and there is more
        than 1 range times, it is broadcasted to all of them
    wavelength : float
        carrier signal wavelength
    look_direction : str | SensorLookDirection
        side where to perform geocoding
    geodetic_altitude : float
        altitude with respect to the WGS84 ellipsoid
    initial_guesses : CoordinatesArrayType | None, optional
        initial guess for Newton method. If not provided a guess will be computed, by default None

    Returns
    -------
    CoordinatesArrayType
        ground points for each input time and position rx value
    """

    look_direction = SensorLookDirection(look_direction)

    # Optional initial guess
    if initial_guesses is None:
        # computing mid range distance
        average_input_range = np.median(range_times) * speed_of_light / 2
        initial_guesses = direct_geocoding_init(
            sensor_positions=sensor_positions_rx,
            sensor_velocities=sensor_velocities_rx,
            range_distance=average_input_range,
            look_direction=look_direction,
        )

    # direct geocoding bistatic core
    ground_points = direct_geocoding_bistatic_core(
        initial_guesses=initial_guesses,
        sensor_positions_rx=sensor_positions_rx,
        sensor_velocities_rx=sensor_velocities_rx,
        sensor_positions_tx=sensor_positions_tx,
        sensor_velocities_tx=sensor_velocities_tx,
        range_times=range_times,
        wavelength=wavelength,
        frequencies_doppler_centroid=frequencies_doppler_centroid,
        geodetic_altitude=geodetic_altitude,
    )

    return ground_points


def direct_geocoding_init(
    sensor_positions: CoordinatesArrayType,
    sensor_velocities: CoordinatesArrayType,
    range_distance: float,
    look_direction: str | SensorLookDirection,
) -> CoordinatesArrayType:
    """Computation of initial guesses for direct geocoding, monostatic approximation.

    Parameters
    ----------
    sensor_positions : CoordinatesArrayType
        sensor positions, with shape (3,) or (N, 3)
    sensor_velocities : CoordinatesArrayType
        sensor velocity, with shape (3,) or (N, 3)
    range_distance : float
        range distance
    look_direction : str | SensorLookDirection
        side where to perform geocoding

    Returns
    -------
    CoordinatesArrayType
        initial guess ground points

    Raises
    ------
    RuntimeError
        if range distance not compatible with sensor position and earth radius
    """

    one_size_array_flag = 0
    if sensor_velocities.ndim == sensor_positions.ndim == 1:
        one_size_array_flag = 1

    if sensor_positions.ndim < sensor_velocities.ndim:
        sensor_positions = np.broadcast_to(sensor_positions, sensor_velocities.shape)

    look_direction = SensorLookDirection(look_direction)
    geocoding_side_factor = 1 if look_direction == SensorLookDirection.RIGHT_LOOKING else -1

    sensor_position_norm = np.linalg.norm(sensor_positions, axis=-1, keepdims=True)
    llh_sat = xyz2llh(sensor_positions)
    if llh_sat.ndim == 1:
        llh_sat[2] = 0.0
    else:
        llh_sat[:, 2] = 0.0
    xyz_sat = llh2xyz(llh_sat)
    earth_radius = np.linalg.norm(xyz_sat, axis=-1, keepdims=True)

    # check earth radius vs range compatibility
    if any(range_distance < sensor_position_norm - earth_radius):
        raise RuntimeError("Cannot find initial guess for direct geocoding")

    u_x = sensor_positions / sensor_position_norm
    u_y = np.cross(sensor_positions, sensor_velocities)
    u_y = u_y / np.linalg.norm(u_y, axis=-1, keepdims=True)
    u_z = np.cross(u_x, u_y)

    # x-coordinate
    coords = (sensor_position_norm**2 + earth_radius**2 - range_distance**2) / (2 * sensor_position_norm)

    # circle radius
    circle_radius = np.sqrt(earth_radius**2 - coords**2)

    # Project velocity on ref frame
    v_x = np.sum(sensor_velocities * u_x, axis=-1)
    v_z = np.sum(sensor_velocities * u_z, axis=-1)

    # Find first solution
    z_solution = (sensor_position_norm - coords).T * v_x / v_z
    y_solution = np.sqrt(circle_radius.T**2 - z_solution**2)

    # inverting y solution by look sign
    y_solution[y_solution * geocoding_side_factor > 0] *= -1

    init_guess = coords * u_x + y_solution.T * u_y + z_solution.T * u_z

    return init_guess if not one_size_array_flag else init_guess.squeeze()
