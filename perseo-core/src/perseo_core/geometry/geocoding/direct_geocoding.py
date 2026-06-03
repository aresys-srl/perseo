# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
This module provides high-level direct geocoding functions for computing ground point coordinates from sensor
state vectors at given sensor times. It supports both monostatic (single sensor) and bistatic
(separate transmitter/receiver) configurations.

Direct geocoding solves the intersection of the range-Doppler-ellipsoid system to find the Earth surface point
corresponding to given sensor times. The equations are solved via Newton iterations.

All functions support vectorized operations on (N, 3) coordinate arrays and handle both scalar
and array inputs for range times and Doppler frequencies. The WGS84 ellipsoid is used as the
Earth model with configurable altitude offsets.

Look direction is specified as `RIGHT` or `LEFT` to determine the geocoding side relative
to the sensor velocity vector.
"""

from __future__ import annotations

from typing import Literal, get_args

import numpy as np
from numpy import typing as npt
from scipy.constants import speed_of_light

from perseo_core.geometry.coords_conversions import llh2xyz, xyz2llh
from perseo_core.geometry.ellipsoid import (
    compute_line_ellipsoid_intersections,
    create_inflated_WGS84_ellipsoid,
)
from perseo_core.geometry.geocoding.direct_geocoding_core import (
    direct_geocoding_bistatic_core,
    direct_geocoding_monostatic_core,
    direct_geocoding_monostatic_core_range_vectorized,
)
from perseo_core.geometry.pointing.antenna_reference_frame import (
    compute_antenna_reference_frame_from_euler_angles,
    compute_pointing_directions,
)
from perseo_core.geometry.pointing.reference_frames import (
    ReferenceFrame,
    compute_sensor_local_axis,
)

SensorLookDirection = Literal["RIGHT", "LEFT"]
_VALID_SENSOR_LOOK_DIRECTIONS = get_args(SensorLookDirection)


def direct_geocoding_with_looking_direction(
    sensor_positions: npt.NDArray[np.floating],
    looking_directions: npt.NDArray[np.floating],
    altitude: float = 0.0,
) -> npt.NDArray[np.floating]:
    """Compute the ground points seen with the given looking directions.

    The looking direction defines a line: its norm and sign do not matter.

    Based on :meth:`perseo_core.geometry.ellipsoid.compute_line_ellipsoid_intersections`

    Parameters
    ----------
    sensor_positions : npt.NDArray[np.floating]
        sensor positions with shape (3,) or (N, 3)
    looking_directions : npt.NDArray[np.floating]
        vectors aligned with a looking direction with shape (3,) or (N, 3)
    altitude : float, optional
        altitude with respect to WGS84 ellipsoid, by default 0.0

    Returns
    -------
    npt.NDArray[np.floating]
        ground points with shape (3,) or (N, 3), np.nan is a place-holder in case of impossible geocoding
    """
    inflated_ellipsoid = create_inflated_WGS84_ellipsoid(altitude)

    first_intersections, _ = compute_line_ellipsoid_intersections(
        line_directions=looking_directions,
        line_origins=sensor_positions,
        ellipsoid=inflated_ellipsoid,
    )

    return first_intersections


def direct_geocoding_with_look_angles(
    sensor_positions: npt.NDArray[np.floating],
    sensor_velocities: npt.NDArray[np.floating],
    reference_frame: ReferenceFrame,
    look_angles: float | npt.NDArray[np.floating],
    altitude: float = 0.0,
) -> npt.NDArray[np.floating]:
    """Compute the points at a given altitude over WGS84 ellipsoid seen with the given look angles.

    Based on :meth:`perseo_core.geometry.geocoding.direct_geocoding.direct_geocoding_with_looking_direction`

    Parameters
    ----------
    sensor_positions : npt.NDArray[np.floating]
        sensor positions with shape (3,) or (N, 3)
    sensor_velocities : npt.NDArray[np.floating]
        sensor velocities with shape (3,) or (N, 3)
    reference_frame : ReferenceFrame
        reference frames oriented so that look angles are measured relative to the nadir direction
    look_angles : float | npt.NDArray[np.floating]
        look angles in radians, scalar or (N,)
    altitude : float, optional
        altitude with respect to WGS84 ellipsoid, by default 0.0

    Returns
    -------
    npt.NDArray[np.floating]
        ground points with shape (3,) or (N, 3), np.nan is a place-holder in case of impossible geocoding
    """
    local_axis = compute_sensor_local_axis(
        sensor_positions=sensor_positions, sensor_velocities=sensor_velocities, reference_frame=reference_frame
    )

    look_angles = np.atleast_1d(look_angles)
    assert look_angles.ndim == 1
    ypr_rad = np.stack([np.zeros_like(look_angles), np.zeros_like(look_angles), -np.asarray(look_angles)], axis=-1)

    arf = compute_antenna_reference_frame_from_euler_angles(
        rotation_order="YPR", ypr_rad=ypr_rad, initial_reference_frame=local_axis
    )
    pointing = arf.squeeze()[..., 2]

    return direct_geocoding_with_looking_direction(
        sensor_positions=sensor_positions, looking_directions=pointing, altitude=altitude
    )


def direct_geocoding_with_pointing(
    sensor_positions: npt.NDArray[np.floating],
    antenna_reference_frames: npt.NDArray[np.floating],
    azimuth_antenna_angles: float | npt.NDArray[np.floating],
    elevation_antenna_angles: float | npt.NDArray[np.floating],
    altitude: float = 0.0,
) -> npt.NDArray[np.floating]:
    """Compute ground points illuminated with the given antenna patterns angles

    Parameters
    ----------
    sensor_positions : npt.NDArray[np.floating]
        sensor positions, with shape (3,) or (N, 3)
    antenna_reference_frames : npt.NDArray[np.floating]
        antenna reference frames as a numpy array, with shape (3, 3) or (N, 3, 3)
    azimuth_antenna_angles : float | npt.NDArray[np.floating]
        scalar or (N,), in radians
    elevation_antenna_angles : float | npt.NDArray[np.floating]
        scalar or (N,), in radians
    altitude : float, optional
        altitude with respect to WGS84 ellipsoid, by default 0.0

    Returns
    -------
    npt.NDArray[np.floating]
        ground points (3,) or (N, 3) numpy array
    """
    if antenna_reference_frames.ndim not in (2, 3) or antenna_reference_frames.shape[-2:] != (3, 3):
        raise ValueError(
            f"antenna_reference_frames must have shape (3, 3) or (N, 3, 3), got {antenna_reference_frames.shape}"
        )

    arf_num = 1 if antenna_reference_frames.ndim == 2 else antenna_reference_frames.shape[0]
    if arf_num != np.size(sensor_positions) // 3:
        raise ValueError(
            f"input shape mismatch: antenna reference frames {arf_num} != sensor positions {sensor_positions.shape}"
        )

    return direct_geocoding_with_looking_direction(
        sensor_positions=sensor_positions,
        looking_directions=compute_pointing_directions(
            antenna_reference_frame=antenna_reference_frames,
            azimuth_antenna_angles=azimuth_antenna_angles,
            elevation_antenna_angles=elevation_antenna_angles,
        ),
        altitude=altitude,
    )


def direct_geocoding_monostatic(
    sensor_positions: npt.NDArray[np.floating],
    sensor_velocities: npt.NDArray[np.floating],
    range_times: float | npt.NDArray[np.floating],
    doppler_frequencies: float | npt.NDArray[np.floating],
    wavelength: float,
    look_direction: SensorLookDirection,
    altitude: float,
    initial_guesses: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.floating]:
    """Perform monostatic direct geocoding.

    Parameters
    ----------
    sensor_positions : npt.NDArray[np.floating]
        position of the sensor with shape (3,) or (N, 3)
    sensor_velocities : npt.NDArray[np.floating]
        velocity of the sensor with shape (3,) or (N, 3)
    range_times : float | npt.NDArray[np.floating]
        range times scalar or (M,)
    doppler_frequencies : float | npt.NDArray[np.floating]
        doppler frequencies scalar or array (M,)
    wavelength : float
        carrier signal wavelength
    look_direction : SensorLookDirection
        geocoding side, "RIGHT" or "LEFT"
    altitude : float
        altitude with respect to WGS84 ellipsoid
    initial_guesses : npt.NDArray[np.floating] | None, optional
        initial guess for Newton method. If not provided a guess will be computed, by default None

    Returns
    -------
    npt.NDArray[np.floating]
        ground points with shape (N, M, 3)
    """
    if look_direction not in _VALID_SENSOR_LOOK_DIRECTIONS:
        raise ValueError(f"Invalid look direction: {look_direction}. Must be one of {_VALID_SENSOR_LOOK_DIRECTIONS}")

    if initial_guesses is None:
        average_input_range: float = np.median(range_times) * speed_of_light / 2
        initial_guesses = direct_geocoding_init(
            sensor_positions=sensor_positions,
            sensor_velocities=sensor_velocities,
            range_distance=average_input_range,
            look_direction=look_direction,
        )

    if 10 * (sensor_positions.size // 3) < np.size(range_times):
        if not isinstance(range_times, float):
            initial_guesses = np.repeat(
                np.atleast_2d(initial_guesses).mean(axis=0, keepdims=True), np.size(range_times), axis=0
            )
        return direct_geocoding_monostatic_core_range_vectorized(
            initial_guesses=initial_guesses,
            sensor_positions=sensor_positions,
            sensor_velocities=sensor_velocities,
            range_times=range_times,
            doppler_frequencies=doppler_frequencies,
            wavelength=wavelength,
            altitude=altitude,
        )

    return direct_geocoding_monostatic_core(
        initial_guesses=initial_guesses,
        sensor_positions=sensor_positions,
        sensor_velocities=sensor_velocities,
        range_times=range_times,
        doppler_frequencies=doppler_frequencies,
        wavelength=wavelength,
        altitude=altitude,
    )


def direct_geocoding_bistatic(
    sensor_positions_rx: npt.NDArray[np.floating],
    sensor_velocities_rx: npt.NDArray[np.floating],
    sensor_positions_tx: npt.NDArray[np.floating],
    sensor_velocities_tx: npt.NDArray[np.floating],
    range_times: float | npt.NDArray[np.floating],
    doppler_frequencies: float | npt.NDArray[np.floating],
    wavelength: float,
    look_direction: SensorLookDirection,
    altitude: float,
    initial_guesses: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.floating]:
    """Perform direct geocoding for bistatic sensors.

    Parameters
    ----------
    sensor_positions_rx : npt.NDArray[np.floating]
        position of the receiver with shape (3,) or (N, 3)
    sensor_velocities_rx : npt.NDArray[np.floating]
        velocity of the receiver with shape (3,) or (N, 3)
    sensor_positions_tx : npt.NDArray[np.floating]
        position of the transmitter with shape (3,) or (M, 3), where M is the number of range times
    sensor_velocities_tx : npt.NDArray[np.floating]
        velocity of the transmitter with shape (3,) or (M, 3), where M is the number of range times
    range_times : float | npt.NDArray[np.floating]
        range times scalar or shape (M,)
    doppler_frequencies : float | npt.NDArray[np.floating]
        doppler frequencies scalar or shape (M,)
    wavelength : float
        carrier signal wavelength
    look_direction : SensorLookDirection
        geocoding side, "RIGHT" or "LEFT"
    altitude : float
        altitude with respect to the WGS84 ellipsoid
    initial_guesses : npt.NDArray[np.floating] | None, optional
        initial guess for Newton iterations. If not provided a guess will be computed, by default None

    Returns
    -------
    npt.NDArray[np.floating]
        ground points with shape (N, M, 3)
    """
    if look_direction not in _VALID_SENSOR_LOOK_DIRECTIONS:
        raise ValueError(f"Invalid look direction: {look_direction}. Must be one of {_VALID_SENSOR_LOOK_DIRECTIONS}")

    if initial_guesses is None:
        average_input_range = np.median(range_times) * speed_of_light / 2
        initial_guesses = direct_geocoding_init(
            sensor_positions=sensor_positions_rx,
            sensor_velocities=sensor_velocities_rx,
            range_distance=average_input_range,
            look_direction=look_direction,
        )

    return direct_geocoding_bistatic_core(
        initial_guesses=initial_guesses,
        sensor_positions_rx=sensor_positions_rx,
        sensor_velocities_rx=sensor_velocities_rx,
        sensor_positions_tx=sensor_positions_tx,
        sensor_velocities_tx=sensor_velocities_tx,
        range_times=range_times,
        wavelength=wavelength,
        doppler_frequencies=doppler_frequencies,
        altitude=altitude,
    )


def direct_geocoding_init(
    sensor_positions: npt.NDArray[np.floating],
    sensor_velocities: npt.NDArray[np.floating],
    range_distance: float,
    look_direction: SensorLookDirection,
) -> npt.NDArray[np.floating]:
    """Compute initial guesses for direct geocoding, monostatic approximation.

    Parameters
    ----------
    sensor_positions : npt.NDArray[np.floating]
        sensor positions with shape (3,) or (N, 3)
    sensor_velocities : npt.NDArray[np.floating]
        sensor velocities with shape (3,) or (N, 3)
    range_distance : float
        range distance
    look_direction : SensorLookDirection
        side where to perform geocoding, "RIGHT" or "LEFT"

    Returns
    -------
    npt.NDArray[np.floating]
        guess ground points with shape (3,) or (N, 3)
    """

    one_size_array_flag = sensor_velocities.ndim == sensor_positions.ndim == 1

    if sensor_positions.ndim < sensor_velocities.ndim:
        sensor_positions = np.broadcast_to(sensor_positions, sensor_velocities.shape)

    if look_direction not in _VALID_SENSOR_LOOK_DIRECTIONS:
        raise ValueError(f"Invalid look direction: {look_direction}. Must be one of {_VALID_SENSOR_LOOK_DIRECTIONS}")
    geocoding_side_factor = 1 if look_direction == "RIGHT" else -1

    sensor_position_norm = np.linalg.norm(sensor_positions, axis=-1, keepdims=True)
    llh_sat = xyz2llh(sensor_positions)
    llh_sat[..., 2] = 0.0
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
