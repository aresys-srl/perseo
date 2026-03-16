# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Direct geocoding main functionalities"""

from __future__ import annotations

import numpy as np
from numpy import typing as npt
from scipy.constants import speed_of_light
from scipy.spatial.transform import Rotation

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
    compute_pointing_directions,
    compute_sensor_local_axis,
)
from perseo_core.geometry.utilities.rotations import euler_angles_to_rotation
from perseo_core.models.enums import SensorLookDirection


def direct_geocoding_with_looking_direction(
    sensor_positions: npt.NDArray[np.floating],
    looking_directions: npt.NDArray[np.floating],
    altitude: float = 0.0,
) -> npt.NDArray[np.floating]:
    """Compute the ground points seen with the given looking directions.

    The looking direction defines a line: its norm and sign do not matter.

    Based on :meth:`perseo.geometry.utilities.ellipsoid.compute_line_ellipsoid_intersections`

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

    intersections = compute_line_ellipsoid_intersections(
        line_directions=looking_directions,
        line_origins=sensor_positions,
        ellipsoid=inflated_ellipsoid,
    )

    points = np.empty(np.broadcast_shapes(np.shape(looking_directions), np.shape(sensor_positions)))

    if points.ndim == 1:
        intersections = (intersections,)

    for intersections_pair, point in zip(intersections, points.reshape((-1, 3)), strict=False):
        point[:] = intersections_pair[0] if len(intersections_pair) > 0 else np.nan

    return points


def direct_geocoding_with_look_angles(
    sensor_positions: npt.NDArray[np.floating],
    sensor_velocities: npt.NDArray[np.floating],
    reference_frame: ReferenceFrameLike,
    look_angles: float | npt.NDArray[np.floating],
    altitude: float = 0.0,
) -> npt.NDArray[np.floating]:
    """Compute the points at a given altitude over WGS84 ellipsoid seen with the given look angles.

    Based on :meth:`perseo.geometry.geocoding.direct_geocoding.direct_geocoding_with_looking_direction`

    Parameters
    ----------
    sensor_positions : npt.NDArray[np.floating]
        sensor positions with shape (3,) or (N, 3)
    sensor_velocities : npt.NDArray[np.floating]
        sensor velocities with shape (3,) or (N, 3)
    reference_frame : ReferenceFrameLike
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
    rotation = euler_angles_to_rotation(
        euler_angles_rad=np.stack(
            [np.zeros_like(look_angles), np.zeros_like(look_angles), -np.asarray(look_angles)], axis=-1
        ),
        order="YPR",
    )

    pointing = (local_axis * rotation).as_matrix().squeeze()[..., 2]

    return direct_geocoding_with_looking_direction(
        sensor_positions=sensor_positions, looking_directions=pointing, altitude=altitude
    )


def direct_geocoding_with_pointing(
    sensor_positions: npt.NDArray[np.floating],
    antenna_reference_frames: Rotation,
    azimuth_antenna_angles: float | npt.NDArray[np.floating],
    elevation_antenna_angles: float | npt.NDArray[np.floating],
    altitude: float = 0.0,
) -> npt.NDArray[np.floating]:
    """Compute ground points illuminated with the given antenna patterns angles

    Parameters
    ----------
    sensor_positions : npt.NDArray[np.floating]
        sensor positions, with shape (3,) or (N, 3)
    antenna_reference_frames : Rotation
        antenna reference frames as a Rotation object, with 1 or N rotations
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

    arf_num = len(antenna_reference_frames) if len(antenna_reference_frames.shape) == 1 else 1
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
    look_direction: str | SensorLookDirection,
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
    look_direction : str | SensorLookDirection
        geocoding side
    altitude : float
        altitude with respect to WGS84 ellipsoid
    initial_guesses : npt.NDArray[np.floating] | None, optional
        initial guess for Newton method. If not provided a guess will be computed, by default None

    Returns
    -------
    npt.NDArray[np.floating]
        ground points with shape (N, M, 3)
    """
    look_direction = SensorLookDirection(look_direction)

    if initial_guesses is None:
        average_input_range: float = np.median(range_times) * speed_of_light / 2
        initial_guesses = direct_geocoding_init(
            sensor_positions=sensor_positions,
            sensor_velocities=sensor_velocities,
            range_distance=average_input_range,
            look_direction=look_direction,
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
    look_direction: str | SensorLookDirection,
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
    look_direction : str | SensorLookDirection
        geocoding side
    altitude : float
        altitude with respect to the WGS84 ellipsoid
    initial_guesses : npt.NDArray[np.floating] | None, optional
        initial guess for Newton iterations. If not provided a guess will be computed, by default None

    Returns
    -------
    npt.NDArray[np.floating]
        ground points with shape (N, M, 3)
    """
    look_direction = SensorLookDirection(look_direction)

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
    look_direction: str | SensorLookDirection,
) -> npt.NDArray[np.floating]:
    """Computate initial guesses for direct geocoding, monostatic approximation.

    Parameters
    ----------
    sensor_positions : npt.NDArray[np.floating]
        sensor positions with shape (3,) or (N, 3)
    sensor_velocities : npt.NDArray[np.floating]
        sensor velocities with shape (3,) or (N, 3)
    range_distance : float
        range distance
    look_direction : str | SensorLookDirection
        side where to perform geocoding

    Returns
    -------
    npt.NDArray[np.floating]
        guess ground points with shape (3,) or (N, 3)
    """

    one_size_array_flag = sensor_velocities.ndim == sensor_positions.ndim == 1

    if sensor_positions.ndim < sensor_velocities.ndim:
        sensor_positions = np.broadcast_to(sensor_positions, sensor_velocities.shape)

    look_direction = SensorLookDirection(look_direction)
    geocoding_side_factor = 1 if look_direction == SensorLookDirection.RIGHT_LOOKING else -1

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
