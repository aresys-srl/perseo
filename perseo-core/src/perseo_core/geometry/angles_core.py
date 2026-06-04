# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Computation of incidence, look, squint and other angles"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from perseo_core.geometry.coords_conversions import llh2xyz, xyz2llh


def get_geometric_squint_angle(
    sensor_positions: npt.NDArray[np.floating],
    sensor_velocities: npt.NDArray[np.floating],
    ground_points: npt.NDArray[np.floating],
    radians: bool = True,
) -> float | npt.NDArray[np.floating]:
    """Evaluating squint angle geometrically in radians/degrees.

    Parameters
    ----------
    sensor_positions : npt.NDArray[np.floating]
        sensor positions array, with shape (3,) or (N, 3)
    sensor_velocities : npt.NDArray[np.floating]
        sensor velocities array, with shape (3,) or (N, 3)
    ground_points : npt.NDArray[np.floating]
        ground points array, with shape (3,) or (N, 3)
    radians : bool, optional
        output angles are expressed in radians if this flag is set to True (default), if False are expressed in degrees

    Returns
    -------
    float | npt.NDArray[np.floating]
        squint angle in radians/degrees
    """

    # evaluating squint angle
    line_of_sight = ground_points - sensor_positions
    line_of_sight = line_of_sight / np.linalg.norm(line_of_sight, axis=-1, keepdims=True)
    sensor_velocity_norm = sensor_velocities / np.linalg.norm(sensor_velocities, axis=-1, keepdims=True)
    squint_angle = np.arcsin(np.sum(line_of_sight * sensor_velocity_norm, axis=-1))
    if not radians:
        return np.rad2deg(squint_angle)
    return squint_angle


def compute_look_angles_core(
    sensor_positions: npt.NDArray[np.floating],
    ground_points: npt.NDArray[np.floating],
    nadir_directions: npt.NDArray[np.floating] | None = None,
    assume_nadir_directions_normalized: bool = False,
) -> float | npt.NDArray[np.floating]:
    """Compute the look angles in radians from sensor positions and ground points.

    Parameters
    ----------
    sensor_positions : npt.NDArray[np.floating]
        (3,) or (N, 3) one or more sensor positions
    ground_points : npt.NDArray[np.floating]
        (3,) or (N, 3) one or more points
    nadir_directions : npt.NDArray[np.floating]
        (3,) or (N, 3) one or more nadir directions
    assume_nadir_directions_normalized : bool, optional
        True to skip nadir directions normalization, by default False

    Returns
    -------
    float | npt.NDArray[np.floating]
        scalar or (N,) look angles in radians

    Raises
    ------
    ValueError
        in case of invalid input

    Examples
    --------

    1 position, nadir -- 1 point

    >>> look_angle = compute_look_angles(position, nadir_dir, point)

    N positions, nadirs -- 1 point -- point is broadcasted

    >>> look_angles = compute_look_angles(positions, nadir_directions, point)

    1 position, nadir -- N points -- position and nadir are broadcasted

    >>> look_angles = compute_look_angles(position, nadir_dir, points)

    N positions, nadirs -- N points

    >>> look_angles = compute_look_angles(positions, nadir_directions, points)

    Skip normalization of nadir direction

    >>> look_angle = compute_look_angles(position, nadir_dir, point, assume_nadir_directions_normalized=True)

    """
    sensor_positions = np.asarray(sensor_positions)
    ground_points = np.asarray(ground_points)
    if nadir_directions is None:
        nadir_directions = compute_nadir_from_sensor_positions(sensor_positions=sensor_positions)
    else:
        nadir_directions = np.asarray(nadir_directions)

    if sensor_positions.ndim > 2 or sensor_positions.shape[-1] != 3:
        raise ValueError(f"sensor_positions has invalid shape: {sensor_positions.shape}, it should be (3,) or (N, 3)")

    if nadir_directions.ndim > 2 or nadir_directions.shape[-1] != 3:
        raise ValueError(f"nadir_directions has invalid shape: {nadir_directions.shape}, it should be (3,) or (N, 3)")

    if ground_points.ndim > 2 or ground_points.shape[-1] != 3:
        raise ValueError(f"points has invalid shape: {ground_points.shape}, it should be (3,) or (N, 3)")

    los_directions = ground_points - sensor_positions
    with np.errstate(invalid="ignore", divide="ignore"):
        los_directions = los_directions / np.linalg.norm(los_directions, axis=-1, keepdims=True)

    if not assume_nadir_directions_normalized:
        nadir_directions = nadir_directions / np.linalg.norm(nadir_directions, axis=-1, keepdims=True)

    look_angle_cosines = np.sum(nadir_directions * los_directions, axis=-1)

    return np.arccos(np.clip(look_angle_cosines, a_min=-1.0, a_max=1.0))


def compute_incidence_angles_core(
    sensor_positions: npt.NDArray[np.floating],
    ground_points: npt.NDArray[np.floating],
    surface_normals: npt.NDArray[np.floating] | None = None,
    assume_surface_normals_normalized: bool = False,
) -> float | npt.NDArray[np.floating]:
    """Compute the incidence angles in radians from sensor positions and ground points.

    If surface normals are not specified, points are used to define the surface normals

    .. code-block:: python

        surface_normals = points
        assume_surface_normals_normalized = False

    Parameters
    ----------
    sensor_positions : npt.NDArray[np.floating]
        (3,) or (N, 3) one or more sensor positions
    ground_points : npt.NDArray[np.floating]
        (3,) or (N, 3) one or more points
    surface_normals : npt.NDArray[np.floating] | None, optional
        (3,) or (N, 3) one or more surface normal directions, by default None
    assume_surface_normals_normalized : bool, optional
        True to skip surface normals normalization, by default False

    Returns
    -------
    float | npt.NDArray[np.floating]
        scalar or (N,) look angles in radians

    Raises
    ------
    ValueError
        in case of invalid input

    Examples
    --------

    1 position -- 1 point

    >>> incidence_angle = compute_incidence_angles(position, point)

    N positions -- 1 point -- point is broadcasted

    >>> incidence_angles = compute_incidence_angles(positions, point)

    1 position -- N points -- position is broadcasted

    >>> incidence_angles = compute_incidence_angles(position, points)

    N positions -- N points

    >>> incidence_angles = compute_incidence_angles(positions, points)

    User defined surface normal with normalization skipping

    >>> incidence_angle = compute_incidence_angles(position, point,
            surface_normals=surf_norm,
            assume_nadir_directions_normalized=True)
    """

    sensor_positions = np.asarray(sensor_positions)
    ground_points = np.asarray(ground_points)

    if sensor_positions.ndim > 2 or sensor_positions.shape[-1] != 3:
        raise ValueError(f"sensor_positions has invalid shape: {sensor_positions.shape}, it should be (3,) or (N, 3)")

    if ground_points.ndim > 2 or ground_points.shape[-1] != 3:
        raise ValueError(f"points has invalid shape: {ground_points.shape}, it should be (3,) or (N, 3)")

    if surface_normals is not None:
        surface_normals = np.asarray(surface_normals)
        if surface_normals.ndim > 2 or surface_normals.shape[-1] != 3:
            raise ValueError(f"surface_normals has invalid shape: {surface_normals.shape}, it should be (3,) or (N, 3)")

    los_directions = ground_points - sensor_positions
    los_directions = los_directions / np.linalg.norm(los_directions, axis=-1, keepdims=True)

    if surface_normals is None:
        surface_normals = ground_points
        assume_surface_normals_normalized = False

    if not assume_surface_normals_normalized:
        surface_normals = surface_normals / np.linalg.norm(surface_normals, axis=-1, keepdims=True)

    incidence_angle_cosine = -1.0 * np.sum(surface_normals * los_directions, axis=-1)

    return np.arccos(np.clip(incidence_angle_cosine, a_min=-1.0, a_max=1.0))


def compute_nadir_from_sensor_positions(
    sensor_positions: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute nadir positions from sensor positions.

    Parameters
    ----------
    sensor_positions : npt.NDArray[np.floating]
        sensor positions, with shape (3,), (N, 3)

    Returns
    -------
    npt.NDArray[np.floating]
        nadir position, with shape (3,), (N, 3)
    """
    sensor_position_ground = xyz2llh(sensor_positions)
    if sensor_position_ground.ndim == 1:
        sensor_position_ground[2] = 0.0
    else:
        sensor_position_ground[:, 2] = 0.0
    sensor_position_ground = llh2xyz(sensor_position_ground)

    return sensor_position_ground - sensor_positions
