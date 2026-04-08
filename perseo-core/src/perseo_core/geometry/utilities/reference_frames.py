# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Reference frames definitions and computing utilities"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from perseo_core.geometry.coords_conversions import llh2xyz, xyz2llh
from perseo_core.geometry.utilities.ellipsoid import WGS84
from perseo_core.geometry.utilities.rotations import (
    euler_angles_to_rotation,
)

_SIDEREAL_DAY = 86164.09054
_earth_angular_velocity = 2.0 * np.pi / _SIDEREAL_DAY
_semi_axes_ratio_min_max = WGS84.b / WGS84.a
_semi_axis_ratio_sqr = _semi_axes_ratio_min_max**2
_major_semi_axis_sqr = WGS84.a**2
_minor_semi_axis_sqr = WGS84.b**2


ReferenceFrame = Literal["GEOCENTRIC", "GEODETIC", "ZERODOPPLER"]


# TODO: improve documentation, specify input coords must be in Global Ref??
def compute_zerodoppler_reference_frame(sensor_positions: np.ndarray, sensor_velocities: np.ndarray) -> np.ndarray:
    """Compute the ZeroDoppler reference frame at given sensor positions and velocities.

    Reference frame

    - x-unit vector: oriented as sensor non-inertial velocity
    - y-unit vector: given by the cross product between x and sensor position corrected with Earth eccentricity
    - z-unit vector: completing the reference frame

    - output frame has x as first column, y as second one and z as the last one.

    Parameters
    ----------
    sensor_positions : np.ndarray
        sensor positions, with shape (3,) or (N, 3)
    sensor_velocities : np.ndarray
        sensor velocities, with shape (3,) or (N, 3)

    Returns
    -------
    np.ndarray
        zero doppler reference frame at each sensor position, with shape (3,3) or (N, 3, 3)

    Raises
    ------
    ValueError
        in case of invalid input
    """

    if sensor_positions.shape != sensor_velocities.shape:
        raise ValueError(
            "sensor_position and sensor_velocity have different shapes "
            + f"{sensor_positions.shape} != {sensor_velocities.shape}"
        )

    if sensor_positions.ndim > 2 or sensor_positions.shape[-1] != 3:
        raise ValueError(
            "sensor_position has invalid shape: " + f"{sensor_positions.shape}, it should be (3,) or (N, 3)"
        )

    unit_vector_x = sensor_velocities / np.linalg.norm(sensor_velocities, axis=-1, keepdims=True)

    adjusted_position = sensor_positions.copy()
    beta = 0.0060611
    adjusted_position[..., 2] *= 1.0 + beta

    unit_vector_y = np.cross(unit_vector_x, adjusted_position)
    unit_vector_y = unit_vector_y / np.linalg.norm(unit_vector_y, axis=-1, keepdims=True)

    unit_vector_z = np.cross(unit_vector_x, unit_vector_y)
    unit_vector_z = unit_vector_z / np.linalg.norm(unit_vector_z, axis=-1, keepdims=True)

    return np.stack([unit_vector_x, unit_vector_y, unit_vector_z], axis=-1)


def compute_geocentric_reference_frame(sensor_positions: np.ndarray, sensor_velocities: np.ndarray) -> np.ndarray:
    """Computed the geocentric reference frame at given sensor positions and velocities.

    Reference frame

    - x-unit vector: completing the reference frame
    - y-unit vector: given by the cross product between z and sensor inertial velocity
    - z-unit vector: oriented as -sat_pos

    Parameters
    ----------
    sensor_positions : np.ndarray
        sensor positions, with shape (3,) or (N, 3)
    sensor_velocities : np.ndarray
        sensor velocities, with shape (3,) or (N, 3)

    Returns
    -------
    np.ndarray
        geocentric reference frame for each sensor position, with shape (3,3) or (N, 3, 3)

    Raises
    ------
    ValueError
        in case of invalid input
    """

    sensor_positions = np.asarray(sensor_positions)
    sensor_velocities = np.asarray(sensor_velocities)

    if sensor_positions.shape != sensor_velocities.shape:
        raise ValueError(
            f"positions and velocities have different shapes {sensor_positions.shape} != {sensor_velocities.shape}"
        )

    if sensor_positions.ndim > 2 or sensor_positions.shape[-1] != 3:
        raise ValueError(f"sensor_position has invalid shape: {sensor_positions.shape}, it should be (3,) or (N, 3)")

    unit_vector_z = -sensor_positions
    unit_vector_z = unit_vector_z / np.linalg.norm(unit_vector_z, axis=-1, keepdims=True)

    inertial_velocity = compute_inertial_velocity(sensor_positions, sensor_velocities)

    unit_vector_y = np.cross(unit_vector_z, inertial_velocity)
    unit_vector_y = unit_vector_y / np.linalg.norm(unit_vector_y, axis=-1, keepdims=True)

    unit_vector_x = np.cross(unit_vector_y, unit_vector_z)
    unit_vector_x = unit_vector_x / np.linalg.norm(unit_vector_x, axis=-1, keepdims=True)

    return np.stack([unit_vector_x, unit_vector_y, unit_vector_z], axis=-1)


def compute_geodetic_reference_frame(sensor_positions: npt.ArrayLike, sensor_velocities: npt.ArrayLike) -> np.ndarray:
    """Compute the geodetic reference frame at given sensor positions and velocities.

    Parameters
    ----------
    sensor_positions : np.ndarray
        sensor positions, with shape (3,) or (N, 3)
    sensor_velocities : np.ndarray
        sensor velocities, with shape (3,) or (N, 3)

    Returns
    -------
    np.ndarray
        geodetic reference frame for each sensor position, with shape (3,3) or (N, 3, 3)

    Raises
    ------
    ValueError
        in case of invalid input
    """

    if sensor_positions.shape != sensor_velocities.shape:
        raise ValueError(
            "sensor_position and sensor_velocity have different shapes "
            + f"{sensor_positions.shape} != {sensor_velocities.shape}"
        )

    if sensor_positions.ndim > 2 or sensor_positions.shape[-1] != 3:
        raise ValueError(f"sensor_position has invalid shape: {sensor_positions.shape}, it should be (3,) or (N, 3)")

    geodetic_point = compute_geodetic_point(sensor_positions=sensor_positions)

    unit_vector_z = geodetic_point - sensor_positions
    unit_vector_z = unit_vector_z / np.linalg.norm(unit_vector_z, axis=-1, keepdims=True)

    geocentric_frame = compute_geocentric_reference_frame(
        sensor_positions=sensor_positions, sensor_velocities=sensor_velocities
    )

    z_geocentric = np.einsum("...jk, ...j->...k", geocentric_frame, unit_vector_z)
    z_geocentric = z_geocentric / np.linalg.norm(z_geocentric, axis=-1, keepdims=True)

    beta = -np.arctan2(z_geocentric[..., 1], z_geocentric[..., 2])

    rotation = euler_angles_to_rotation(
        order="YPR",
        euler_angles_rad=np.stack([np.zeros_like(beta), np.zeros_like(beta), beta], axis=-1),
    )

    rotated_frame = np.matmul(geocentric_frame, rotation.as_matrix())

    z_rotated = np.einsum("...jk, ...j->...k", rotated_frame, unit_vector_z)
    z_rotated = z_rotated / np.linalg.norm(z_rotated, axis=-1, keepdims=True)
    xsi = np.arctan2(z_rotated[..., 0], z_rotated[..., 2])

    second_rotation = euler_angles_to_rotation(
        order="YPR", euler_angles_rad=np.stack([np.zeros_like(xsi), xsi, np.zeros_like(xsi)], axis=-1)
    )

    return np.matmul(rotated_frame, second_rotation.as_matrix())


# TODO: improve documentation
def compute_sensor_local_axis(
    sensor_positions: np.ndarray,
    sensor_velocities: np.ndarray,
    reference_frame: ReferenceFrame,
) -> Rotation:
    """Compute the axis of the local reference frame given the sensor's positions and velocities.

    Parameters
    ----------
    sensor_positions : np.ndarray
        sensor positions, with shape (3,) or (N, 3)
    sensor_velocities : np.ndarray
        sensor velocities, with shape (3,) or (N, 3)
    reference_frame : "GEOCENTRIC", "GEODETIC", "ZERODOPPLER"
        reference frame

    Returns
    -------
    Rotation
        sensor's local axis as a Rotation object

    Examples
    --------

    single position and velocity

    >>> print(position.shape)
    (3,)
    >>> axis = compute_sensor_local_axis(position, velocity, "ZERODOPPLER")
    >>> print(axis.as_matrix().shape)
    (3, 3)

    multiple position and velocity

    >>> print(positions.shape)
    (10, 3)
    >>> axes = compute_sensor_local_axis(positions, velocities, "ZERODOPPLER")
    >>> print(axes.as_matrix().shape)
    (10, 3, 3)

    reference frame as string

    >>> compute_sensor_local_axis(position, velocity, "ZERODOPPLER")
    """
    match reference_frame:
        case "GEOCENTRIC":
            return Rotation.from_matrix(compute_geocentric_reference_frame(sensor_positions, sensor_velocities))
        case "GEODETIC":
            return Rotation.from_matrix(compute_geodetic_reference_frame(sensor_positions, sensor_velocities))
        case "ZERODOPPLER":
            return Rotation.from_matrix(compute_zerodoppler_reference_frame(sensor_positions, sensor_velocities))
        case _:
            raise ValueError(
                f"Unexpected reference_frame value: {reference_frame}. "
                "Must be one of 'GEOCENTRIC', 'GEODETIC', or 'ZERODOPPLER'."
            )


def compute_pointing_directions(
    antenna_reference_frame: Rotation,
    azimuth_antenna_angles: npt.ArrayLike,
    elevation_antenna_angles: npt.ArrayLike,
) -> np.ndarray:
    """Compute the pointing directions corresponding to the given antenna angles.

    Parameters
    ----------
    antenna_reference_frame : Rotation
        antenna reference frame for the sensor as a Rotation object, with 1 or N rotations
    azimuth_antenna_angles : npt.ArrayLike
        scalar or (N,) array like, in radians
    elevation_antenna_angles : npt.ArrayLike
        scalar or (N,) array like, in radians

    Returns
    -------
    np.ndarray
        pointing directions, with shape (3,) or (N, 3)

    Raises
    ------
    ValueError
        in case of mismatching input dimensions
    """
    azimuth_antenna_angles = np.asarray(azimuth_antenna_angles)
    elevation_antenna_angles = np.asarray(elevation_antenna_angles)
    assert isinstance(antenna_reference_frame, Rotation), "antenna_reference_frame must be a Rotation object"
    try:
        arf_num = len(antenna_reference_frame)
    except TypeError:
        arf_num = 1

    if arf_num != 1 and azimuth_antenna_angles.size != 1 and arf_num != np.size(azimuth_antenna_angles):
        raise ValueError(
            f"input shape mismatch: antenna reference frame {arf_num} != azimuth antenna angles "
            + f"{np.size(azimuth_antenna_angles)}"
        )

    if arf_num != 1 and elevation_antenna_angles.size != 1 and arf_num != np.size(elevation_antenna_angles):
        raise ValueError(
            f"input shape mismatch: antenna reference frame {arf_num} != elevation antenna angles "
            + f"{np.size(elevation_antenna_angles)}"
        )

    if (
        elevation_antenna_angles.size != 1
        and azimuth_antenna_angles.size != 1
        and elevation_antenna_angles.size != azimuth_antenna_angles.size
    ):
        raise ValueError(
            "Incompatible azimuth and elevation antenna angles numerosity: "
            + f"{azimuth_antenna_angles.size}, {elevation_antenna_angles.size}"
        )

    if azimuth_antenna_angles.shape != elevation_antenna_angles.shape:
        broadcast_shape = np.broadcast_shapes(azimuth_antenna_angles.shape, elevation_antenna_angles.shape)
        azimuth_antenna_angles = np.broadcast_to(azimuth_antenna_angles, broadcast_shape)
        elevation_antenna_angles = np.broadcast_to(elevation_antenna_angles, broadcast_shape)

    ux = np.tan(azimuth_antenna_angles)
    uy = np.tan(elevation_antenna_angles)
    uz = np.ones_like(ux)
    local_directions = np.stack([ux, uy, uz], axis=-1)
    local_directions = local_directions / np.linalg.norm(local_directions, axis=-1, keepdims=True)

    return antenna_reference_frame.apply(local_directions)


def compute_inertial_velocity(sensor_positions: np.ndarray, sensor_velocities: np.ndarray) -> np.ndarray:
    """Compute the sensor inertial velocity

    Parameters
    ----------
    sensor_positions : np.ndarray
        sensor positions, with shape (3,) or (N, 3)
    sensor_velocities : np.ndarray
        sensor velocities, with shape (3,) or (N, 3)

    Returns
    -------
    np.ndarray
        inertial velocity, one for each sensor position, with shape (3,) or (N, 3)

    Raises
    ------
    ValueError
        in case of invalid input
    """

    inertial_velocity = sensor_velocities.copy()
    inertial_velocity[..., 0] += -_earth_angular_velocity * sensor_positions[..., 1]
    inertial_velocity[..., 1] += _earth_angular_velocity * sensor_positions[..., 0]

    return inertial_velocity


def compute_geodetic_point(sensor_positions: np.ndarray) -> np.ndarray:
    """Compute the geodetic point that corresponds to a sensor position.

    Parameters
    ----------
    sensor_positions : np.ndarray
        sensor positions, with shape (3,) or (N, 3)

    Returns
    -------
    np.ndarray
        geodetic points, with shape (3,) or (N, 3)

    Raises
    ------
    ValueError
        sensor positions invalid shape
    """

    if sensor_positions.ndim > 2 or sensor_positions.shape[-1] != 3:
        raise ValueError(f"sensor_positions has invalid shape: {sensor_positions.shape}, it should be (3,) or (N, 3)")

    sensor_positions = sensor_positions.copy()
    change_sign = sensor_positions[..., 2] < 0
    sensor_positions[change_sign, 2] *= -1

    geodetic_points = xyz2llh(sensor_positions)
    if geodetic_points.ndim == 1:
        geodetic_points[2] = 0.0
    else:
        geodetic_points[:, 2] = 0.0
    geodetic_points = llh2xyz(geodetic_points).reshape(sensor_positions.shape)

    for sensor_position, point in zip(
        sensor_positions.reshape((-1, 3)),
        geodetic_points.reshape((-1, 3)),
        strict=False,
    ):
        increment_norm_threshold = 1e-9
        max_num_iterations = 10
        for _ in range(max_num_iterations):
            jacobian = _compute_geodetic_jacobian(point[:], sensor_position)
            rhs = _compute_geodetic_rhs(point[:], sensor_position)

            increment = -np.linalg.solve(jacobian, rhs)

            increment_norm = np.linalg.norm(increment)

            point[:] += increment

            if increment_norm < increment_norm_threshold:
                break

    geodetic_points[change_sign, 2] *= -1

    return geodetic_points


def _compute_geodetic_jacobian(point: np.ndarray, sensor_position: np.ndarray) -> np.ndarray:
    """Computing the geodetic Jacobian.

    Parameters
    ----------
    point : np.ndarray
        point on geoid
    sensor_position : np.ndarray
        sensor position

    Returns
    -------
    np.ndarray
        geodetic jacobian at that point
    """
    jac = np.empty(shape=(3, 3), dtype=float)

    zed_diff = point[2] - sensor_position[2]

    jac[0][0] = 2.0 * point[0] / _major_semi_axis_sqr
    jac[0][1] = 2.0 * point[1] / _major_semi_axis_sqr
    jac[0][2] = 2.0 * point[2] / _minor_semi_axis_sqr
    jac[1][0] = 1.0 + zed_diff * _ze_xx(point[0], point[1])
    jac[1][1] = zed_diff * _ze_xy(point[0], point[1])
    jac[1][2] = _ze_x(point[0], point[1])
    jac[2][0] = jac[1][1]
    jac[2][1] = 1.0 + zed_diff * _ze_xx(point[1], point[0])
    jac[2][2] = _ze_x(point[1], point[0])

    return jac


def _compute_geodetic_rhs(point: np.ndarray, sensor_position: np.ndarray) -> np.ndarray:
    """Computing geodetic rhs.

    Parameters
    ----------
    point : np.ndarray
        point on geoid
    sensor_position : np.ndarray
        sensor position

    Returns
    -------
    np.ndarray
        geodetic rhs at given point
    """
    rhs = np.empty(shape=(3,), dtype=float)
    los = point - sensor_position
    rhs[0] = (point[0] ** 2 + point[1] ** 2) / _major_semi_axis_sqr + point[2] ** 2 / _minor_semi_axis_sqr - 1
    rhs[1] = los[0] + los[2] * _ze_x(point[0], point[1])
    rhs[2] = los[1] + los[2] * _ze_x(point[1], point[0])
    return rhs


def _ze(x: float, y: float) -> float:
    eps = 1.0e-21

    return np.sqrt(_minor_semi_axis_sqr - _semi_axis_ratio_sqr * (x**2 + y**2)) + eps


def _ze_x(x: float, y: float) -> float:
    return -_semi_axes_ratio_min_max * x / _ze(x, y)


def _ze_xy(x: float, y: float) -> float:
    # pylint: disable-next=arguments-out-of-order
    return (_semi_axis_ratio_sqr * x) / _ze(x, y) ** 2 * _ze_x(y, x)


def _ze_xx(x: float, y: float) -> float:
    return -_semi_axis_ratio_sqr / _ze(x, y) + (_semi_axis_ratio_sqr * x) / _ze(x, y) ** 2 * _ze_x(x, y)
