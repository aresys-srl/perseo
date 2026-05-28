# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Computation of Doppler-related quantities"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.constants import speed_of_light

from perseo_core.geometry.angles import get_geometric_squint_angle
from perseo_core.models.trajectory import Trajectory
from perseo_core.timing.precise_datetime import PreciseDateTime


# TODO: this is defined also in direct_geocoding_core as _doppler_equation, duplicated to avoid circular import
def doppler_equation(
    wavelength: float,
    pv_scalar: float | npt.NDArray[np.floating],
    distance: float | npt.NDArray[np.floating],
    doppler_frequency: float | npt.NDArray[np.floating],
    sensor_velocity: npt.NDArray[np.floating],
    los: npt.NDArray[np.floating],
) -> tuple[float | npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Doppler equation solver.

    Parameters
    ----------
    wavelength : float
        carrier signal wavelength
    pv_scalar : float | npt.NDArray[np.floating],
        scalar product between sensor velocity and line of sight scalar or shape (N,)
    distance : float
        ground point - sensor position distance scalar or shape (N,)
    doppler_frequency : float
        doppler frequency scalar or shape (N,)
    sensor_velocity : npt.NDArray[np.floating]
        sensor velocity (3,) or shape (N, 3)
    los : npt.NDArray[np.floating]
        ground point - sensor position (3,) or shape (N, 3)

    Returns
    -------
    float | npt.NDArray[np.floating]
        doppler equation solution scalar or shape (N,)
    npt.NDArray[np.floating]
        doppler equation gradient (3,) or shape (N, 3)
    """

    c_factor = 2.0 / wavelength / distance
    doppler_equation = c_factor * pv_scalar + doppler_frequency
    norm_pv = pv_scalar / distance**2
    grad_doppler_equation = (c_factor * (-sensor_velocity + (norm_pv * los.T).T).T).T
    return doppler_equation, grad_doppler_equation


def doppler_equation_monostatic_residuals(
    ground_point: npt.NDArray[np.floating],
    sensor_positions: npt.NDArray[np.floating],
    sensor_velocities: npt.NDArray[np.floating],
    doppler_frequency: float,
    wavelength: float,
) -> float | npt.NDArray[np.floating]:
    """Evaluate SAR doppler equation residual, assuming monostatic approximation.

    *Doppler Equation*

    $$
    f_{doppler} = \\frac{2}{\\lambda}  \\frac{v_{sensor} \\cdot LOS}{\\|LOS\\|}
    $$

    *Doppler Equation Residual*

    $$
    residual = \\frac{2}{\\lambda}  \\frac{v_{sensor} \\cdot LOS}{\\|LOS\\|} - f_{doppler}
    $$

    where LOS is defined as the line of sight, a.k.a. the position difference between the ground point and the sensor
    positions.

    Parameters
    ----------
    ground_point : npt.NDArray[np.floating]
        ground point in ECEF coordinates, with shape (3,)
    sensor_positions : npt.NDArray[np.floating]
        sensor positions, with shape (3,) or (N, 3)
    sensor_velocities : npt.NDArray[np.floating]
        sensor velocities, with shape (3,) or (N, 3)
    doppler_frequency : float
        frequency doppler centroid in Hz
    wavelength : float
        signal carrier wavelength in meters

    Returns
    -------
    float | npt.NDArray[np.floating]
        doppler equation residual (Hz) for each input sensor position, scalar or (N,)
    """

    if sensor_positions.ndim > 2 or sensor_positions.shape[-1] != 3:
        raise ValueError(f"sensor_positions has invalid shape: {sensor_positions.shape}, it should be (3,) or (N, 3)")

    if sensor_velocities.ndim > 2 or sensor_velocities.shape[-1] != 3:
        raise ValueError(f"sensor_velocities has invalid shape: {sensor_velocities.shape}, it should be (3,) or (N, 3)")

    is_scalar = ground_point.ndim == 1 and sensor_positions.ndim == 1 and sensor_velocities.ndim == 1

    ground_point = np.atleast_1d(ground_point)
    sensor_positions = np.atleast_2d(sensor_positions)
    sensor_velocities = np.atleast_2d(sensor_velocities)

    line_of_sight = ground_point - sensor_positions
    line_of_sight_norm = np.linalg.norm(line_of_sight, axis=1)

    def col_wise_scalar_product(matrix_a, matrix_b):
        return np.einsum("ij,ij->i", matrix_a, matrix_b)  # Einstein notation -- col wise dot product.

    result = (
        np.divide(
            2 / wavelength * col_wise_scalar_product(line_of_sight, sensor_velocities),
            line_of_sight_norm,
        )
        - doppler_frequency
    )

    if is_scalar:
        return float(result[0])
    return result


def doppler_equation_bistatic_residuals(
    sensor_pos_rx: npt.NDArray[np.floating],
    sensor_pos_tx: npt.NDArray[np.floating],
    sensor_vel_rx: npt.NDArray[np.floating],
    sensor_vel_tx: npt.NDArray[np.floating],
    ground_points: npt.NDArray[np.floating],
    wavelength: float,
    doppler_frequency: float,
) -> float | npt.NDArray[np.floating]:
    """Evaluating doppler equation residual for bistatic sensors.

    Parameters
    ----------
    sensor_pos_rx : npt.NDArray[np.floating]
        sensor rx position, (3,) or (N, 3)
    sensor_pos_tx : npt.NDArray[np.floating]
        sensor tx position, (3,) or (N, 3)
    sensor_vel_rx : npt.NDArray[np.floating]
        sensor rx velocity, (3,) or (N, 3)
    sensor_vel_tx : npt.NDArray[np.floating]
        sensor tx velocity, (3,) or (N, 3)
    ground_points : npt.NDArray[np.floating]
        ground points from direct geocoding solution, (3,) or (N, 3)
    wavelength : float
        carrier signal wavelength in meters
    doppler_frequency : float
        doppler frequency in Hz

    Returns
    -------
    float | npt.NDArray[np.floating]
        doppler equation residual, scalar or (N,)
    """

    los_rx = sensor_pos_rx - ground_points
    los_tx = sensor_pos_tx - ground_points
    los_vel_prod_rx = np.sum(sensor_vel_rx * los_rx, axis=-1)
    los_vel_prod_tx = np.sum(sensor_vel_tx * los_tx, axis=-1)
    distance_rx = np.sqrt(np.sum(los_rx * los_rx, axis=-1))
    distance_tx = np.sqrt(np.sum(los_tx * los_tx, axis=-1))

    doppler_residual_rx, _ = doppler_equation(
        pv_scalar=los_vel_prod_rx,
        los=los_rx,
        sensor_velocity=sensor_vel_rx,
        distance=distance_rx,
        wavelength=wavelength,
        doppler_frequency=doppler_frequency,
    )

    doppler_residual_tx, _ = doppler_equation(
        pv_scalar=los_vel_prod_tx,
        los=los_tx,
        sensor_velocity=sensor_vel_tx,
        distance=distance_tx,
        wavelength=wavelength,
        doppler_frequency=doppler_frequency,
    )

    is_scalar = (
        sensor_pos_rx.ndim == 1
        and sensor_pos_tx.ndim == 1
        and sensor_vel_rx.ndim == 1
        and sensor_vel_tx.ndim == 1
        and ground_points.ndim == 1
    )

    result = doppler_residual_rx + doppler_residual_tx

    if is_scalar:
        return float(result)
    return result


def get_geometric_doppler_centroid(
    sensor_positions: npt.NDArray[np.floating],
    sensor_velocities: npt.NDArray[np.floating],
    ground_points: npt.NDArray[np.floating],
    wavelength: float,
) -> float:
    """Calculating doppler centroid (geometrically) from squint angle.

    Parameters
    ----------
    sensor_positions : npt.NDArray[np.floating]
        sensor positions array, in the form (3,) or (N, 3)
    sensor_velocities : npt.NDArray[np.floating]
        sensor velocities array, in the form (3,) or (N, 3)
    ground_points : npt.NDArray[np.floating]
        ground points array, in the form (3,) or (N, 3)
    wavelength : int
        carrier signal wavelength in meters

    Returns
    -------
    float | npt.NDArray[np.floating]
        doppler centroid in Hz, scalar or with shape (N,)
    """

    # evaluating squint
    squint_angles = get_geometric_squint_angle(
        sensor_positions=sensor_positions,
        sensor_velocities=sensor_velocities,
        ground_points=ground_points,
    )
    sensor_velocity_norm = np.linalg.norm(sensor_velocities, axis=-1)

    doppler_centroid = 2 * sensor_velocity_norm * np.sin(squint_angles) / wavelength

    if np.ndim(doppler_centroid) == 0:
        return float(doppler_centroid)

    return doppler_centroid


def compute_theoretical_doppler_rate(
    trajectory: Trajectory,
    azimuth_time: PreciseDateTime | np.datetime64,
    ground_points: npt.NDArray[np.floating],
    carrier_frequency: float,
) -> float | npt.NDArray[np.floating]:
    """Compute theoretical doppler rate.

    Parameters
    ----------
    trajectory : Trajectory
        sensor trajectory
    azimuth_time : PreciseDateTime | np.datetime64
        azimuth time when to evaluate the doppler rate
    ground_points : npt.NDArray[np.floating]
        ground point coordinates, with shape (3,) or (N, 3)
    carrier_frequency : float
        signal carrier frequency in Hz

    Returns
    -------
    float | npt.NDArray[np.floating]
        theoretical doppler rate in Hz/s, scalar or with shape (N,) if multiple ground points are provided
    """
    sensor_position = trajectory.position(azimuth_time)
    sensor_velocity = trajectory.velocity(azimuth_time)
    sensor_acceleration = trajectory.acceleration(azimuth_time)

    los = sensor_position - ground_points
    los_norm = np.linalg.norm(los, axis=-1)

    wavelength = speed_of_light / carrier_frequency
    v_norm_sq = np.linalg.norm(sensor_velocity) ** 2

    los_dot_acc = np.sum(los * sensor_acceleration, axis=-1)
    los_dot_vel = np.sum(los * sensor_velocity, axis=-1)

    result = -2.0 / wavelength / los_norm * (v_norm_sq + los_dot_acc - (los_dot_vel / los_norm) ** 2)

    if np.ndim(result) == 0:
        return float(result)
    return result


def compute_steering_doppler_frequency(
    trajectory: Trajectory,
    azimuth_time: PreciseDateTime | np.datetime64,
    az_mid_burst_time: PreciseDateTime | np.datetime64,
    doppler_rate: float,
    az_steering_rate: float,
    carrier_frequency: float,
) -> float:
    """Compute doppler frequency related to the antenna electrical steering.

    Parameters
    ----------
    trajectory : Trajectory
        sensor trajectory
    azimuth_time : PreciseDateTime | np.datetime64
        azimuth time at which compute the steering frequency
    az_mid_burst_time : PreciseDateTime | np.datetime64
        azimuth mid burst time
    doppler_rate : float
        sensor doppler rate in Hz/s
    az_steering_rate : float
        azimuth steering rate in rad/s
    carrier_frequency : float
        signal carrier frequency

    Returns
    -------
    float
        steering doppler frequency in Hz
    """
    sat_vel_norm = np.linalg.norm(trajectory.velocity(azimuth_time))
    # azimuth steering rate conversion from rad/s to Hz/s
    az_steering_rate_hz_s = 2 * sat_vel_norm / (speed_of_light / carrier_frequency) * az_steering_rate
    # antenna modulation rate
    antenna_modulation_rate = -doppler_rate * az_steering_rate_hz_s / (az_steering_rate_hz_s - doppler_rate)

    return antenna_modulation_rate * (azimuth_time - az_mid_burst_time)
