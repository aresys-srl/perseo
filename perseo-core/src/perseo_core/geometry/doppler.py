# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Geometry - Doppler-related Utilities
------------------------------------
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.constants import speed_of_light

from perseo_core.geometry.angles import get_geometric_squint_angle
from perseo_core.models.protocols import TwiceDifferentiable3DCurve
from perseo_core.models.types import ExtendedDatetimeType


# TODO: this is defined also in direct_geocoding_core as _doppler_equation, duplicated to avoid circular import
def doppler_equation(
    wavelength: float,
    pv_scalar: float,
    distance: float,
    frequency_doppler_centroid: float,
    sensor_velocity: npt.NDArray[np.floating],
    los: np.ndarray,
) -> tuple[float, npt.NDArray[np.floating]]:
    """Doppler equation solver.

    Parameters
    ----------
    wavelength : float
        carrier signal wavelength
    pv_scalar : float
        scalar product between sensor velocity and line of sight
    distance : float
        ground point - sensor distance
    frequency_doppler_centroid : float
        frequency doppler centroid
    sensor_velocity : npt.NDArray[np.floating]
        sensor velocity
    los : np.ndarray
        line of sight

    Returns
    -------
    float
        doppler equation solution
    npt.NDArray[np.floating]
        doppler equation gradient
    """

    c_factor = 2.0 / wavelength / distance
    doppler_equation = c_factor * pv_scalar + frequency_doppler_centroid
    norm_pv = pv_scalar / distance**2
    grad_doppler_equation = (c_factor * (-sensor_velocity + (norm_pv * los.T).T).T).T
    return doppler_equation, grad_doppler_equation


def doppler_equation_monostatic_residuals(
    ground_point: npt.NDArray[np.floating],
    sensor_positions: npt.NDArray[np.floating],
    sensor_velocities: npt.NDArray[np.floating],
    frequency_doppler_centroid: float,
    wavelength: float,
) -> npt.NDArray[np.floating]:
    """Evaluate SAR doppler equation residual, assuming monostatic approximation.

    *Doppler Equation*

    .. math::

        f_{doppler} = \\frac{2}{\\lambda}  \\frac{v_{sensor} \\cdot LOS}{\\|LOS\\|}

    *Doppler Equation Residual*

    .. math::

        Residual = \\frac{2}{\\lambda}  \\frac{v_{sensor} \\cdot LOS}{\\|LOS\\|} - f_{doppler}

    where LOS is defined as the line of sight, a.k.a. the position difference between the ground point and the sensor
    positions.

    Parameters
    ----------
    ground_point : npt.NDArray[np.floating]
        ground point in ECEF coordinates, with shape (3,)
    sensor_positions : npt.NDArray[np.floating]
        sensor positions, with shape (N, 3)
    sensor_velocities : npt.NDArray[np.floating]
        sensor velocities, with shape (N, 3)
    frequency_doppler_centroid : float
        frequency doppler centroid
    wavelength : float
        signal carrier wavelength

    Returns
    -------
    npt.NDArray[np.floating]
        doppler equation residual (Hz) for each input sensor position, (N,)
    """

    if sensor_positions.ndim > 2 or sensor_positions.shape[-1] != 3:
        raise ValueError(f"sensor_positions has invalid shape: {sensor_positions.shape}, it should be (3,) or (N, 3)")

    if sensor_velocities.ndim > 2 or sensor_velocities.shape[-1] != 3:
        raise ValueError(f"sensor_velocities has invalid shape: {sensor_velocities.shape}, it should be (3,) or (N, 3)")

    line_of_sight = ground_point - sensor_positions
    line_of_sight_norm = np.linalg.norm(line_of_sight, axis=1)

    def col_wise_scalar_product(matrix_a, matrix_b):
        return np.einsum("ij,ij->i", matrix_a, matrix_b)  # Einstein notation -- col wise dot product.

    return (
        np.divide(
            2 / wavelength * col_wise_scalar_product(line_of_sight, sensor_velocities),
            line_of_sight_norm,
        )
        - frequency_doppler_centroid
    )


def doppler_equation_bistatic_residuals(
    sensor_pos_rx: npt.NDArray[np.floating],
    sensor_pos_tx: npt.NDArray[np.floating],
    sensor_vel_rx: npt.NDArray[np.floating],
    sensor_vel_tx: npt.NDArray[np.floating],
    ground_points: npt.NDArray[np.floating],
    wavelength: float,
    doppler_freq: float,
) -> npt.NDArray[np.floating]:
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
        carrier signal wavelength
    doppler_freq : float
        doppler frequency

    Returns
    -------
    npt.NDArray[np.floating]
        doppler equation residual
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
        frequency_doppler_centroid=doppler_freq,
    )

    doppler_residual_tx, _ = doppler_equation(
        pv_scalar=los_vel_prod_tx,
        los=los_tx,
        sensor_velocity=sensor_vel_tx,
        distance=distance_tx,
        wavelength=wavelength,
        frequency_doppler_centroid=doppler_freq,
    )

    return np.array(doppler_residual_rx + doppler_residual_tx)


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
    float
        doppler centroid in Hz
    """

    # evaluating squint
    squint_angles = get_geometric_squint_angle(
        sensor_positions=sensor_positions,
        sensor_velocities=sensor_velocities,
        ground_points=ground_points,
    )
    sensor_velocity_norm = np.linalg.norm(sensor_velocities, axis=-1)

    return 2 * sensor_velocity_norm * np.sin(squint_angles) / wavelength


# TODO: new, add unittest
def compute_theoretical_doppler_rate(
    trajectory: TwiceDifferentiable3DCurve,
    azimuth_time: ExtendedDatetimeType,
    coords: np.ndarray,
    fc_hz: float,
) -> np.ndarray:
    """Compute theoretical doppler rate.

    Parameters
    ----------
    trajectory : TwiceDifferentiable3DCurve
        sensor trajectory
    azimuth_time : ExtendedDatetimeType
        azimuth time when to evaluate the doppler rate
    coords : np.ndarray
        ground point coordinates
    fc_hz : float
        signal carrier frequency

    Returns
    -------
    np.ndarray
        theoretical doppler rate
    """
    sat_pos = trajectory.evaluate(azimuth_time)
    sat_vel = trajectory.evaluate_first_derivatives(azimuth_time)
    sat_acc = trajectory.evaluate_second_derivatives(azimuth_time)

    los = (sat_pos - coords).transpose()
    los_norm = np.linalg.norm(los)

    return (
        -2
        / (speed_of_light / fc_hz)
        / los_norm
        * (np.linalg.norm(sat_vel) ** 2 + float(np.dot(los, sat_acc)) - (float(np.dot(los, sat_vel)) / los_norm) ** 2)
    )


# TODO: new, add unittest
def compute_steering_doppler_frequency(
    trajectory: TwiceDifferentiable3DCurve,
    azimuth_time: ExtendedDatetimeType,
    az_mid_burst_time: ExtendedDatetimeType,
    doppler_rate: float,
    az_steering_rate_rad_s: float,
    fc_hz: float,
) -> float:
    """Compute doppler frequency related to the antenna electrical steering.

    Parameters
    ----------
    trajectory : TwiceDifferentiable3DCurve
        sensor trajectory
    azimuth_time : ExtendedDatetimeType
        azimuth time at which compute the steering frequency
    az_mid_burst_time : ExtendedDatetimeType
        azimuth mid burst time
    doppler_rate : float
        sensor doppler rate
    az_steering_rate_rad_s : float
        azimuth steering rate in rad/s
    fc_hz : float
        signal carrier frequency

    Returns
    -------
    float
        steering doppler frequency
    """
    sat_vel_norm = np.linalg.norm(trajectory.evaluate_first_derivatives(azimuth_time))
    # azimuth steering rate conversion from rad/s to Hz/s
    az_steering_rate_hz_s = 2 * sat_vel_norm / (speed_of_light / fc_hz) * az_steering_rate_rad_s
    # antenna modulation rate
    antenna_modulation_rate = -doppler_rate * az_steering_rate_hz_s / (az_steering_rate_hz_s - doppler_rate)

    return antenna_modulation_rate * (azimuth_time - az_mid_burst_time)
