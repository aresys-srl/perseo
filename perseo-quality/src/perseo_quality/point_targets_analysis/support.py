# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Point Target Analysis support functionalities"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from perseo_core.geometry import get_geometric_squint_angle
from perseo_core.geometry.geocoding import (
    direct_geocoding_monostatic,
    inverse_geocoding_monostatic,
    inverse_geocoding_monostatic_with_attitude,
)
from perseo_core.geometry.navigation import Trajectory
from perseo_core.geometry.pointing import Attitude
from perseo_core.timing import PreciseDateTime
from scipy.constants import speed_of_light

from perseo_quality.core.generic_dataclasses import SideLobesDirections
from perseo_quality.io.quality_input_protocol import ChannelData


def get_squint_angle(
    trajectory: Trajectory, azimuth_times: PreciseDateTime | npt.NDArray, ground_points: npt.NDArray[np.floating]
) -> float | npt.NDArray[np.floating]:
    """Compute squint angle (radians) for given azimuth times and ground points.

    Parameters
    ----------
    trajectory : Trajectory
        trajectory
    azimuth_times : PreciseDateTime | npt.NDArray
        azimuth times at which compute the squint angle
    ground_points : npt.NDArray[np.floating]
        ground points

    Returns
    -------
    float | npt.NDArray[np.floating]
        squint angle (rad)
    """
    sensor_positions = trajectory.position(azimuth_times)
    sensor_velocities = trajectory.velocity(azimuth_times)

    return get_geometric_squint_angle(
        sensor_positions=sensor_positions, sensor_velocities=sensor_velocities, ground_points=ground_points
    )


def squint_to_doppler(
    squint_angles: float | npt.NDArray[np.floating],
    trajectory: Trajectory,
    carrier_frequency: float,
    azimuth_times: PreciseDateTime | npt.NDArray,
) -> float | npt.NDArray[np.floating]:
    """Computing doppler centroid frequency from azimuth time and its corresponding squint angle.

    Parameters
    ----------
    squint_angles : float | npt.NDArray[np.floating]
        squint angle (rad)
    trajectory : Trajectory
        trajectory
    carrier_frequency : float
        carrier frequency (Hz)
    azimuth_times : PreciseDateTime | npt.NDArray
        azimuth times at which compute the doppler centroid

    Returns
    -------
    float | npt.NDArray[np.floating]
        doppler centroid frequency (Hz)
    """

    sensor_velocity = trajectory.velocity(azimuth_times)
    sensor_velocity_norm = np.linalg.norm(sensor_velocity, axis=-1)

    return 2.0 * carrier_frequency / speed_of_light * sensor_velocity_norm * np.sin(squint_angles)


def compute_squint_and_doppler_from_antenna_pointing(
    trajectory: Trajectory,
    attitude: Attitude,
    ground_points: npt.NDArray[np.floating],
    initial_azimuth_time_guess: PreciseDateTime,
    carrier_frequency: float,
) -> tuple[float, float, PreciseDateTime]:
    """Computing squint angle and doppler centroid from attitude information.

    Parameters
    ----------
    trajectory : Trajectory
        trajectory
    attitude : Attitude
        attitude
    ground_points : npt.NDArray[np.floating]
        single ground point, shape ``(3,)``
    initial_azimuth_time_guess : PreciseDateTime
        initial azimuth time guess
    carrier_frequency : float
        carrier frequency (Hz)

    Returns
    -------
    float
        squint angle (rad)
    float
        doppler centroid (Hz)
    PreciseDateTime
        sensor time with doppler
    """
    sensor_time_with_doppler, _ = inverse_geocoding_monostatic_with_attitude(
        trajectory=trajectory,
        attitude=attitude,
        ground_points=ground_points,
        az_initial_time_guesses=initial_azimuth_time_guess,
        doppler_frequencies=0,
        wavelength=speed_of_light / carrier_frequency,
    )
    assert isinstance(sensor_time_with_doppler, PreciseDateTime)

    squint_angle = get_squint_angle(
        trajectory=trajectory,
        azimuth_times=sensor_time_with_doppler,
        ground_points=ground_points,
    )
    doppler_centroid = squint_to_doppler(
        squint_angles=squint_angle,
        trajectory=trajectory,
        carrier_frequency=carrier_frequency,
        azimuth_times=sensor_time_with_doppler,
    )

    squint_angle = float(squint_angle)
    doppler_centroid = float(doppler_centroid)

    return squint_angle, doppler_centroid, sensor_time_with_doppler


def compute_squint_and_doppler_from_polynomials(
    trajectory: Trajectory,
    doppler_centroid_poly,
    ground_points: npt.NDArray[np.floating],
    azimuth_time: PreciseDateTime,
    range_time: float,
    carrier_frequency: float,
) -> tuple[float, float, PreciseDateTime]:
    """Computing squint angle and doppler centroid from doppler centroid polynomial.

    Parameters
    ----------
    trajectory : Trajectory
        trajectory
    doppler_centroid_poly : DopplerCentroidPolynomial
        doppler centroid polynomial
    ground_points : npt.NDArray[np.floating]
        single ground point, shape ``(3,)``
    azimuth_time : PreciseDateTime
        azimuth time at which to compute the output
    range_time : float
        range time at which to compute the output
    carrier_frequency : float
        carrier frequency (Hz)

    Returns
    -------
    float
        squint angle (rad)
    float
        doppler centroid (Hz)
    PreciseDateTime
        sensor time with doppler
    """

    doppler_centroid = float(doppler_centroid_poly.evaluate(azimuth_time=azimuth_time, range_time=range_time))
    sensor_time_with_doppler, _ = inverse_geocoding_monostatic(
        trajectory=trajectory,
        ground_points=ground_points,
        doppler_frequencies=doppler_centroid,
        wavelength=speed_of_light / carrier_frequency,
        az_initial_time_guesses=azimuth_time,
    )
    assert isinstance(sensor_time_with_doppler, PreciseDateTime)

    sensor_velocity = float(np.linalg.norm(trajectory.velocity(azimuth_time)))
    squint_angle = doppler_centroid / (2.0 * sensor_velocity / (speed_of_light / carrier_frequency))

    return squint_angle, doppler_centroid, sensor_time_with_doppler


def compute_side_lobes_directions(
    channel_data: ChannelData,
    azimuth_time: PreciseDateTime,
    range_time: float,
    azimuth_step_m: float,
) -> tuple[SideLobesDirections, float, float]:
    """Computing side lobe directions for squinted data and squint angle.

    Parameters
    ----------
    channel_data : ChannelManager
        ChannelManager instance
    azimuth_time : PreciseDateTime
        azimuth time where to compute the side lobes directions
    range_time : float
        range time where to compute the side lobes directions
    azimuth_step_m : float
        azimuth step in meters

    Returns
    -------
    SideLobesDirections
        range and azimuth cuts angular coefficients in samples
    float
        squint angle (rad)
    float
        doppler centroid (Hz)
    """

    sensor_pos = channel_data.trajectory.position(azimuth_time)
    sensor_vel = channel_data.trajectory.velocity(azimuth_time)

    earth_point_zero_doppler = direct_geocoding_monostatic(
        sensor_positions=sensor_pos,
        sensor_velocities=sensor_vel,
        range_times=range_time,
        doppler_frequencies=0,
        wavelength=1,
        look_direction=channel_data.looking_side.value,
        altitude=0,
    )

    if channel_data.attitude is None and channel_data.doppler_centroid is None:
        # no attitude or doppler centroid provided, returning zero doppler condition
        return (np.inf, 0.0), 0, 0

    if channel_data.attitude is not None:
        # computing side lobes with attitude
        squint_angle, doppler_centroid, sensor_time_with_doppler = compute_squint_and_doppler_from_antenna_pointing(
            trajectory=channel_data.trajectory,
            attitude=channel_data.attitude,
            ground_points=earth_point_zero_doppler,
            initial_azimuth_time_guess=azimuth_time,
            carrier_frequency=channel_data.carrier_frequency,
        )
    else:
        assert channel_data.doppler_centroid is not None
        squint_angle, doppler_centroid, sensor_time_with_doppler = compute_squint_and_doppler_from_polynomials(
            trajectory=channel_data.trajectory,
            doppler_centroid_poly=channel_data.doppler_centroid,
            ground_points=earth_point_zero_doppler,
            azimuth_time=azimuth_time,
            range_time=range_time,
            carrier_frequency=channel_data.carrier_frequency,
        )

    sensor_position_zero_doppler = channel_data.trajectory.position(azimuth_time)
    sensor_position_with_doppler = channel_data.trajectory.position(sensor_time_with_doppler)

    los_zero_doppler = np.squeeze(sensor_position_zero_doppler - earth_point_zero_doppler)
    los_with_doppler = np.squeeze(sensor_position_with_doppler - earth_point_zero_doppler)
    slope = np.sign(doppler_centroid) * np.arctan2(
        np.linalg.norm(np.cross(los_zero_doppler, los_with_doppler)), np.dot(los_zero_doppler, los_with_doppler)
    )

    # evaluating range and azimuth angular coefficients in samples (IRF Rng and Az cuts)
    step_ratio = azimuth_step_m / channel_data.range_step_m
    rng_cut = step_ratio / np.tan(slope)
    az_cut = -np.tan(slope) * step_ratio

    return (rng_cut, az_cut), squint_angle, doppler_centroid
