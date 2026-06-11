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
from perseo_core.timing import PreciseDateTime
from scipy.constants import speed_of_light

from perseo_quality.core.generic_dataclasses import SideLobesDirections
from perseo_quality.io.quality_input_protocol import ChannelData


def get_squint_angle(
    channel_data: ChannelData, azimuth_time: PreciseDateTime, ground_point: npt.NDArray[np.floating]
) -> float:
    """Compute squint angle (radians) for a given azimuth time and ground point.

    Parameters
    ----------
    channel_data : ChannelManager
        ChannelManager instance
    azimuth_time : PreciseDateTime
        azimuth time at which compute the squint angle
    ground_point : npt.NDArray[np.floating]
        ground point seen by the sensor at the provided azimuth time

    Returns
    -------
    float
        squint angle (rad)
    """
    sensor_position = channel_data.trajectory.position(azimuth_time).squeeze()
    sensor_velocity = channel_data.trajectory.velocity(azimuth_time).squeeze()

    return get_geometric_squint_angle(
        sensor_positions=sensor_position, sensor_velocities=sensor_velocity, ground_points=ground_point
    )


def get_doppler_centroid(
    channel_data: ChannelData, azimuth_time: PreciseDateTime, ground_point: npt.NDArray[np.floating]
) -> float:
    """Computing doppler centroid frequency from azimuth time and its corresponding squint angle.

    Parameters
    ----------
    channel_data : ChannelManager
        ChannelManager instance
    azimuth_time : PreciseDateTime
        azimuth time at which compute doppler centroid frequency
    ground_point : npt.NDArray[np.floating]
        ground point seen by the sensor at the provided azimuth time

    Returns
    -------
    float
        doppler centroid frequency (Hz)
    """

    squint_angle = get_squint_angle(channel_data=channel_data, azimuth_time=azimuth_time, ground_point=ground_point)
    sensor_velocity = channel_data.trajectory.velocity(azimuth_time).squeeze()
    sensor_velocity_norm = np.linalg.norm(sensor_velocity, axis=-1)
    carrier_freq = channel_data.carrier_frequency / speed_of_light

    return 2.0 * carrier_freq * sensor_velocity_norm * np.sin(squint_angle)


def compute_side_lobes_directions(
    channel_data: ChannelData,
    peak_azimuth_time: PreciseDateTime,
    peak_range_time: float,
    azimuth_step_m: float,
) -> tuple[SideLobesDirections, float, float]:
    """Computing side lobe directions for squinted data and squint angle.

    Parameters
    ----------
    channel_data : ChannelManager
        ChannelManager instance
    peak_azimuth_time : PreciseDateTime
        azimuth time corresponding to the point target signal peak
    peak_range_time : float
        range time corresponding to the point target signal peak

    Returns
    -------
    SideLobesDirections
        range and azimuth cuts angular coefficients in samples
    float
        squint angle (rad)
    float
        doppler centroid (Hz)
    """

    sensor_pos = channel_data.trajectory.position(peak_azimuth_time)
    sensor_vel = channel_data.trajectory.velocity(peak_azimuth_time)

    earth_point_zero_doppler = direct_geocoding_monostatic(
        sensor_positions=sensor_pos,
        sensor_velocities=sensor_vel,
        range_times=peak_range_time,
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
        sensor_time_with_doppler, _ = inverse_geocoding_monostatic_with_attitude(
            trajectory=channel_data.trajectory,
            attitude=channel_data.attitude,
            ground_points=earth_point_zero_doppler,
            doppler_frequencies=0,
            wavelength=1,
            initial_guesses=peak_azimuth_time,
        )

        # computing squint angle and doppler centroid
        squint_angle = get_squint_angle(
            channel_data=channel_data, azimuth_time=sensor_time_with_doppler, ground_point=earth_point_zero_doppler
        )
        doppler_centroid = get_doppler_centroid(
            channel_data=channel_data, azimuth_time=sensor_time_with_doppler, ground_point=earth_point_zero_doppler
        )

    elif channel_data.doppler_centroid is not None:
        # computing side lobes with doppler
        doppler_centroid = channel_data.doppler_centroid.evaluate(
            azimuth_time=peak_azimuth_time, range_time=peak_range_time
        )
        sensor_time_with_doppler, _ = inverse_geocoding_monostatic(
            trajectory=channel_data.trajectory,
            ground_points=earth_point_zero_doppler,
            doppler_frequencies=doppler_centroid,
            wavelength=speed_of_light / channel_data.carrier_frequency,
            az_initial_time_guesses=peak_azimuth_time,
        )
        sat_velocity = np.linalg.norm(channel_data.trajectory.velocity(peak_azimuth_time))
        squint_angle = doppler_centroid / (2 * sat_velocity / (speed_of_light / channel_data.carrier_frequency))

    sensor_position_zero_doppler = channel_data.trajectory.position(peak_azimuth_time).T
    sensor_position_with_doppler = channel_data.trajectory.position(sensor_time_with_doppler).T

    los_zd = np.squeeze(sensor_position_zero_doppler - earth_point_zero_doppler)
    los_hd = np.squeeze(sensor_position_with_doppler - earth_point_zero_doppler)
    slope = np.sign(doppler_centroid) * np.arctan2(np.linalg.norm(np.cross(los_zd, los_hd)), np.dot(los_zd, los_hd))

    # evaluating range and azimuth angular coefficients in samples (IRF Rng and Az cuts)
    step_ratio = azimuth_step_m / channel_data.range_step_m
    rng_cut = step_ratio / np.tan(slope)
    az_cut = -np.tan(slope) * step_ratio

    return (rng_cut, az_cut), squint_angle, doppler_centroid
