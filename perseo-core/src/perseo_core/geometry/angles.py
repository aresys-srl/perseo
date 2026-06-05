# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""High-level computation of incidence, look, squint, and other angles from sensor trajectories."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from perseo_core.geometry.angles_core import (
    compute_incidence_angles_core,
    compute_look_angles_core,
)
from perseo_core.geometry.geocoding.direct_geocoding import (
    SensorLookDirection,
    direct_geocoding_monostatic,
)

if TYPE_CHECKING:
    from perseo_core.geometry.navigation.trajectory import Trajectory
    from perseo_core.timing.precise_datetime import PreciseDateTime


def compute_incidence_angles(
    trajectory: Trajectory,
    azimuth_time: PreciseDateTime | np.datetime64,
    range_times: float | npt.NDArray[np.floating],
    look_direction: SensorLookDirection,
    geodetic_altitude: float | None = None,
    doppler_frequencies: float | npt.NDArray[np.floating] | None = None,
    carrier_wavelength: float | None = None,
    *,
    radians: bool = True,
) -> float | npt.NDArray[np.floating]:
    """Compute incidence angles in radians/degrees from sensor trajectory.

    Parameters
    ----------
    trajectory : Trajectory
        sensor trajectory
    azimuth_time : PreciseDateTime | np.datetime64
        azimuth time at which compute the incidence angles corresponding to the input range times
    range_times : float | npt.NDArray[np.floating]
        range times where to compute the incidence angles, a float or a (N,) array
    look_direction : SensorLookDirection
        sensor looking side where to perform geocoding, "RIGHT" or "LEFT"
    geodetic_altitude : float | None, optional
        the altitude over wgs84, if None is set to 0, by default None
    doppler_frequencies : float | npt.NDArray[np.floating] | None, optional
        frequency_doppler_centroid value or set of values, one for each range time, if None is set to 0, by default None
    carrier_wavelength : float | None, optional
        carrier signal wavelength, if None is set to 1, by default None
    radians : bool, optional
        output angles are expressed in radians if this flag is set to True (default), if False are expressed in degrees

    Returns
    -------
    float | npt.NDArray[np.floating]
        incidence angles in radians/degrees corresponding to the input range times at the given azimuth time

    """
    sensor_position = trajectory.position(azimuth_time)
    sensor_velocity = trajectory.velocity(azimuth_time)
    ground_points = direct_geocoding_monostatic(
        sensor_positions=sensor_position,
        sensor_velocities=sensor_velocity,
        range_times=range_times,
        doppler_frequencies=(doppler_frequencies if doppler_frequencies is not None else 0),
        wavelength=carrier_wavelength if carrier_wavelength is not None else 1,
        look_direction=look_direction,
        altitude=geodetic_altitude if geodetic_altitude is not None else 0,
    )
    angles = compute_incidence_angles_core(sensor_positions=sensor_position, ground_points=ground_points)
    if not radians:
        return np.rad2deg(angles)
    return angles


def compute_look_angles(
    trajectory: Trajectory,
    azimuth_time: PreciseDateTime | np.datetime64,
    range_times: float | npt.NDArray[np.floating],
    look_direction: SensorLookDirection,
    geodetic_altitude: float | None = None,
    doppler_frequencies: float | npt.NDArray[np.floating] | None = None,
    carrier_wavelength: float | None = None,
    *,
    radians: bool = True,
) -> float | npt.NDArray[np.floating]:
    """Compute look angles in radians/degrees from sensor trajectory.

    Parameters
    ----------
    trajectory : Trajectory
        sensor trajectory
    azimuth_time : PreciseDateTime | np.datetime64
        azimuth time at which compute the look a angles corresponding to the input range times
    range_times : float | npt.NDArray[np.floating]
        range times where to compute the look angles, a float or a (N,) array
    look_direction : SensorLookDirection
        sensor looking side where to perform geocoding, "RIGHT" or "LEFT"
    geodetic_altitude : float | None, optional
        the altitude over wgs84, if None is set to 0, by default None,
    doppler_frequencies : float | npt.NDArray[np.floating] | None, optional
        doppler frequencies values, if None is set to 0, by default None
    carrier_wavelength : float | None, optional
        carrier signal wavelength, if None is set to 1, by default None
    radians : bool, optional
        output angles are expressed in radians if this flag is set to True (default), if False are expressed in degrees

    Returns
    -------
    float | npt.NDArray[np.floating]
        look angles in radians/degrees corresponding to the input range times at the given azimuth time

    """
    sensor_position = trajectory.position(azimuth_time)
    sensor_velocity = trajectory.velocity(azimuth_time)
    ground_points = direct_geocoding_monostatic(
        sensor_positions=sensor_position,
        sensor_velocities=sensor_velocity,
        range_times=range_times,
        doppler_frequencies=(doppler_frequencies if doppler_frequencies is not None else 0),
        wavelength=carrier_wavelength if carrier_wavelength is not None else 1,
        look_direction=look_direction,
        altitude=geodetic_altitude if geodetic_altitude is not None else 0,
    )
    angles = compute_look_angles_core(sensor_positions=sensor_position, ground_points=ground_points)
    if not radians:
        return np.rad2deg(angles)
    return angles


__all__ = ["compute_incidence_angles", "compute_look_angles"]
