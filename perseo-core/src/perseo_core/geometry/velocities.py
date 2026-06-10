# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Computation of sensor velocity quantities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from perseo_core.geometry.geocoding.direct import direct_geocoding_with_look_angles

if TYPE_CHECKING:
    from perseo_core.geometry.navigation.trajectory import Trajectory
    from perseo_core.geometry.pointing.reference_frames import ReferenceFrame
    from perseo_core.timing.precise_datetime import PreciseDateTime


def compute_ground_velocity(
    trajectory: Trajectory,
    azimuth_time: PreciseDateTime | np.datetime64,
    look_angles_rad: npt.NDArray[np.floating],
    reference_frame: ReferenceFrame = "ZERODOPPLER",
    geodetic_altitude: float = 0,
    averaging_interval_relative_origin: float = 0,
    averaging_interval_duration: float = 1,
    averaging_interval_num_points: int = 11,
) -> npt.NDArray[np.floating] | float:
    """Numerical computation of the ground velocity [m/s] at given look angles.

    The algorithm is based on the direct geocoding, via look angles, of points at different azimuth times in a
    averaging interval.

    Parameters
    ----------
    trajectory : Trajectory
        sensor trajectory
    azimuth_time : PreciseDateTime | np.datetime64
        azimuth time at which compute the ground velocity
    look_angles_rad : npt.NDArray[np.floating]
        look angles in radians, float or array with shape (N,)
    reference_frame : "GEOCENTRIC", "GEODETIC", "ZERODOPPLER", optional
        the reference frames in which the look angles are intended, by default "ZERODOPPLER"
    geodetic_altitude : float, optional
        altitude of the points over wgs84, by default 0
    averaging_interval_relative_origin : float, optional
        averaging interval starts at time_point + averaging_interval_relative_origin, by default 0
    averaging_interval_duration : float, optional
        total duration of the averaging interval, by default 1.0
    averaging_interval_num_points : int, optional
        number of time points in the averaging interval, by default 11

    Returns
    -------
    npt.NDArray[np.floating] | float
        ground velocity in m/s

    """
    # generating the averaging time interval axis
    relative_averaging_time_axis = np.linspace(
        averaging_interval_relative_origin, averaging_interval_duration, averaging_interval_num_points
    )

    averaging_time_axis = relative_averaging_time_axis + azimuth_time  # type: ignore
    sensor_positions = trajectory.position(averaging_time_axis)
    sensor_velocities = trajectory.velocity(averaging_time_axis)

    # computing ground points at each sensor position/velocity in the selected averaging time interval, for each
    # input look angle
    look_angles = np.atleast_1d(look_angles_rad)
    ground_points = [
        direct_geocoding_with_look_angles(
            sensor_positions=sensor_positions,
            sensor_velocities=sensor_velocities,
            reference_frame=reference_frame,
            look_angles=angle,
            altitude=geodetic_altitude,
        )
        for angle in look_angles
    ]

    # computing ground velocity components (as ground points coordinates diff) for each time interval, for each
    # input look angle, and then computing their norm
    ground_velocities_norm = [np.linalg.norm(np.diff(g, axis=0), axis=-1) for g in ground_points]
    ground_velocities = np.array([np.sum(v, axis=-1) / averaging_interval_duration for v in ground_velocities_norm])

    return ground_velocities if not isinstance(look_angles_rad, float) else ground_velocities[0]


__all__ = ["compute_ground_velocity"]
