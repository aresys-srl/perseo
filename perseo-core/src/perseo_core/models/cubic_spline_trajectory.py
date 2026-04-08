# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Cubic Spline Trajectory
-----------------------
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from perseo_core.models.types import CoordinatesArrayType, ExtendedDatetimeArrayType, ExtendedDatetimeType


class CubicSplineTrajectory:
    """Trajectory based on a Cubic Spline interpolator"""

    def __init__(
        self, times: ExtendedDatetimeArrayType, positions: CoordinatesArrayType, velocities: CoordinatesArrayType
    ) -> None:
        """Trajectory object creation depending on positions, velocities and time axis.
        Time axis can be specified as relative or absolute (actual dates), while positions and velocities must be
        specified as (N, 3) arrays of floats.

        This object is based on a scipy CubicSpline interpolator.
        **Extrapolation** outside time validity boundaries **is not allowed**.

        Parameters
        ----------
        times : ExtendedDatetimeArrayType
            time axis as numpy array of shape (N,), it can be relative or absolute
        positions : CoordinatesArrayType
            positions as numpy array of shape (N, 3), with coordinates being x, y, z
        velocities : CoordinatesArrayType
            velocities as numpy array of shape (N, 3), with coordinates being x, y, z
        """
        assert positions.shape[1] == 3 and positions.ndim == 2
        assert velocities.shape[1] == 3 and velocities.ndim == 2
        self._positions = positions
        self._velocities = velocities
        self._times = times
        self._time_origin = times[0]
        self._last_time = times[-1]
        self._time_relative = np.array(times - times[0], dtype=float)
        self._domain = (self._time_origin, self._last_time)
        self._interpolator = self._create_interpolator()

    @property
    def positions(self) -> np.ndarray:
        """Accessing trajectory positions vector"""
        return self._positions

    @property
    def velocities(self) -> np.ndarray:
        """Accessing trajectory velocities vector"""
        return self._velocities

    @property
    def times(self) -> np.ndarray:
        """Accessing trajectory times vector"""
        return self._times

    @property
    def domain(self) -> np.ndarray:
        """Trajectory time domain"""
        return self._domain

    def _create_interpolator(self) -> CubicSpline:
        """Generating the Cubic Spline interpolator from given inputs.

        Returns
        -------
        CubicSpline
            CubicSpline scipy interpolator object
        """
        return CubicSpline(
            x=self._time_relative,
            y=self._positions,
            bc_type=((1, self._velocities[0, :]), (1, self._velocities[-1, :])),
            extrapolate=False,
        )

    def _check_time_validity(self, times: ExtendedDatetimeType | ExtendedDatetimeArrayType) -> None:
        """Check input times validity with respect to the time validity boundaries.

        Parameters
        ----------
        times : ExtendedDatetimeType | ExtendedDatetimeArrayType
            input times at which interpolate the trajectory

        Raises
        ------
        RuntimeError
            if one or more of the input times is not inside the time boundaries of trajectory definition
        """
        if np.any(times < self._time_origin) or np.any(times > self._last_time):
            raise RuntimeError("One (or more) of the input times is outside of trajectory time boundaries")

    def position(self, time: ExtendedDatetimeType | ExtendedDatetimeArrayType) -> np.ndarray:
        """Evaluate x, y, z interpolated values at given times.

        Time values must be specified with a type that is the same as the construction "times" array used to build the
        interpolator.

        Parameters
        ----------
        time : ExtendedDatetimeType | ExtendedDatetimeArrayType
            time compatible with the time type used for building the trajectory interpolator

        Returns
        -------
        np.ndarray
            interpolated values for x, y, z at given times
        """
        self._check_time_validity(time)
        # TODO: this difference here can be problematic when using datetime64 as input, creating timedelta64 instead of
        # TODO: floats but an additional check should be performed to ensure that the input is a datetime64 array and
        # TODO: not a timedelta64 array
        relative_times = time - self._time_origin
        return self._interpolator(relative_times, 0, extrapolate=False)

    def velocity(self, time: ExtendedDatetimeType | ExtendedDatetimeArrayType) -> np.ndarray:
        """Evaluate x, y, z first derivatives (vx, vy, vz) interpolated values at given times.

        Time values must be specified with a type that is the same as the construction "times" array used to build the
        interpolator.

        Parameters
        ----------
        time : ExtendedDatetimeType | ExtendedDatetimeArrayType
            time compatible with the time type used for building the trajectory interpolator

        Returns
        -------
        np.ndarray
            interpolated first derivatives values for x, y, z at given times
        """
        self._check_time_validity(time)
        relative_times = time - self._time_origin
        # TODO: same here
        return self._interpolator(relative_times, 1, extrapolate=False)

    def acceleration(self, time: ExtendedDatetimeType | ExtendedDatetimeArrayType) -> np.ndarray:
        """Evaluate x, y, z second derivatives (ax, ay, az) interpolated values at given times.

        Time values must be specified with a type that is the same as the construction "times" array used to build the
        interpolator.

        Parameters
        ----------
        time : ExtendedDatetimeType | ExtendedDatetimeArrayType
            time compatible with the time type used for building the trajectory interpolator

        Returns
        -------
        np.ndarray
            interpolated second derivatives values for x, y, z at given times
        """
        self._check_time_validity(time)
        relative_times = time - self._time_origin
        # TODO: same here
        return self._interpolator(relative_times, 2, extrapolate=False)
