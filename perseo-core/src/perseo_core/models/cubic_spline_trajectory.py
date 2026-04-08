# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Cubic Spline Trajectory
-----------------------
"""

from __future__ import annotations

from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline

T = TypeVar("T", bound=np.generic)


class CubicSplineTrajectory(Generic[T]):
    """Trajectory based on a Cubic Spline interpolator"""

    def __init__(
        self, times: npt.NDArray[T], positions: npt.NDArray[np.floating], velocities: npt.NDArray[np.floating]
    ) -> None:
        """Trajectory object creation depending on positions, velocities and time axis.
        Time axis can be specified as relative or absolute (actual dates), while positions and velocities must be
        specified as (N, 3) arrays of floats.

        This object is based on a scipy CubicSpline interpolator.
        **Extrapolation** outside time validity boundaries **is not allowed**.

        Parameters
        ----------
        times : npt.NDArray[T]
            time axis as numpy array of shape (N,), it can be relative or absolute
        positions : npt.NDArray[np.floating]
            positions as numpy array of shape (N, 3), with coordinates being x, y, z
        velocities : npt.NDArray[np.floating]
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
    def domain(self) -> tuple[T, T]:
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

    def _check_time_validity(self, times: T | npt.NDArray[T]) -> None:
        """Check input times validity with respect to the time validity boundaries.

        Parameters
        ----------
        times : T | npt.NDArray[T]
            input times at which interpolate the trajectory

        Raises
        ------
        RuntimeError
            if one or more of the input times is not inside the time boundaries of trajectory definition
        """
        if np.any(times < self._time_origin) or np.any(times > self._last_time):
            raise RuntimeError("One (or more) of the input times is outside of trajectory time boundaries")

    def position(self, time: T | npt.NDArray[T]) -> npt.NDArray[np.floating]:
        """Evaluate x, y, z interpolated values at given times.

        Time values must be specified with a type that is the same as the construction "times" array used to build the
        interpolator.

        Parameters
        ----------
        time : T | npt.NDArray[T]
            time compatible with the time type used for building the trajectory interpolator

        Returns
        -------
        np.ndarray
            interpolated values for x, y, z at given times
        """
        self._check_time_validity(time)
        relative_times = time - self._time_origin
        return self._interpolator(relative_times, 0, extrapolate=False)

    def velocity(self, time: T | npt.NDArray[T]) -> npt.NDArray[np.floating]:
        """Evaluate x, y, z first derivatives (vx, vy, vz) interpolated values at given times.

        Time values must be specified with a type that is the same as the construction "times" array used to build the
        interpolator.

        Parameters
        ----------
        time : T | npt.NDArray[T]
            time compatible with the time type used for building the trajectory interpolator

        Returns
        -------
        np.ndarray
            interpolated first derivatives values for x, y, z at given times
        """
        self._check_time_validity(time)
        relative_times = time - self._time_origin
        return self._interpolator(relative_times, 1, extrapolate=False)

    def acceleration(self, time: T | npt.NDArray[T]) -> npt.NDArray[np.floating]:
        """Evaluate x, y, z second derivatives (ax, ay, az) interpolated values at given times.

        Time values must be specified with a type that is the same as the construction "times" array used to build the
        interpolator.

        Parameters
        ----------
        time : T | npt.NDArray[T]
            time compatible with the time type used for building the trajectory interpolator

        Returns
        -------
        np.ndarray
            interpolated second derivatives values for x, y, z at given times
        """
        self._check_time_validity(time)
        relative_times = time - self._time_origin
        return self._interpolator(relative_times, 2, extrapolate=False)
