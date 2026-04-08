# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Cubic Spline Trajectory
-----------------------
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline

from perseo_core.models.trajectory import Trajectory

T = TypeVar("T", bound=np.generic)


class CubicSplineTrajectory(Trajectory[T]):
    """Trajectory based on a Cubic Spline interpolator"""

    def __init__(
        self, times: npt.NDArray[T], positions: npt.NDArray[np.floating], velocities: npt.NDArray[np.floating]
    ) -> None:
        """Create a CubicSplineTrajectory from state vectors: times, positions and velocities.

        Times must be of type T, either dates or floats.

        Positions and velocities must be specified as (N, 3) arrays of floats.

        CubicSplineTrajectory wraps scipy CubicSpline interpolator.

        **Extrapolation** outside trajectory domain **is not allowed**.

        Parameters
        ----------
        times : npt.NDArray[T]
            time axis as numpy array of shape (N,)
        positions : npt.NDArray[np.floating]
            positions as numpy array of shape (N, 3), with coordinates being x, y, z
        velocities : npt.NDArray[np.floating]
            velocities as numpy array of shape (N, 3), with coordinates being x, y, z
        """
        if times.ndim != 1:
            raise ValueError("Times must be a 1D array")

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("Positions must be a 2D array with shape (N, 3)")

        if velocities.ndim != 2 or velocities.shape[1] != 3:
            raise ValueError("Velocities must be a 2D array with shape (N, 3)")

        if not (len(times) == positions.shape[0] == velocities.shape[0]):
            raise ValueError("Times, positions and velocities must have the same number of samples")

        self._positions = positions
        self._velocities = velocities
        self._times = times

        self._domain: tuple[T, T] = (times[0], times[-1])

        bc_left = (1, velocities[0, :])
        bc_right = (1, velocities[-1, :])
        boundary_conditions = (bc_left, bc_right)

        self._interpolator = CubicSpline(
            x=np.array(times - times[0], dtype=float),
            y=self._positions,
            bc_type=boundary_conditions,
            extrapolate=False,
        )

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

    def _relative_times(self, time: T | npt.NDArray[T]) -> float | npt.NDArray[np.floating]:
        """Retrieve relative times from given absolute times."""
        if not self._is_time_valid(time):
            raise RuntimeError("One (or more) of the input times is outside of trajectory time boundaries")
        return time - self.domain[0]

    def position(self, time: T | npt.NDArray[T]) -> npt.NDArray[np.floating]:
        """Evaluate x, y, z position at given time.

        Parameters
        ----------
        time : T | npt.NDArray[T]
            time of the same type of the initialization times axis

        Returns
        -------
        np.ndarray
            position with shape (3,) or (N, 3) with coordinates being x, y, z
        """
        relative_times = self._relative_times(time)
        return self._interpolator(relative_times, 0, extrapolate=False)

    def velocity(self, time: T | npt.NDArray[T]) -> npt.NDArray[np.floating]:
        """Evaluate vx, vy, vz velocity at given time.

        Parameters
        ----------
        time : T | npt.NDArray[T]
            time of the same type of the initialization times axis

        Returns
        -------
        np.ndarray
            velocity with shape (3,) or (N, 3) with coordinates being x, y, z
        """
        relative_times = self._relative_times(time)
        return self._interpolator(relative_times, 1, extrapolate=False)

    def acceleration(self, time: T | npt.NDArray[T]) -> npt.NDArray[np.floating]:
        """Evaluate ax, ay, az acceleration at given time.

        Parameters
        ----------
        time : T | npt.NDArray[T]
            time of the same type of the initialization times axis

        Returns
        -------
        np.ndarray
            acceleration with shape (3,) or (N, 3) with coordinates being x, y, z
        """
        relative_times = self._relative_times(time)
        return self._interpolator(relative_times, 2, extrapolate=False)
