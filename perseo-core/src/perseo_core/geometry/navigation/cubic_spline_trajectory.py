# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Cubic Spline Trajectory module.

This module provides the `CubicSplineTrajectory` class, a concrete implementation of the
`Trajectory` abstract base class that uses scipy's CubicSpline interpolator for continuous trajectory representation.

The CubicSplineTrajectory creates a twice-differentiable cubic spline that interpolates
between recorded position and velocity state vectors. Extrapolation outside the time domain
is not allowed.

The trajectory is initialized from three arrays of equal length:

- `times`: Time axis (N,), either floats (relative times) or PreciseDateTime objects (absolute times)
- `positions`: Position state vectors (N, 3) as [x, y, z] coordinates
- `velocities`: Velocity state vectors (N, 3) as [vx, vy, vz] components
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline

from perseo_core.geometry.navigation.trajectory import Trajectory

T = TypeVar("T", bound=np.generic)


class CubicSplineTrajectory(Trajectory[T]):
    """Trajectory based on a Cubic Spline interpolator."""

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
            msg = "Times must be a 1D array"
            raise ValueError(msg)

        if positions.ndim != 2 or positions.shape[1] != 3:
            msg = "Positions must be a 2D array with shape (N, 3)"
            raise ValueError(msg)

        if velocities.ndim != 2 or velocities.shape[1] != 3:
            msg = "Velocities must be a 2D array with shape (N, 3)"
            raise ValueError(msg)

        if not (len(times) == positions.shape[0] == velocities.shape[0]):
            msg = "Times, positions and velocities must have the same number of samples"
            raise ValueError(msg)

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
            bc_type=boundary_conditions,  # type: ignore
            extrapolate=False,
        )

    @property
    def positions(self) -> np.ndarray:
        """Accessing trajectory positions vector."""
        return self._positions

    @property
    def velocities(self) -> np.ndarray:
        """Accessing trajectory velocities vector."""
        return self._velocities

    @property
    def times(self) -> np.ndarray:
        """Accessing trajectory times vector."""
        return self._times

    @property
    def domain(self) -> tuple[T, T]:
        """Trajectory time domain."""
        return self._domain

    def _relative_times(self, time: T | npt.NDArray[T]) -> float | npt.NDArray[np.floating]:
        """Retrieve relative times from given absolute times."""
        if not self._is_time_valid(time):
            msg = "One (or more) of the input times is outside of trajectory time boundaries"
            raise RuntimeError(msg)
        return time - self.domain[0]  # type: ignore

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


__all__ = ["CubicSplineTrajectory"]
