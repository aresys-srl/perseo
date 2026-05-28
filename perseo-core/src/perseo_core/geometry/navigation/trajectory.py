# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
This module defines the `Trajectory` abstract base class, which establishes the interface
for all trajectory implementations in the PERSEO framework. A trajectory represents the
path of a sensor through space over time, providing position, velocity, and
acceleration vectors at arbitrary time points within its defined domain.

The Trajectory ABC specifies three core evaluation methods that all concrete implementations
must provide:

- `position(time)`: Returns sensor position as (x, y, z) coordinates
- `velocity(time)`: Returns sensor velocity as (vx, vy, vz) components
- `acceleration(time)`: Returns sensor acceleration as (ax, ay, az) components

The abstract interface enables polymorphic use throughout PERSEO geometry computations,
allowing different trajectory representations (e.g., cubic splines, Keplerian orbits,
polynomial fits) to be used interchangeably in geocoding, pointing, and SAR processing.

### Implementation tips

The concrete implementation of this class and its methods must support vectorized evaluation, with scalar times
returning (3,) arrays and array times (N,) returning (N, 3) arrays. The `domain` property defines the valid time
range [start, end]. Input query times *can be checked* to ensure they are within the domain bounds
(using _is_time_valid) to avoid extrapolation outside the time domain.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt

T = TypeVar("T", bound=np.generic)


class Trajectory(ABC, Generic[T]):
    """Trajectory interface"""

    @property
    @abstractmethod
    def domain(self) -> tuple[T, T]:
        """Trajectory time domain as a tuple of [start, end]"""

    def _is_time_valid(self, time: T | npt.NDArray[T]) -> bool:
        """Check if time is within the trajectory domain."""
        start, end = self.domain
        time_array = np.atleast_1d(time)
        return bool(np.all((time_array >= start) & (time_array <= end)))

    @abstractmethod
    def position(self, time: T | npt.NDArray[T]) -> npt.NDArray[np.floating]:
        """Retrieve position at given time.

        Parameters
        ----------
        time : T | npt.NDArray[T]
            evaluation time: scalar or array with shape (N,)

        Returns
        -------
        npt.NDArray[np.floating]
            position with shape (3,) or (N, 3)
        """

    @abstractmethod
    def velocity(self, time: T | npt.NDArray[T]) -> npt.NDArray[np.floating]:
        """Retrieve velocity at given time.

        Parameters
        ----------
        time : T | npt.NDArray[T]
            evaluation time: scalar or array with shape (N,)

        Returns
        -------
        npt.NDArray[np.floating]
            velocity with shape (3,) or (N, 3)
        """

    @abstractmethod
    def acceleration(self, time: T | npt.NDArray[T]) -> npt.NDArray[np.floating]:
        """Retrieve acceleration at given time.

        Parameters
        ----------
        time : T | npt.NDArray[T]
            evaluation time: scalar or array with shape (N,)

        Returns
        -------
        npt.NDArray[np.floating]
            acceleration with shape (3,) or (N, 3)
        """

    def evaluate(
        self, time: T | npt.NDArray[T]
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Evaluate position, velocity and acceleration at given time.

        Parameters
        ----------
        time : T | npt.NDArray[T]
            evaluation time: scalar or array with shape (N,)

        Returns
        -------
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]
            position, velocity and acceleration with shape (3,) or (N, 3)
        """
        return self.position(time), self.velocity(time), self.acceleration(time)
